from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import math
from model.Encoder import FCEncoder
import pickle


class CrossEmbed2GraphByProduct(nn.Module):

    def __init__(self, input_dim, roi_num=264):
        super().__init__()

    def get_subnetwork_matrix(self, adjacency_matrix, assignment, eps=1e-6):
        # assignment: [N, K], soft membership of each node to K subnetworks
        # Compute weighted mean connectivity between subnetworks
        # numerator: A^T * adjacency * A
        # adjacency: [B, N, N], assignment: [N, K] -> [B, K, K]
        temp = torch.einsum('bnm,nk->bkm', adjacency_matrix, assignment)
        subnetwork_matrix = torch.einsum('bkm,ml->bkl', temp, assignment)

        weights = assignment.sum(dim=0)  # [K]
        denom = torch.outer(weights, weights).clamp_min(eps)  # [K, K]
        subnetwork_matrix = subnetwork_matrix / denom.unsqueeze(0)
        return subnetwork_matrix

    def forward(self, embeddings, assignment):
        # Compute full adjacency matrix
        #embeddings???[16,200,8]?????????????????djacency_matrix?????????X_A??????????????????A
        adjacency_matrix = torch.einsum('ijk,ipk->ijp', embeddings, embeddings)#[16,200,200]

        # Soft intra-mask: probability that two nodes belong to the same subnetwork
        # intra_mask: [N, N]
        intra_mask = torch.matmul(assignment, assignment.t())
        intra_adjacency = adjacency_matrix * intra_mask.unsqueeze(0)
#intra_adjacency??????[16,200,200]?????ntra_mask??rue?????????false????????
#???intra_adjacency?????????????????????????????????

        # Compute subnetwork-level connectivity
        inter_adjacency = self.get_subnetwork_matrix(adjacency_matrix, assignment)#[16,8,8]

        # Add channel dimension for consistency
        intra_adjacency = torch.unsqueeze(intra_adjacency, -1)
        inter_adjacency = torch.unsqueeze(inter_adjacency, -1)
        adjacency_matrix = torch.unsqueeze(adjacency_matrix, -1)

        return intra_adjacency, inter_adjacency, adjacency_matrix


class CrossGCNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        self.roi_num = roi_num
        # Graph convolution layers
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, roi_num),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(roi_num, roi_num)
        )
        self.bn1 = nn.BatchNorm1d(roi_num)

        self.gcn1 = nn.Sequential(
            nn.Linear(roi_num, roi_num),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = nn.BatchNorm1d(roi_num)

        self.gcn2 = nn.Sequential(
            nn.Linear(roi_num, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = nn.BatchNorm1d(roi_num)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(8 * roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )

    def average_subnetwork_features(self, features, assignment, eps=1e-6):
        # features: [B, N, F], assignment: [N, K]
        weighted_sum = torch.einsum('bnf,nk->bkf', features, assignment)
        weights = assignment.sum(dim=0).clamp_min(eps)  # [K]
        subnetwork_features = weighted_sum / weights.view(1, -1, 1)
        return subnetwork_features#[16,8,200]???8????????????????????????200?????

    def propagate_subnetwork_features(self, subnetwork_features, node_features, assignment):
        # subnetwork_features: [B, K, F], assignment: [N, K]
        propagated_features = torch.einsum('nk,bkf->bnf', assignment, subnetwork_features)
        # Combine original and propagated features
        return (node_features + propagated_features) / 2 #[16,200,200]

    def forward(self, adjacency_matrix, intra_adjacency, inter_adjacency, node_features, assignment):
        batch_size = intra_adjacency.shape[0]#16

        # First propagation layer
        intra_features = torch.einsum('ijk,ijp->ijp', intra_adjacency, node_features)#[16,200,200]，表示子网内特征
        subnetwork_features = self.average_subnetwork_features(node_features, assignment)#[16,8,200]表示8个子网的特征
        subnetwork_features = torch.einsum('ijk,ijp->ijp', inter_adjacency, subnetwork_features)#[16,8,200]
        x = self.propagate_subnetwork_features(subnetwork_features, intra_features, assignment)#[16,200,200]
        x = self.gcn(x)#[16,200,200]

        x = x.reshape((batch_size * self.roi_num, -1))#[3200,200]
        x = self.bn1(x)
        x = x.reshape((batch_size, self.roi_num, -1))#[16,200,200]

        # Second propagation layer
        intra_features = torch.einsum('ijk,ijp->ijp', intra_adjacency, x)#[16,200,200]
        subnetwork_features = self.average_subnetwork_features(x, assignment)
        subnetwork_features = torch.einsum('ijk,ijp->ijp', inter_adjacency, subnetwork_features)#[16,8,200]
        x = self.propagate_subnetwork_features(subnetwork_features, intra_features, assignment)#[16,200,200]
        x = self.gcn1(x)

        x = x.reshape((batch_size * self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((batch_size, self.roi_num, -1))

        # Third propagation layer
        intra_features = torch.einsum('ijk,ijp->ijp', intra_adjacency, x)
        subnetwork_features = self.average_subnetwork_features(x, assignment)
        subnetwork_features = torch.einsum('ijk,ijp->ijp', inter_adjacency, subnetwork_features)
        x = self.propagate_subnetwork_features(subnetwork_features, intra_features, assignment)#[16,200,200]
        x = self.gcn2(x)#[16,200,8]，gcn2里面的两个全连接层把特征维度从200到64到8
        x = self.bn3(x)#[16,200,8]

        # Classifier
        x = x.view(batch_size, -1)#后两个维度合并得到[16,1600]
        return self.classifier(x)#其实就是3个全连接层，把特征维度从1600到256到32到2


class Embed2GraphByLinear(nn.Module):

    def __init__(self, input_dim, roi_num=360):
        super().__init__()

        self.feature_proj = nn.Linear(input_dim * 2, input_dim)
        self.edge_predictor = nn.Linear(input_dim, 1)

        def encode_onehot(labels):
            classes = set(labels)
            class_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
            return np.array(list(map(class_dict.get, labels)), dtype=np.int32)

        # Create receiver and sender matrices
        off_diag = np.ones([roi_num, roi_num])
        rel_rec = encode_onehot(np.where(off_diag)[0])
        rel_send = encode_onehot(np.where(off_diag)[1])

        self.receiver_matrix = torch.FloatTensor(rel_rec).cuda()
        self.sender_matrix = torch.FloatTensor(rel_send).cuda()

    def forward(self, embeddings):
        batch_size, region_count, _ = embeddings.shape

        receivers = torch.matmul(self.receiver_matrix, embeddings)
        senders = torch.matmul(self.sender_matrix, embeddings)

        # Concatenate and predict edges
        edge_features = torch.cat([senders, receivers], dim=2)
        edge_features = torch.relu(self.feature_proj(edge_features))
        edge_scores = self.edge_predictor(edge_features)
        edge_scores = torch.relu(edge_scores)

        # Reshape to adjacency matrix
        adjacency_matrix = edge_scores.reshape(batch_size, region_count, region_count, -1)
        return adjacency_matrix


class DHGFormer(nn.Module):

    def __init__(self, model_config, roi_num=360, node_feature_dim=360, time_series_len=512):
        super().__init__()
        self.graph_generation = model_config['graph_generation']

        # Feature extractor
        if model_config['extractor_type'] == 'transformer':
            self.feature_extractor = FCEncoder(
                input_dim=time_series_len,
                num_head=4,
                embed_dim=model_config['embedding_size']
            )

        # Graph generator
        if self.graph_generation == "linear":
            self.graph_generator = Embed2GraphByLinear(
                model_config['embedding_size'],
                roi_num=roi_num
            )
        elif self.graph_generation == "product":
            self.graph_generator = CrossEmbed2GraphByProduct(
                model_config['embedding_size'],
                roi_num=roi_num
            )

        self.predictor = CrossGCNPredictor(node_feature_dim, roi_num=roi_num)

        # Load node cluster mapping
        with open('./node_clus_map.pickle', 'rb') as f:
            self.node_cluster_map = pickle.load(f)

        self.num_subnetworks = model_config.get('num_subnetworks', 8)
        self.subnetwork_temperature = model_config.get('subnetwork_temperature', 1.0)
        self.subnetwork_ends = model_config.get('subnetwork_ends', [41, 70, 91, 110, 130, 137, 158, 200])
        self.cluster_order = list(self.node_cluster_map.keys())
        self.subnetwork_logits = nn.Parameter(
            self._init_subnetwork_logits(roi_num, self.num_subnetworks, self.subnetwork_ends)
        )
        self.last_assignment_entropy = None

    def _init_subnetwork_logits(self, roi_num, num_subnetworks, subnetwork_ends):
        logits = torch.zeros(roi_num, num_subnetworks)
        if num_subnetworks == len(subnetwork_ends) and subnetwork_ends[-1] == roi_num:
            logits[:] = -2.0
            starts = [0] + subnetwork_ends[:-1]
            for k, (s, e) in enumerate(zip(starts, subnetwork_ends)):
                logits[s:e, k] = 2.0
        return logits

    def get_subnetwork_assignment(self):
        temperature = max(self.subnetwork_temperature, 1e-6)
        return F.softmax(self.subnetwork_logits / temperature, dim=-1)

    def reorder_nodes(self, features, dimension=1):
        """Reorder features according to cluster mapping"""
        return features[:, self.cluster_order, :] if dimension == 1 else \
            features[:, self.cluster_order, :][:, :, self.cluster_order]

    def forward(self, time_series: torch.Tensor, node_features: torch.Tensor):
        # Reorder inputs according to cluster mapping
        time_series = self.reorder_nodes(time_series, dimension=1) #[16,200,100]
        node_features = self.reorder_nodes(node_features, dimension=2)#[16,200,200]

        # Extract features and generate graph
        embeddings = self.feature_extractor(time_series, node_features)#feature_extractor就是Encoder
        embeddings = F.softmax(embeddings, dim=-1)

        assignment = self.get_subnetwork_assignment()
        # Entropy regularization term (can be added to loss externally)
        self.last_assignment_entropy = (-assignment * torch.log(assignment + 1e-8)).sum(dim=-1).mean()#[16,200,8]，对应论文中的X_A

        # Generate adjacency matrices
        intra_adjacency, inter_adjacency, full_adjacency = self.graph_generator(
            embeddings, assignment
        )#self.graph_generator对应CrossEmbed2GraphByProduct

        # Remove channel dimension
        full_adjacency = full_adjacency[:, :, :, 0]#[16,200,200]表示全连接的网络，即论文中第一部分的矩阵A
        intra_adjacency = intra_adjacency[:, :, :, 0]#[16,200,200]可视为8个子矩阵，表示子网络内的连接
        inter_adjacency = inter_adjacency[:, :, :, 0]#[16,8,8]表示子网络间的连接

        # Compute edge variance regularization
        batch_size = full_adjacency.shape[0]#16
        edge_variance = torch.mean(torch.var(full_adjacency.reshape((batch_size, -1)), dim=1))#将全连接矩阵展平后先算方差后取平均

        # Make prediction
        prediction = self.predictor(
            full_adjacency,
            intra_adjacency,
            inter_adjacency,
            node_features,
            assignment
        )

        return prediction, full_adjacency, edge_variance#维度分别是[16,2]，[16,200,200]和一个数