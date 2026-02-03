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

    def get_subnetwork_matrix(self, adjacency_matrix, subnetwork_ends):
        subnetwork_starts = [0] + subnetwork_ends[:-1]#包含8个子网起始节点的序号[0, 41, 70, 91, 110, 130, 137, 158]
        num_subnetworks = len(subnetwork_ends)#8个子网[41, 70, 91, 110, 130, 137, 158, 200]
        batch_size = adjacency_matrix.shape[0]#16
        subnetwork_matrix = torch.zeros((batch_size, num_subnetworks, num_subnetworks),
                                        device=adjacency_matrix.device)#[16,8,8]

        for i in range(num_subnetworks):
            for j in range(i, num_subnetworks):
                block = adjacency_matrix[:,
                        subnetwork_starts[i]:subnetwork_ends[i],
                        subnetwork_starts[j]:subnetwork_ends[j]]#取出子网络i和子网络j之间的邻接矩阵
                mean_strength = block.mean(dim=(1, 2))#[16]，对block在第1维和第2维同时取平均，视为子网络i和j之间的连接强度
                subnetwork_matrix[:, i, j] = mean_strength
                subnetwork_matrix[:, j, i] = mean_strength
        return subnetwork_matrix#[16,8,8]

    def forward(self, embeddings, subnetwork_ends):
        # Compute full adjacency matrix
        #embeddings维度[16,200,8]，与自己的转置相乘得到adjacency_matrix，对应论文中X_A与自己的转置相乘得到矩阵A
        adjacency_matrix = torch.einsum('ijk,ipk->ijp', embeddings, embeddings)#[16,200,200]

        roi_count = embeddings.shape[1]#200
        start_index = 0
        device = embeddings.device
        intra_mask = torch.zeros((roi_count, roi_count), dtype=torch.bool, device=device)#[200,200]

        for end_index in subnetwork_ends:
            intra_mask[start_index:end_index, start_index:end_index] = True
            start_index = end_index
#最终intra_mask仍为[200,200]的bool型矩阵，对于subnetwork_ends中的每个数，例如41和70，以intra_mask[0][0]和intra_mask[41][41]为
#左上、右下顶点的子矩阵均为true
#以intra_mask[41][41]和intra_mask[70][70]为左上、右下顶点的子矩阵均为true
#以此类推，intra_mask按主对角线划分为8个大小不等的子矩阵，这些子矩阵元素全为true，不在这8个子矩阵中的全为flase

        intra_adjacency = adjacency_matrix * intra_mask.unsqueeze(0)
#intra_adjacency维度仍为[16,200,200]，根据intra_mask为true的位置不变，false的位置设为0
#这样intra_adjacency也可看作由8个子矩阵构成的对角阵，表示8个子网

        # Compute subnetwork-level connectivity
        inter_adjacency = self.get_subnetwork_matrix(adjacency_matrix, subnetwork_ends)#[16,8,8]

        # Add channel dimension for consistency
        intra_adjacency = torch.unsqueeze(intra_adjacency, -1)
        inter_adjacency = torch.unsqueeze(inter_adjacency, -1)
        adjacency_matrix = torch.unsqueeze(adjacency_matrix, -1)

        return intra_adjacency, inter_adjacency, adjacency_matrix


class CrossGCNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        self.roi_num = roi_num
        self.subnetwork_ends = [41, 70, 91, 110, 130, 137, 158, 200]

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

    def average_subnetwork_features(self, features, subnetwork_ends):
        batch_size, _, feature_dim = features.shape#16,200,200
        num_subnetworks = len(subnetwork_ends)#8
        subnetwork_starts = [0] + subnetwork_ends[:-1]#[0, 41, 70, 91, 110, 130, 137, 158]
        subnetwork_features = torch.zeros((batch_size, num_subnetworks, feature_dim),
                                          device=features.device)#[16,8,200]

        for i in range(num_subnetworks):
            start_idx = subnetwork_starts[i]
            end_idx = subnetwork_ends[i]
            region_features = features[:, start_idx:end_idx, :]
            subnetwork_features[:, i, :] = region_features.mean(dim=1)

        return subnetwork_features#[16,8,200]表示8个子网，每个子网特征是一个长度为200的向量

    def propagate_subnetwork_features(self, subnetwork_features, node_features, subnetwork_ends):
        subnetwork_starts = [0] + subnetwork_ends[:-1]#0, 41, 70, 91, 110, 130, 137, 158
        num_subnetworks = len(subnetwork_ends)#8
        propagated_features = torch.zeros_like(node_features)#子网特征由[16,8,200]扩展到[16,200,200]，方法在以下for循环中
        for i in range(num_subnetworks):
            start_idx = subnetwork_starts[i]
            end_idx = subnetwork_ends[i]
            # Expand subnetwork feature to match region count，subnetwork_features维度是[16,8,200]
            #取出第i个子网的特征，维度是[16,200]，在中间加一个维度[16,1,200]，然后把中间的维度扩展为子网中节点数[16,N,200]得到expanded_features
            expanded_features = subnetwork_features[:, i, :].unsqueeze(1).expand(-1, end_idx - start_idx, -1)
            propagated_features[:, start_idx:end_idx, :] = expanded_features

        # Combine original and propagated features
        return (node_features + propagated_features) / 2 #[16,200,200]

    def forward(self, adjacency_matrix, intra_adjacency, inter_adjacency, node_features):
        batch_size = intra_adjacency.shape[0]#16

        # First propagation layer
        intra_features = torch.einsum('ijk,ijp->ijp', intra_adjacency, node_features)#[16,200,200]，表示子网内特征
        subnetwork_features = self.average_subnetwork_features(node_features, self.subnetwork_ends)#[16,8,200]表示8个子网的特征
        subnetwork_features = torch.einsum('ijk,ijp->ijp', inter_adjacency, subnetwork_features)#[16,8,200]
        x = self.propagate_subnetwork_features(subnetwork_features, intra_features, self.subnetwork_ends)#[16,200,200]
        x = self.gcn(x)#[16,200,200]

        x = x.reshape((batch_size * self.roi_num, -1))#[3200,200]
        x = self.bn1(x)
        x = x.reshape((batch_size, self.roi_num, -1))#[16,200,200]

        # Second propagation layer
        intra_features = torch.einsum('ijk,ijp->ijp', intra_adjacency, x)#[16,200,200]
        subnetwork_features = self.average_subnetwork_features(x, self.subnetwork_ends)
        subnetwork_features = torch.einsum('ijk,ijp->ijp', inter_adjacency, subnetwork_features)#[16,8,200]
        x = self.propagate_subnetwork_features(subnetwork_features, intra_features, self.subnetwork_ends)#[16,200,200]
        x = self.gcn1(x)

        x = x.reshape((batch_size * self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((batch_size, self.roi_num, -1))

        # Third propagation layer
        intra_features = torch.einsum('ijk,ijp->ijp', intra_adjacency, x)
        subnetwork_features = self.average_subnetwork_features(x, self.subnetwork_ends)
        subnetwork_features = torch.einsum('ijk,ijp->ijp', inter_adjacency, subnetwork_features)
        x = self.propagate_subnetwork_features(subnetwork_features, intra_features, self.subnetwork_ends)#[16,200,200]
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

        self.subnetwork_ends = [41, 70, 91, 110, 130, 137, 158, 200]
        self.cluster_order = list(self.node_cluster_map.keys())

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
        embeddings = F.softmax(embeddings, dim=-1)#[16,200,8]，对应论文中的X_A

        # Generate adjacency matrices
        intra_adjacency, inter_adjacency, full_adjacency = self.graph_generator(
            embeddings, self.subnetwork_ends
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
            node_features
        )

        return prediction, full_adjacency, edge_variance#维度分别是[16,2]，[16,200,200]和一个数