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
                mean_strength = block.mean(dim=(1, 2))
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

#intra_mask按主对角线划分为8个大小不等的子矩阵，每个子矩阵对应一个子网，内部全为true。不在这8个子矩阵内的元素对应子网间连接，设为false

        intra_adjacency = adjacency_matrix * intra_mask.unsqueeze(0)#原来的功能连接矩阵只保留子网内连接

        # Compute subnetwork-level connectivity
        inter_adjacency = self.get_subnetwork_matrix(adjacency_matrix, subnetwork_ends)

        # Add channel dimension for consistency
        intra_adjacency = torch.unsqueeze(intra_adjacency, -1)#[16,200,200]子网内
        inter_adjacency = torch.unsqueeze(inter_adjacency, -1)#[16,8,8]子网间
        adjacency_matrix = torch.unsqueeze(adjacency_matrix, -1)#[16,200,200]原邻接矩阵

        return intra_adjacency, inter_adjacency, adjacency_matrix




class TokenAdditiveRFF(nn.Module):

    def __init__(self, token_dim=8, num_tokens=8, rff_dim=128, sigma=1.0):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.rff_dim = rff_dim
        self.sigma = sigma

        for g in range(num_tokens):
            W = torch.randn(token_dim, rff_dim) / sigma
            b = 2 * math.pi * torch.rand(rff_dim)
            self.register_buffer(f"rff_W_{g}", W)
            self.register_buffer(f"rff_b_{g}", b)

    def forward(self, tokens):#每个子网的特征向量做随机傅里叶特征映射
        feats = []#保存每个子网映射后的特征
        scale = math.sqrt(2.0 / self.rff_dim)#缩放因子
        for g in range(self.num_tokens):#num_tokens==8
            W = getattr(self, f"rff_W_{g}")#取出第g个子网的随机矩阵W和随机偏置项b
            b = getattr(self, f"rff_b_{g}")
            proj = tokens[:, g, :] @ W + b
            z = scale * torch.cos(proj)#z是第g个子网映射后的特征向量，由原来的8维映射到64维
            feats.append(z)
        return torch.cat(feats, dim=-1)


class BayesianLinearClassifier(nn.Module):#没用上不用看

    def __init__(self, feat_dim, num_classes=2, prior_var=1.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.prior_var = prior_var

        self.W_mu = nn.Parameter(torch.zeros(num_classes, feat_dim))
        self.W_logvar = nn.Parameter(torch.zeros(num_classes, feat_dim))
        self.b_mu = nn.Parameter(torch.zeros(num_classes))
        self.b_logvar = nn.Parameter(torch.zeros(num_classes))

    def forward(self, z, mc_samples=3):
        if mc_samples < 1:
            mc_samples = 1
        eps_w = torch.randn(mc_samples, self.num_classes, self.feat_dim, device=z.device)
        eps_b = torch.randn(mc_samples, self.num_classes, device=z.device)
        W = self.W_mu + torch.exp(0.5 * self.W_logvar) * eps_w
        b = self.b_mu + torch.exp(0.5 * self.b_logvar) * eps_b
        logits = torch.einsum('bd,scd->sbc', z, W) + b.unsqueeze(1)
        return logits.mean(dim=0)

    def kl_divergence(self):
        var_w = torch.exp(self.W_logvar)
        var_b = torch.exp(self.b_logvar)
        kl_w = 0.5 * ((var_w + self.W_mu ** 2) / self.prior_var - 1.0 - self.W_logvar + math.log(self.prior_var)).sum()
        kl_b = 0.5 * ((var_b + self.b_mu ** 2) / self.prior_var - 1.0 - self.b_logvar + math.log(self.prior_var)).sum()
        return kl_w + kl_b


class SubnetTokenDKLBayesHead(nn.Module):#没用上不用看

    def __init__(self, num_tokens=8, token_dim=8, rff_dim=128, num_classes=2):
        super().__init__()
        self.rff = TokenAdditiveRFF(token_dim=token_dim, num_tokens=num_tokens, rff_dim=rff_dim, sigma=1.0)
        feat_dim = num_tokens * rff_dim
        self.bayes = BayesianLinearClassifier(feat_dim=feat_dim, num_classes=num_classes)

    def forward(self, tokens, mc_samples=3):
        z = self.rff(tokens)
        logits = self.bayes(z, mc_samples=mc_samples)
        kl = self.bayes.kl_divergence()
        return logits, kl





class AdditiveKernelRegHead(nn.Module):

    def __init__(self, num_tokens=8, token_dim=8, rff_dim=64):
        super().__init__()
        self.rff = TokenAdditiveRFF(token_dim=token_dim, num_tokens=num_tokens, rff_dim=rff_dim, sigma=1.0)
        self.num_tokens = num_tokens
        self.rff_dim = rff_dim

    def forward(self, tokens):
        z = self.rff(tokens)
        B = z.shape[0]
        z = z.view(B, self.num_tokens, self.rff_dim)#[16,8,64]，8个子网，每个子网64维特征

        zn = F.normalize(z, p=2, dim=-1, eps=1e-8)#子网特征单位化
        sim = torch.matmul(zn, zn.transpose(-1, -2))#计算任意两个子网特征相似度，sim维度是[16,8,8]，0表示正交，1表示相同，-1表示相反
        eye = torch.eye(self.num_tokens, device=z.device).unsqueeze(0)#单位阵
        offdiag = (sim - eye).abs()#把sim的主对角线置0，其余位置取绝对值
        loss_ortho = offdiag.mean()#正交损失，不同子网的特征差异越大则这一项越小

        loss_prior = (z ** 2).mean()#L2正则项，为了让各子网特征向量的模长不要过大
        return loss_ortho + 0.01 * loss_prior




class GatedMoEClassifier(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes=2,
        num_experts=3,
        shared_dim=256,
        expert_hidden=128,
        dropout=0.3,
        temperature=1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature

        self.shared_stem = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, num_classes),
            )
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(shared_dim, num_experts)

    def forward(self, x):
        x = self.shared_stem(x)#[16,1600]变为[16,256]，对应共享特征提取层
        gate_logits = self.gate(x) / max(self.temperature, 1e-6)#[16,4]
        gate_probs = torch.softmax(gate_logits, dim=-1)#[16,4]，对应4个分类头的权重

        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)#[16,4,2]，四个分类头的输出结果
        logits = torch.sum(gate_probs.unsqueeze(-1) * expert_logits, dim=1)#[16,2]，根据每个分类头的权重，加权求和

        mean_gate = gate_probs.mean(dim=0)
        uniform = torch.full_like(mean_gate, 1.0 / self.num_experts)
        balance_loss = torch.mean((mean_gate - uniform) ** 2)#对应论文中的平衡损失
        entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=1).mean()#样本级门控熵约束
        return logits, balance_loss, entropy, gate_probs


class CrossGCNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360, moe_config=None):
        super().__init__()
        self.roi_num = roi_num
        self.subnetwork_ends = [41, 70, 91, 110, 130, 137, 158, 200]
        moe_config = moe_config or {}

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

        self.moe_classifier = GatedMoEClassifier(
            input_dim=8 * roi_num,
            num_classes=2,
            num_experts=moe_config.get("num_experts", 3),
            shared_dim=moe_config.get("shared_dim", 256),
            expert_hidden=moe_config.get("expert_hidden", 128),
            dropout=moe_config.get("dropout", 0.3),
            temperature=moe_config.get("temperature", 1.0),
        )
        self.moe_balance_loss = torch.tensor(0.0)
        self.moe_entropy_loss = torch.tensor(0.0)
        self.gate_probs = None

    def subnetwork_pool_tokens(self, roi_feat, subnetwork_ends):
        return self.average_subnetwork_features(roi_feat, subnetwork_ends)

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
        propagated_features = torch.zeros_like(node_features)#子网特征由[16,8,200]扩展到[16,200,200]，方法在以下for循环�?
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
        x = self.gcn2(x)#[16,200,8]
        x = self.bn3(x)#[16,200,8]

        tokens = self.subnetwork_pool_tokens(x, self.subnetwork_ends)
        self.subnet_tokens = tokens.detach()

        # Classifier
        x = x.view(batch_size, -1)#后两个维度合并得到[16,1600]
        logits, balance_loss, entropy_loss, gate_probs = self.moe_classifier(x)#MOE分类头
        self.moe_balance_loss = balance_loss
        self.moe_entropy_loss = entropy_loss
        self.gate_probs = gate_probs.detach()
        return logits#其实就是3个全连接层，把特征维度从1600�?56�?2�?


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
                embed_dim=model_config['embedding_size'],
                topo_gamma=model_config.get('topo_gamma', 1.0),
                topo_tau=model_config.get('topo_tau', 2.0),
                topo_max_hop=model_config.get('topo_max_hop', 6),
                topo_eps=model_config.get('topo_eps', 1e-6)
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

        self.predictor = CrossGCNPredictor(
            node_feature_dim,
            roi_num=roi_num,
            moe_config=model_config.get("moe", {}),
        )

        self.additive_reg_head = AdditiveKernelRegHead(num_tokens=8, token_dim=8, rff_dim=model_config.get("rff_dim", 64))
        self.additive_kernel_loss = torch.tensor(0.0)
        self.moe_balance_loss = torch.tensor(0.0)
        self.moe_entropy_loss = torch.tensor(0.0)



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
        embeddings = self.feature_extractor(time_series, node_features)#完成注意力计算和特征维度变换，[16,200,8]
        self.topo_reg_loss = getattr(self.feature_extractor, 'topo_reg_loss', torch.tensor(0.0))#feature_extractor就是Encoder
        embeddings = F.softmax(embeddings, dim=-1)#[16,200,8]，对应论文中的X_A

        # Generate adjacency matrices
        intra_adjacency, inter_adjacency, full_adjacency = self.graph_generator(
            embeddings, self.subnetwork_ends
        )#self.graph_generator对应CrossEmbed2GraphByProduct

        # Remove channel dimension
        full_adjacency = full_adjacency[:, :, :, 0]#[16,200,200]原邻接矩阵
        intra_adjacency = intra_adjacency[:, :, :, 0]#[16,200,200]子网内
        inter_adjacency = inter_adjacency[:, :, :, 0]#[16,8,8]子网间

        # Compute edge variance regularization
        batch_size = full_adjacency.shape[0]
        edge_variance = torch.mean(torch.var(full_adjacency.reshape((batch_size, -1)), dim=1))#将全连接矩阵展平后先算方差后取平均

        # Make prediction
        prediction = self.predictor(
            full_adjacency,
            intra_adjacency,
            inter_adjacency,
            node_features
        )
        self.moe_balance_loss = getattr(
            self.predictor, "moe_balance_loss", torch.tensor(0.0, device=prediction.device)
        )
        self.moe_entropy_loss = getattr(
            self.predictor, "moe_entropy_loss", torch.tensor(0.0, device=prediction.device)
        )

        self.readout = getattr(self.predictor, "readout", None)
        tokens = getattr(self.predictor, "subnet_tokens", None)#[16,8,8]，8个子网的提取的特征向量
        if tokens is not None:
            self.additive_kernel_loss = self.additive_reg_head(tokens.detach())
        else:
            self.additive_kernel_loss = torch.tensor(0.0, device=prediction.device)

        return prediction, full_adjacency, edge_variance
