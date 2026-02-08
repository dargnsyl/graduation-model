import torch
import torch.nn


class FullyConnectedOutput(torch.nn.Module):
    def __init__(self, embed_dim, input_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 32),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, embed_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(p=0.1),
        )
        self.norm = torch.nn.LayerNorm(normalized_shape=embed_dim, elementwise_affine=True)

    def forward(self, x):
        x = self.norm(x)
        out = self.fc(x)
        return out


def _compute_shortest_dist(adj, max_hop):
    # adj: [B, N, N] boolean
    b, n, _ = adj.shape
    device = adj.device
    dist = torch.full((b, n, n), float("inf"), device=device)
    dist[:, torch.arange(n), torch.arange(n)] = 0.0

    reach = adj.clone()
    for hop in range(1, max_hop + 1):
        newly = reach & dist.isinf()
        dist[newly] = float(hop)
        reach = (reach.float() @ adj.float()) > 0
    return dist


def attention(Q, K, V, mask=None, topo_gamma=1.0, topo_tau=2.0, topo_max_hop=6, topo_eps=1e-6):
    # Q,K,V: [B, H, N, D]
    l = Q.shape[2]
    num_head = Q.shape[1]
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))
    score /= (Q.shape[-1] ** 0.5)

    topo_reg_loss = torch.tensor(0.0, device=score.device)

    if mask is not None:
        fc_bias = torch.abs(mask)
        fc_bias = fc_bias.unsqueeze(1).expand(-1, num_head, -1, -1)
        score = score + fc_bias

        adj = (mask > 0)
        dist = _compute_shortest_dist(adj, topo_max_hop)
        dist = dist.clamp_max(topo_max_hop + 1)
        decay = torch.exp(-topo_gamma * torch.relu(dist - topo_tau))
        topo_bias = torch.log(decay + topo_eps).unsqueeze(1).expand(-1, num_head, -1, -1)
        score = score + topo_bias

        attn = torch.softmax(score, dim=-1)
        dist_penalty = torch.relu(dist - topo_tau).unsqueeze(1)
        topo_reg_loss = (attn * dist_penalty).mean()
    else:
        attn = torch.softmax(score, dim=-1)

    x = torch.matmul(attn, V)
    x = x.permute(0, 2, 1, 3).reshape(-1, l, num_head * Q.shape[3])
    return x, topo_reg_loss


class MultiHead(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim, topo_gamma=1.0, topo_tau=2.0, topo_max_hop=6, topo_eps=1e-6):
        super().__init__()
        self.fc_Q = torch.nn.Linear(input_dim, 32)
        self.fc_K = torch.nn.Linear(input_dim, 32)
        self.fc_V = torch.nn.Linear(input_dim, 32)

        self.num_head = num_head
        self.out_fc = torch.nn.Linear(32, embed_dim)

        self.norm = torch.nn.LayerNorm(normalized_shape=input_dim, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.1)

        self.topo_gamma = topo_gamma
        self.topo_tau = topo_tau
        self.topo_max_hop = topo_max_hop
        self.topo_eps = topo_eps
        self.topo_reg_loss = torch.tensor(0.0)

    def forward(self, Q, K, V, mask=None):
        b = Q.shape[0]
        length = Q.shape[1]

        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)

        Q = Q.reshape(b, length, self.num_head, -1).permute(0, 2, 1, 3)
        K = K.reshape(b, length, self.num_head, -1).permute(0, 2, 1, 3)
        V = V.reshape(b, length, self.num_head, -1).permute(0, 2, 1, 3)

        score, topo_reg_loss = attention(
            Q, K, V, mask,
            topo_gamma=self.topo_gamma,
            topo_tau=self.topo_tau,
            topo_max_hop=self.topo_max_hop,
            topo_eps=self.topo_eps,
        )
        self.topo_reg_loss = topo_reg_loss
        score = self.dropout(self.out_fc(score))
        return score


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim, topo_gamma=1.0, topo_tau=2.0, topo_max_hop=6, topo_eps=1e-6):
        super(EncoderLayer, self).__init__()
        self.mh = MultiHead(input_dim, num_head, embed_dim, topo_gamma, topo_tau, topo_max_hop, topo_eps)
        self.fc = FullyConnectedOutput(embed_dim, input_dim)

    def forward(self, x, mask=None):
        score = self.mh(x, x, x, mask)
        out = self.fc(score)
        return out


class FCEncoder(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim, topo_gamma=1.0, topo_tau=2.0, topo_max_hop=6, topo_eps=1e-6):
        super(FCEncoder, self).__init__()
        self.layer = EncoderLayer(input_dim, num_head, embed_dim, topo_gamma, topo_tau, topo_max_hop, topo_eps)
        self.topo_reg_loss = torch.tensor(0.0)

    def forward(self, x, mask=None):
        x = self.layer(x, mask)
        self.topo_reg_loss = self.layer.mh.topo_reg_loss
        return x
