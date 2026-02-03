import torch.nn
import torch


class FullyConnectedOutput(torch.nn.Module):
    def __init__(self, embed_dim, input_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 32),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, embed_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(p=0.1)
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=embed_dim, elementwise_affine=True)

    def forward(self, x):

        x = self.norm(x)

        out = self.fc(x)

        return out#[16,200,8]


def attention(Q, K, V, mask=None):#同时计算多个头的注意力得分
    l = Q.shape[2]#QKV都是[16,4,200,8]，mask是[16,200,200]，l是200
    num_head = Q.shape[1]#4
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))#QK矩阵乘法
    score /= (Q.shape[-1] ** 0.5)#QK/(√dk)

    if mask is not None:
        mask = torch.abs(mask)#mask[16,200,200]取绝对值
        mask = mask.unsqueeze(1)#变为[16,1,200,200]
        mask = mask.expand(-1, 4, -1, -1)#[16,4,200,200]
        score = score * mask#矩阵对应位置相乘，这部分对应论文中上部分QK/(√dk)与注意力引导项Xfc逐元素相乘

    score = torch.softmax(score, dim=-1)#score[16,4,200,200]
    x = torch.matmul(score, V)#乘矩阵V，得到x[16,4,200,8]

    x = x.permute(0, 2, 1, 3).reshape(-1, l, num_head * Q.shape[3])
    return x    #最终x[16,200,32]


class MultiHead(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim):
        super().__init__()
        self.fc_Q = torch.nn.Linear(input_dim, 32)
        self.fc_K = torch.nn.Linear(input_dim, 32)
        self.fc_V = torch.nn.Linear(input_dim, 32)

        self.num_head = num_head

        self.out_fc = torch.nn.Linear(32, embed_dim)

        self.norm = torch.nn.LayerNorm(normalized_shape=input_dim, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask=None):


        b = Q.shape[0]#batch_size==16
        len = Q.shape[1]#200

        Q = self.norm(Q)#归一化
        K = self.norm(K)
        V = self.norm(V)

        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)#经过全连接层，QKV都是[16,200,32]

        Q = Q.reshape(b, len, self.num_head, -1).permute(0, 2, 1, 3)
        K = K.reshape(b, len, self.num_head, -1).permute(0, 2, 1, 3)
        V = V.reshape(b, len, self.num_head, -1).permute(0, 2, 1, 3)
#维度变换，QKV都是[16,4,200,8]，其实就是把上面的[16,200,32]里面的32拆分为4(num_head)和8
        score = attention(Q, K, V, mask)
        score = self.dropout(self.out_fc(score))

        return score


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim):
        super(EncoderLayer, self).__init__()
        self.mh = MultiHead(input_dim, num_head, embed_dim)
        self.fc = FullyConnectedOutput(embed_dim, input_dim)

    def forward(self, x, mask=None):
        score = self.mh(x, x, x, mask)#输入的x是time_series，mask是node_features
        out = self.fc(score)#[16,200,8]

        return out


class FCEncoder(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim):
        super(FCEncoder, self).__init__()
        self.layer = EncoderLayer(input_dim, num_head, embed_dim)

    def forward(self, x, mask=None):
        x = self.layer(x, mask)#输入的x是time_series，mask是node_features

        return x
