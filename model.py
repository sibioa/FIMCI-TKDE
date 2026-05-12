import torch
import torch.nn as nn
import torch.nn.init as init
import os
# from Layers import EncoderLayer, DecoderLayer
# from Embed import Embedder, PositionalEncoder

import copy
import math

from sklearn.cluster import KMeans
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def setEmbedingModel(d_list, d_out):
    return nn.ModuleList([nn.Linear(d, d_out) for d in d_list])


def setReEmbedingModel(d_list, d_out):
    return nn.ModuleList([nn.Linear(d_out, d) for d in d_list])


class Mlp(nn.Module):
    """ Transformer Feed-Forward Block """

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.2):
        super(Mlp, self).__init__()
        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout1:
            out = self.dropout2(out)
        return out


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # scores shape is [bs heads view view]
    if mask is not None:
        mask = mask.unsqueeze(1).float()
        mask = mask.unsqueeze(-1).matmul(mask.unsqueeze(-2))  # mask shape is [bs 1 view view]
        scores = scores.masked_fill(mask == 0, -1e9)  # mask invalid view

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model/h
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.2):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout_1(F.relu(self.linear_1(x)))
        x = self.dropout_2(self.linear_2(x))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)

        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
                                           src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = Parameter(self.pe[:, :seq_len])
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads, dropout)

    def forward(self, src, src_mask):
        e_outputs = self.encoder(src, src_mask)
        return e_outputs


class TransformerWoDecoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(d_model, N, heads, dropout)
        self.decoder = Decoder(d_model, N, heads, dropout)

    def forward(self, src, src_mask=None):
        e_outputs = self.encoder(src, None)
        d_output = self.decoder(src, e_outputs, src_mask, src_mask)
        return e_outputs, d_output


def softmax(x):
    stacked_x = torch.stack(x)
    return torch.nn.functional.softmax(stacked_x, dim=0)


def similarity_function(x, y):
    return torch.norm(x - y, p=2) / 100


def fusion(X, w):
    h_plus = torch.sum(X, dim=1) / X.shape[1]
    alpha = []
    for i in range(X.shape[1]):
        s_values = []
        tem = X[:, i, :]
        s_values.append(similarity_function(tem, h_plus))
        exp_s_values = torch.exp(torch.stack(s_values))
        alpha.append(w[i] * exp_s_values / torch.sum(exp_s_values))
    return softmax(alpha)


class client(nn.Module):
    def __init__(self, d_model, d_list):
        super(client, self).__init__()
        self.re_embeddinglayers = nn.Linear(d_model, d_list).to('cuda:0')
        self.embeddinglayers = nn.Linear(d_list, d_model).to('cuda:0')

    def forward(self, x):
        x = self.embeddinglayers(x)
        return x

    def re_embedding(self, x, x_b, alpha):
        x_bar = self.re_embeddinglayers(x_b)
        x_bar_w = self.re_embeddinglayers(alpha * x)
        return x_bar, x_bar_w


class server(nn.Module):
    def __init__(self, d_model, n_layers, heads, dropout, classes_num, view):
        super(server, self).__init__()
        self.ETrans = Transformer(d_model, n_layers, heads, dropout)
        self.DTrans_1 = TransformerWoDecoder(d_model, n_layers, heads, dropout)
        self.DTrans_2 = TransformerWoDecoder(d_model, n_layers, heads, dropout)
        self.weight = nn.Parameter(torch.full((view,), 1 / view))
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=32),
            nn.Linear(32, 8),
            nn.LayerNorm(normalized_shape=8),
            nn.Linear(8, classes_num))

    def forward(self, x, mask):
        view_num = len(x)
        x = torch.stack(x, dim=1)
        x = self.ETrans(x, mask)
        x, x_b = self.DTrans_1(x, None)
        encX = x
        alpha = fusion(encX, self.weight)
        z = torch.zeros_like(x[:, 1, :])
        for i in range(len(alpha)):
            z += alpha[i] * x[:, i, :]

        r = F.softmax(self.classifier(x), dim=2)

        q = torch.zeros_like(r[:, 0, :]).to('cuda:0')
        for i in range(view_num):
            q += alpha[i] * r[:, i].to('cuda:0')
        return q, x_b, r, z, alpha, encX


class Model(nn.Module):
    def __init__(self, input_len, d_model, n_layers, heads, d_list, classes_num, dropout, view):
        super().__init__()
        self.view = view
        self.client = []
        for i in range(view):
            self.client.append(client(d_model, d_list[i]))
        self.server = server(d_model, n_layers, heads, dropout, classes_num, view).to('cuda:0')

    def forward(self, x, mask=None):
        for i in range(self.view):
            x[i] = self.client[i](x[i])
        q, x_b, r, z, alpha, encX = self.server(x, mask)

        x_bar_list = [None] * self.view
        x_bar_w_list = [None] * self.view
        for i in range(self.view):
            x_bar, x_bar_w = self.client[i].re_embedding(encX[:, i], x_b[:, i], alpha[i])
            x_bar_list[i] = x_bar
            x_bar_w_list[i] = x_bar_w
        return q, x_bar_list, x_bar_w_list, r, z


def get_model(d_list, d_model=768, n_layers=2, heads=4, classes_num=10, dropout=0.2, load_weights=None,
              device=torch.device('cuda:0'), view=2):
    assert d_model % heads == 0
    assert dropout < 1

    model = Model(len(d_list), d_model, n_layers, heads, d_list, classes_num, dropout, view)

    if load_weights is not None:
        print("loading pretrained weights...")
        # model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    model = model.to(device)

    return model


class EarlyStopping:
    def __init__(self, args, patience=7, verbose=False, delta=0.01):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.args = args

    def __call__(self, val_loss, model, result):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, result)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, result)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, result):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.args.save_path, 'model.pth'))
        self.val_loss_min = val_loss
        self.result = result



