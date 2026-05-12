import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def cosdis(x1,x2):
    return (1-torch.cosine_similarity(x1,x2,dim=-1))/2

def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

def compute_HG(g_views):
    m = g_views.shape[0]
    n = g_views.shape[1]
    c = g_views.shape[2]
    H_G = 0
    for i in range(n):
        for j in range(c):
            prob = (1 / m) * torch.sum(g_views[:,i,j])
            H_G += prob * torch.log(prob)
    return H_G


def compute_gaussian_kernel(features, sigma=1.0):


    batch_size = features.size(0)
    feat_sq = torch.sum(features ** 2, dim=1, keepdim=True)  # [batch_size, 1]
    dist_matrix = feat_sq + feat_sq.t() - 2 * torch.mm(features, features.t())
    K = torch.exp(-dist_matrix / (2 * sigma ** 2))
    return K

class DDC(nn.Module):
    def __init__(self, num_clusters, lambda_d1=1.0, lambda_d2=1.0, lambda_d3=1.0):
        super(DDC, self).__init__()
        self.num_clusters = num_clusters
        self.lambda_d1 = lambda_d1
        self.lambda_d2 = lambda_d2
        self.lambda_d3 = lambda_d3

    def forward(self, G, K, B=None):
        if B is None:
            B = G

        d1 = self._compute_d1(G, K)
        d2 = self._compute_d2(G, B, K)
        d3 = self._compute_d3(G)
        total_loss = self.lambda_d1 * d1 + self.lambda_d2 * d2 + self.lambda_d3 * d3
        loss_components = {
            'D1': d1,
            'D2': d2,
            'D3': d3,
            'total': total_loss
        }

        return total_loss, loss_components

    def _compute_d1(self, G, K):
        batch_size, num_clusters = G.size()
        d1 = torch.tensor(0.0, device=G.device)

        for i in range(num_clusters - 1):
            for j in range(i + 1, num_clusters):
                delta_i = G[:, i].unsqueeze(1)  # [batch_size, 1]
                delta_j = G[:, j].unsqueeze(1)

                numerator = torch.mm(torch.mm(delta_i.t(), K), delta_j)
                denominator = torch.sqrt(
                    torch.mm(torch.mm(delta_i.t(), K), delta_i) *
                    torch.mm(torch.mm(delta_j.t(), K), delta_j)
                )

                if denominator.item() > 1e-8:
                    d1 += numerator.item()/ denominator.item()

        d1 = (1.0 / num_clusters) * d1
        return d1.squeeze()

    def _compute_d2(self, G, B, K):
        batch_size, num_clusters = G.size()
        d2 = torch.tensor(0.0, device=G.device)

        for i in range(num_clusters - 1):
            for j in range(i + 1, num_clusters):
                lambda_i = B[:, i].unsqueeze(1)  # [batch_size, 1]
                lambda_j = B[:, j].unsqueeze(1)

                numerator = torch.mm(torch.mm(lambda_i.t(), K), lambda_j)
                denominator = torch.sqrt(
                    torch.mm(torch.mm(lambda_i.t(), K), lambda_i) *
                    torch.mm(torch.mm(lambda_j.t(), K), lambda_j)
                )

                if denominator.item() > 1e-8:
                    d2 += numerator.item() / denominator.item()

        d2 = (1.0 / num_clusters) * d2
        return d2.squeeze()

    def _compute_d3(self, G):
        G = torch.nn.functional.normalize(G, dim=0, p=2)
        batch_size, num_clusters = G.size()
        G_T_G = torch.mm(G.t(), G)
        mask = torch.triu(torch.ones(num_clusters, num_clusters, device=G.device), diagonal=1)
        d3 = torch.sum((G_T_G * mask) ** 2)
        return d3


class MyLoss(nn.Module):
    def __init__(self,classes_num):
        super(MyLoss, self).__init__()
        self.similarity = nn.CosineSimilarity(dim=0)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.t = 1
        self.tripletloss = nn.TripletMarginWithDistanceLoss(margin=1.0,distance_function=cosdis)
        self.classes_num = classes_num
        self.ddc = DDC(classes_num, lambda_d1=1.0, lambda_d2=1.0, lambda_d3=1.0)

    def weighted_wmse_loss(self,input, target, weight, reduction='mean'):
        if weight is None:
            weight = torch.ones(input[0].shape[0],len(input)).to(input[0].device)
            # weight = np.ones((input[0].shape[0],len(input)))
        if isinstance(input,list):
            loss = [0]*len(input)
            for i in range(len(input)):
                loss[i] = torch.mean(weight[:,i:i+1].mul(target[i] - input[i]) ** 2)
            loss = torch.stack(loss,0)
        else:
            loss = (weight.unsqueeze(-1).mul(target - input)) ** 2

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        return loss

    def spc(self,y):
        c = self.classes_num
        loss = 0
        for i in range(c):
            for j in range(c):
                if i != j:
                    up = y[:, i].unsqueeze(0)  @ y[:, i].unsqueeze(1)
                    down = torch.sqrt(y[:, i].unsqueeze(0)  @ y[:, i].unsqueeze(1) @ y[:, j].unsqueeze(0) @ y[:, j].unsqueeze(1))
                    if down != 0:
                        loss += up / down
        return loss/c


    def L_lc(self,Q,tau_1=1,view_number=2):
        L_ccl = 0
        for i in range(view_number):
            for j in range(view_number):
                if i != j:
                    g_i = Q[:, i].t()
                    g_j = Q[:, j].t()
                    up = torch.exp(torch.cosine_similarity(g_i, g_j, dim=1) / tau_1)
                    d_1 = 0
                    d_2 = 0
                    for k in range(g_i.shape[0]):
                        d_1 += torch.exp(torch.cosine_similarity(g_i[k], g_i, dim=1) / tau_1)
                        d_2 += torch.exp(torch.cosine_similarity(g_j[k], g_j, dim=1) / tau_1)
                    down = d_1 + d_2
                    L_ccl += -1 * torch.sum(torch.log(up / down), dim=0) / g_i.shape[0]
        L_ccl = 0.5 * L_ccl - compute_HG(Q)
        return L_ccl

    def l_ddc(self,Q,Z):
        K = compute_gaussian_kernel(Z)
        total_loss, loss_components = self.ddc(Q,K)
        return total_loss










