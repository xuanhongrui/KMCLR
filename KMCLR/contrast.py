from Kg_Model import Model
from random import random
import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
import Kg_Par


def drop_edge_random(item2entities, p_drop, padding):
    res = dict()
    for item, es in item2entities.items():
        new_es = list()
        for e in es.tolist():
            if (random() > p_drop):
                new_es.append(e)
            else:
                new_es.append(padding)
        res[item] = torch.IntTensor(new_es).to(Kg_Par.device)
    return res


class Contrast(nn.Module):
    def __init__(self, gcn_model, tau=Kg_Par.kgc_temp):
        super(Contrast, self).__init__()
        self.gcn_model: Model = gcn_model
        self.tau = tau

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        if z1.size()[0] == z2.size()[0]:
            return F.cosine_similarity(z1, z2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())

    def pair_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.pair_sim(z1, z2))
        return torch.sum(-torch.log(between_sim.diag() / (between_sim.sum(1) - between_sim.diag())))

    def grace_loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = False, batch_size: int = 0):
        h1 = z1
        h2 = z2
        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            # l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
        return l1

    def get_kg_views(self):
        kg = self.gcn_model.kg_dict
        view1 = drop_edge_random(kg, Kg_Par.kg_p_drop,
                                 self.gcn_model.num_entities)
        view2 = drop_edge_random(kg, Kg_Par.kg_p_drop,
                                 self.gcn_model.num_entities)
        return view1, view2

    def item_kg_stability(self, view1, view2, index):
        kgv1_ro = self.gcn_model.cal_item_embedding_from_kg(view1, index=index)
        kgv2_ro = self.gcn_model.cal_item_embedding_from_kg(view2, index=index)
        sim = self.sim(kgv1_ro, kgv2_ro)
        return sim


    def get_adj_mat(self, tmp_adj):
        adj_mat = tmp_adj + tmp_adj.T

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        coo = adj_matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        g = torch.sparse.FloatTensor(index, data, torch.Size(
            coo.shape)).coalesce().to(Kg_Par.device)
        g.requires_grad = False
        return g

    def ui_batch_drop_weighted(self, item_mask, start, end):
        item_mask = item_mask.cpu().numpy()
        n_nodes = self.gcn_model.num_users + self.gcn_model.num_items
        item_np = self.gcn_model.dataset.trainItem
        user_np = self.gcn_model.dataset.trainUser
        indices = np.where((user_np >= start) & (user_np < end))[0]
        batch_item = item_np[indices]
        batch_user = user_np[indices]

        keep_idx = list()
        for u, i in zip(batch_user, batch_item):
            if item_mask[u - start, i]:
                keep_idx.append([u, i])

        keep_idx = np.array(keep_idx)
        user_np = keep_idx[:, 0]
        item_np = keep_idx[:, 1] + self.gcn_model.num_users
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix(
            (ratings, (user_np, item_np)),
            shape=(n_nodes, n_nodes))
        return tmp_adj

    def get_ui_views_weighted_with_uemb(self, item_stabilities, user_score, start, end, init_view):
        user_item_stabilities = F.softmax(user_score, dim=-1) * item_stabilities
        k = (1 - 0.6) / (user_item_stabilities.max() - user_item_stabilities.min())
        weights = 0.6 + k * (user_item_stabilities - user_item_stabilities.min())
        item_mask = torch.bernoulli(weights).to(torch.bool)
        tmp_adj = self.ui_batch_drop_weighted(item_mask, start, end)
        if init_view != None:
            tmp_adj = init_view + tmp_adj
        return tmp_adj

    def get_ui_kg_view(self, aug_side="both"):
        if aug_side == "ui":
            kgv1, kgv2 = None, None
            kgv3, kgv4 = None, None
        else:
            kgv1, kgv2 = self.get_kg_views()
            kgv3, kgv4 = self.get_kg_views()

        stability1 = self.item_kg_stability(kgv1, kgv2, index=0).to(Kg_Par.device)
        stability2 = self.item_kg_stability(kgv3, kgv4, index=1).to(Kg_Par.device)
        u = self.gcn_model.embedding_user.weight
        i1 = self.gcn_model.emb_item_list[0].weight
        i2 = self.gcn_model.emb_item_list[1].weight

        user_length = self.gcn_model.num_users
        size = 2048
        step = user_length // size + 1
        init_view1, init_view2 = None, None
        for s in range(step):
            start = s * size
            end = (s + 1) * size
            u_i_s1 = u[start:end] @ i1.T
            u_i_s2 = u[start:end] @ i2.T
            uiv1_batch_view = self.get_ui_views_weighted_with_uemb(stability1, u_i_s1, start, end, init_view1)
            uiv2_batch_view = self.get_ui_views_weighted_with_uemb(stability2, u_i_s2, start, end, init_view2)
            init_view1 = uiv1_batch_view
            init_view2 = uiv2_batch_view

        uiv1 = self.get_adj_mat(init_view1)
        uiv2 = self.get_adj_mat(init_view2)

        contrast_views = {
            "uiv1": uiv1,
            "uiv2": uiv2
        }
        return contrast_views

