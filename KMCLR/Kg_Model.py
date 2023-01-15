import Kg_Par
import torch
from Kg_Data import BasicDataset
from torch import nn
from GAT import GAT
import torch.nn.functional as F
from utils import _L2_loss_mean

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError

class Model(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset, kg_dataset):
        super(Model, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.kg_dataset = kg_dataset
        self.__init_weight()
        self.gat = GAT(self.latent_dim, self.latent_dim, dropout=0.4, alpha=0.2).train()


    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.num_entities = self.kg_dataset.entity_count
        self.num_relations = self.kg_dataset.relation_count
        print("user:{}, item:{}, entity:{}".format(self.num_users,
                                                   self.num_items,
                                                   self.num_entities))
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        self.embedding_user = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.emb_item_list = nn.ModuleList([
            torch.nn.Embedding(self.num_items, self.latent_dim),
            torch.nn.Embedding(self.num_items, self.latent_dim)
        ])
        self.emb_entity_list = nn.ModuleList([
            nn.Embedding(self.num_entities + 1, self.latent_dim),
            nn.Embedding(self.num_entities + 1, self.latent_dim)
        ])
        self.emb_relation_list = nn.ModuleList([
            nn.Embedding(self.num_relations + 1, self.latent_dim),
            nn.Embedding(self.num_relations + 1, self.latent_dim)
        ])

        for i in range(2):
            nn.init.normal_(self.emb_item_list[i].weight, std=0.1)
            nn.init.normal_(self.emb_entity_list[i].weight, std=0.1)
            nn.init.normal_(self.emb_relation_list[i].weight, std=0.1)

        self.transR_W = nn.Parameter(torch.Tensor(self.num_relations + 1, self.latent_dim, self.latent_dim))
        self.TATEC_W = nn.Parameter(torch.Tensor(self.num_relations + 1, self.latent_dim, self.latent_dim))

        nn.init.xavier_uniform_(self.transR_W, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.TATEC_W, gain=nn.init.calculate_gain('relu'))

        self.W_R = nn.Parameter(
            torch.Tensor(self.num_relations, self.latent_dim, self.latent_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))
        nn.init.normal_(self.embedding_user.weight, std=0.1)

        self.co_user_score = nn.Linear(self.latent_dim, 1)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(
            self.num_items)


    def cal_item_embedding_from_kg(self, kg: dict = None, index=0):
        if kg is None:
            kg = self.kg_dict

        return self.cal_item_embedding_rgat(kg, index)


    def cal_item_embedding_rgat(self, kg: dict, index):
        item_embs = self.emb_item_list[index](
            torch.IntTensor(list(kg.keys())).to(
                Kg_Par.device))
        item_entities = torch.stack(list(
            kg.values()))
        item_relations = torch.stack(list(self.item2relations.values()))
        entity_embs = self.emb_entity_list[index](
            item_entities)
        relation_embs = self.emb_relation_list[index](
            item_relations)
        padding_mask = torch.where(item_entities != self.num_entities,
                                   torch.ones_like(item_entities),
                                   torch.zeros_like(item_entities)).float()
        return self.gat.forward_relation(item_embs, entity_embs, relation_embs,
                                         padding_mask)


    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.embedding_user(users)

        pos_emb_ego0 = self.emb_item_list[0](pos_items)
        pos_emb_ego1 = self.emb_item_list[1](pos_items)
        neg_emb_ego0 = self.emb_item_list[0](neg_items)
        neg_emb_ego1 = self.emb_item_list[1](neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego0, pos_emb_ego1, neg_emb_ego0, neg_emb_ego1

    def getAll(self):
        all_users, all_items = self.computer()
        return all_users, all_items


    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, pos_emb_ego0,
         pos_emb_ego1, neg_emb_ego0, neg_emb_ego1) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) + pos_emb_ego0.norm(2).pow(2) + pos_emb_ego1.norm(2).pow(2)
                              + neg_emb_ego0.norm(2).pow(2) + neg_emb_ego1.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        return loss, reg_loss

    def computer(self):
        users_emb = self.embedding_user.weight

        items_emb0 = self.cal_item_embedding_from_kg(index=0)
        items_emb1 = self.cal_item_embedding_from_kg(index=1)

        items_emb = (items_emb0 + items_emb1) / 2

        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def view_computer_all(self, g_droped, index):
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(index=index)
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def calc_kg_loss_transR(self, h, r, pos_t, neg_t, index):
        r_embed = self.emb_relation_list[index](r).unsqueeze(-1)
        h_embed = self.emb_item_list[index](h).unsqueeze(-1)
        pos_t_embed = self.emb_entity_list[index](pos_t).unsqueeze(-1)
        neg_t_embed = self.emb_entity_list[index](neg_t).unsqueeze(-1)

        r_matrix = self.transR_W[r]
        h_embed = torch.matmul(r_matrix, h_embed)
        pos_t_embed = torch.matmul(r_matrix, pos_t_embed)
        neg_t_embed = torch.matmul(r_matrix, neg_t_embed)

        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2),
                              dim=1)
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2),
                              dim=1)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)
        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(
            r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed) + torch.norm(self.transR_W)

        loss = kg_loss + 1e-3 * l2_loss

        return loss

    def calc_kg_loss_TATEC(self, h, r, pos_t, neg_t, index):
        r_embed = self.emb_relation_list[index](r).unsqueeze(-1)
        h_embed = self.emb_item_list[index](h).unsqueeze(-1)
        pos_t_embed = self.emb_entity_list[index](pos_t).unsqueeze(-1)
        neg_t_embed = self.emb_entity_list[index](neg_t).unsqueeze(-1)

        r_matrix = self.TATEC_W[r]
        pos_mrt = torch.matmul(r_matrix, pos_t_embed)
        neg_mrt = torch.matmul(r_matrix, neg_t_embed)

        pos_hmrt = torch.sum(h_embed * pos_mrt, dim=1)
        neg_hmrt = torch.sum(h_embed * neg_mrt, dim=1)

        hr = torch.sum(h_embed * r_embed, dim=1)
        pos_tr = torch.sum(pos_t_embed * r_embed, dim=1)
        neg_tr = torch.sum(neg_t_embed * r_embed, dim=1)

        pos_ht = torch.sum(h_embed * pos_t_embed, dim=1)
        neg_ht = torch.sum(h_embed * neg_t_embed, dim=1)

        pos_score = pos_hmrt + hr + pos_tr + pos_ht
        neg_score = neg_hmrt + hr + neg_tr + neg_ht

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)
        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(
            r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed) + torch.norm(self.TATEC_W)

        loss = kg_loss + 1e-3 * l2_loss

        return loss
