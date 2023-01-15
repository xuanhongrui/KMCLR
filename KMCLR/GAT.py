import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.num_heads = 1
        self.layers = nn.ModuleList([GraphAttentionLayer(nfeat,nhid,dropout=dropout,alpha=alpha,concat=True) for _ in range(self.num_heads)])
        self.out = nn.Linear(nhid * self.num_heads, nhid)

    def forward(self, item_embs, entity_embs, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.out(x, y, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward_relation(self, item_embs, entity_embs, w_r, adj):
        x = F.dropout(entity_embs, self.dropout, training=self.training)
        x = torch.cat([att.forward_relation(item_embs, x, w_r, adj) for att in self.layers ], dim=1)

        x = self.out(x + item_embs)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.fc = nn.Linear(in_features * 3, 1)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward_relation(self, item_embs, entity_embs, relations, adj):
        Wh = item_embs.unsqueeze(1).expand(entity_embs.shape[0],entity_embs.shape[1], -1)
        We = entity_embs
        e_input = self.fc(torch.cat([Wh, relations, We], dim=-1)).squeeze()
        e = self.leakyrelu(e_input)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout,training=self.training)
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted
        return h_prime

    def forward(self, item_embs, entity_embs, adj):
        Wh = torch.mm(item_embs,self.W)
        We = torch.matmul(entity_embs, self.W)
        a_input = self._prepare_cat(Wh, We)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout,training=self.training)
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1),entity_embs).squeeze()
        h_prime = entity_emb_weighted + item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_cat(self, Wh, We):
        Wh = Wh.unsqueeze(1).expand(We.size())
        return torch.cat((Wh, We), dim=-1)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
