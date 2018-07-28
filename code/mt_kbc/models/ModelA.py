from collections import defaultdict
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

torch_t = torch.cuda


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # device setting
        self.use_gpu = args.use_gpu
        self.device = args.device

        # loss setting
        self.threshold = args.threshold

        # embbeding setting
        self.ent_dim = args.ent_dim
        self.rel_dim = args.rel_dim
        self.ent_size = args.ent_size
        self.rel_size = args.rel_size

        # embbding layer
        self.ent_embed = nn.Embedding(self.ent_size + 1, self.ent_dim)
        self.rel_embed = nn.Embedding(self.rel_size, self.rel_dim)

        self.ent_embed.weight.requires_grad = True
        self.rel_embed.weight.requires_grad = True

        # matrix for socoring. score function is the norm of ( W_r1 * V_h + V_r - W_r2 * V_t )
        self.trans_params = nn.ParameterList([
            nn.Parameter(torch.randn(self.rel_dim, self.ent_dim), requires_grad=True)
            for _ in range(self.rel_size * 2)
        ])

        # lstm setting, usually using one layer
        self.lstm_dim = args.lstm_dim
        self.lstm_layers = args.lstm_layers
        self.lstm_dropout = args.lstm_dropout
        # sample_size also control the max len of lstm
        self.sample_size = args.sample_size
        self.lstm_activate = args.lstm_activate
        self.leaky_relu_negative_slope = args.leaky_relu_negative_slope
        self.is_batch_norm = args.is_batch_norm

        # lstm layer
        self.lstm = nn.LSTM(self.ent_dim, self.lstm_dim, self.lstm_layers,
                            batch_first=True, dropout=self.lstm_dropout, bidirectional=True)
        self.lstm_output_linear = nn.Linear(self.ent_dim * 2, self.ent_dim, bias=True)
        self.batch_norm = nn.BatchNorm1d(self.ent_dim)

        self.init_weight()

    def init_weight(self):
        torch_t.manual_seed_all(1)

        # embedding init
        init.xavier_normal_(self.ent_embed.weight)
        init.xavier_normal_(self.rel_embed.weight)

        # matrix init
        for p in self.trans_params:
            init.xavier_normal_(p)

        if self.use_gpu:
            self.cuda(self.device)

    def forward(self, positive, negative, links, edges):
        entities = set()
        for h, r, t in positive:
            entities.add(h)
            entities.add(t)
        for h, r, t in negative:
            entities.add(h)
            entities.add(t)

        entities = list(entities)

        x = self.get_context(entities, links, edges)
        # x is a list of entity embedding, which length is len(entities)
        edict = dict()
        for e, x in zip(entities, x):
            edict[e] = x

        pos, rels = [], []
        for h, r, t in positive:
            rels.append(r)
            pos.append(torch.mv(self.trans_params[r], edict[h])
                       - torch.mv(self.trans_params[r + self.rel_size], edict[t]))
        pos = torch.cat(pos, dim=0).view(-1, self.rel_dim)
        xr = self.rel_embed(torch_t.LongTensor(rels, device=self.device))
        pos_norm = torch.norm(pos + xr, p=2, dim=1, keepdim=False)

        neg, rels = [], []
        for h, r, t in negative:
            rels.append(r)
            neg.append(torch.mv(self.trans_params[r], edict[h])
                       - torch.mv(self.trans_params[r + self.rel_size], edict[t]))
        neg = torch.cat(neg, dim=0).view(-1, self.rel_dim)
        xr = self.rel_embed(torch_t.LongTensor(rels, device=self.device))
        neg_norm = torch.norm(neg + xr, p=2, dim=1, keepdim=False)

        return sum(pos_norm + F.relu(self.threshold - neg_norm))

    def get_context(self, entities, links, edges):

        all_Ent_neighbors_idx, neighbors_size = [], []

        for i, e in enumerate(entities):
            if e in links:
                if len(links[e]) <= self.sample_size:
                    n_idx = links[e]
                else:
                    n_idx = random.sample(links[e], self.sample_size)
                if len(n_idx) == 0:
                    print('something wrong @ modelS')
                    sys.exit(1)
            else:
                if len(edges[e]) <= self.sample_size:
                    n_idx = edges[e]
                else:
                    n_idx = random.sample(edges[e], self.sample_size)
                if len(n_idx) == 0:
                    print('something wrong @ modelS')
                    sys.exit(1)

            all_Ent_neighbors_idx.append(n_idx)
            neighbors_size.append(len(n_idx))

        # pad idx for lstm
        decreasing_idx_np = np.argsort(np.array(neighbors_size))[::-1].copy()
        decreasing_idx_torch = torch_t.LongTensor(decreasing_idx_np, device=self.device)

        for i in range(len(neighbors_size)):
            all_Ent_neighbors_idx[i].extend([self.ent_size] * (max(neighbors_size) - neighbors_size[i]))

        neighbors_idx, n_size = [], []
        for i in decreasing_idx_np:
            neighbors_idx.append(all_Ent_neighbors_idx[i])
            n_size.append(neighbors_size[i])

        # lstm
        all_Ent_neighbors_idx_torch = torch_t.LongTensor(neighbors_idx, device=self.device)
        neighbors_size_torch = torch_t.LongTensor(n_size, device=self.device)

        inp_sorted = nn.utils.rnn.pack_padded_sequence(all_Ent_neighbors_idx_torch,
                                                       neighbors_size_torch, batch_first=True)
        inp_sorted_emb = nn.utils.rnn.PackedSequence(self.ent_embed(inp_sorted.data), inp_sorted.batch_sizes)
        _, (lstm_out, _) = self.lstm(inp_sorted_emb)

        lstm_out = torch.cat([lstm_out[0], lstm_out[1]], -1)

        # undo sorting
        resul = torch.zeros_like(lstm_out)
        resul.index_copy_(0, decreasing_idx_torch, lstm_out)

        # linear layer and batch_norm
        if self.is_batch_norm:
            resul = torch.split(self.batch_norm(self.lstm_output_linear(resul)), 1, dim=0)
        else:
            resul = torch.split(self.lstm_output_linear(resul), 1, dim=0)

        entities_embed = []

        for t in resul:
            if self.lstm_activate == 'None':
                entities_embed.append(t.squeeze_(dim=0))
            if self.lstm_activate == 'relu':
                entities_embed.append(F.relu_(t.squeeze_(dim=0)))
            if self.lstm_activate == 'tanh':
                entities_embed.append(F.tanh(t.squeeze_(dim=0)))
            if self.lstm_activate == 'sigmoid':
                entities_embed.append(F.sigmoid(t.squeeze_(dim=0)))
            if self.lstm_activate == 'hardtanh_':
                entities_embed.append(F.hardtanh_(t.squeeze_(dim=0)))
            if self.lstm_activate == 'leaky_relu':
                entities_embed.append(F.leaky_relu_(t.squeeze_(dim=0), negative_slope=self.leaky_relu_negative_slope))

        return entities_embed

    def get_scores(self, candidates, links, edges):
        """ used in dev and test."""
        entities = set()
        for h, r, t, l in candidates:
            entities.add(h)
            entities.add(t)
        entities = list(entities)

        xe = self.get_context(entities, links, edges)

        edict = dict()
        for e, x in zip(entities, xe):
            edict[e] = x

        diffs, rels = [], []
        for h, r, t, l in candidates:
            rels.append(r)
            diffs.append(torch.mv(self.trans_params[r], edict[h])
                         - torch.mv(self.trans_params[r + self.rel_size], edict[t]))
        diffs = torch.cat(diffs, dim=0).view(-1, self.rel_dim)
        xr = self.rel_embed(torch_t.LongTensor(rels, device=self.device))
        score_norm = torch.norm(diffs + xr, p=2, dim=1, keepdim=False)
        return score_norm



