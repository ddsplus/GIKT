import torch
import torch.nn as nn
import torch.nn.functional as F


class Aggregator(nn.Module):
    def __init__(self, dim, dropout=0.0, act=torch.relu, name=None):
        super().__init__()
        self.dim = dim
        self.dropout = float(dropout)
        self.act = act
        self.name = name or self.__class__.__name__.lower()

    def forward(self, self_vectors, neighbor_vectors, question_embeddings=None):
        return self._call(self_vectors, neighbor_vectors, question_embeddings)

    def _call(self, self_vectors, neighbor_vectors, question_embeddings=None):
        raise NotImplementedError


class SumAggregator(Aggregator):
    def __init__(self, dim, dropout=0.0, act=torch.relu, name=None):
        super().__init__(dim=dim, dropout=dropout, act=act, name=name)
        self.weights = nn.Linear(dim, dim, bias=True)

    def _call(self, self_vectors, neighbor_vectors, question_embeddings=None):
        neighbors_agg = neighbor_vectors.mean(dim=-2)
        output = self_vectors + neighbors_agg
        output = output.reshape(-1, self.dim)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.weights(output)
        output = output.reshape(*self_vectors.shape[:-1], self.dim)
        return self.act(output)


class ConcatAggregator(Aggregator):
    def __init__(self, dim, dropout=0.0, act=torch.relu, name=None):
        super().__init__(dim=dim, dropout=dropout, act=act, name=name)
        self.weights = nn.Linear(dim * 2, dim, bias=True)

    def _call(self, self_vectors, neighbor_vectors, question_embeddings=None):
        neighbors_agg = neighbor_vectors.mean(dim=-2)
        output = torch.cat([self_vectors, neighbors_agg], dim=-1)
        output = output.reshape(-1, self.dim * 2)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.weights(output)
        output = output.reshape(*self_vectors.shape[:-1], self.dim)
        return self.act(output)
