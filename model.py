# encoding:utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from aggregators import SumAggregator, ConcatAggregator


class GIKT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_neurons = args.hidden_neurons
        self.max_step = args.max_step - 1
        self.feature_answer_size = args.feature_answer_size
        self.field_size = args.field_size
        self.embedding_size = args.embedding_size

        self.dropout_keep_probs = eval(args.dropout_keep_probs) if isinstance(args.dropout_keep_probs, str) else list(args.dropout_keep_probs)
        self.select_index = list(args.select_index) if not isinstance(args.select_index, list) else args.select_index
        self.hist_neighbor_num = args.hist_neighbor_num
        self.next_neighbor_num = args.next_neighbor_num
        self.lr = args.lr
        self.n_hop = args.n_hop
        self.question_neighbor_num = args.question_neighbor_num
        self.skill_neighbor_num = args.skill_neighbor_num

        self.hidden_size = self.hidden_neurons[-1]

        self.feature_embedding = nn.Embedding(self.feature_answer_size, self.embedding_size)
        self.feature_layer = nn.Linear(self.embedding_size, self.hidden_size)
        self.input_layer = nn.Linear(self.hidden_size + self.embedding_size, self.hidden_size)

        self.rnn_cells = nn.ModuleList()
        for idx, hidden_size in enumerate(self.hidden_neurons):
            input_size = self.hidden_size if idx == 0 else self.hidden_neurons[idx - 1]
            self.rnn_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))

        self.atn_weights_1 = nn.Linear(self.hidden_size, 1)
        self.atn_weights_2 = nn.Linear(self.hidden_size, 1)

        if args.aggregator == "sum":
            agg_cls = SumAggregator
        elif args.aggregator == "concat":
            agg_cls = ConcatAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)
        self.aggregators = nn.ModuleList(
            [agg_cls(dim=self.embedding_size, dropout=1.0 - self.dropout_keep_probs[1], act=torch.tanh) for _ in range(max(1, self.n_hop))]
        )

        self.register_buffer("question_neighbors", torch.LongTensor(args.question_neighbors))
        self.register_buffer("skill_neighbors", torch.LongTensor(args.skill_neighbors))

    def _gather_feature_embedding(self, idx):
        idx = idx.clamp(min=0, max=self.feature_answer_size - 1).long()
        return self.feature_embedding(idx)

    def get_neighbors(self, n_hop, question_index):
        seeds = [question_index]
        for i in range(n_hop):
            if i % 2 == 0:
                neighbor = self.question_neighbors[seeds[i].reshape(-1)].reshape(-1, self.max_step, self.question_neighbor_num)
            else:
                neighbor = self.skill_neighbors[seeds[i].reshape(-1)].reshape(-1, self.max_step, self.skill_neighbor_num)
            seeds.append(neighbor)
        return seeds

    def aggregate(self, input_neighbors):
        sq_neighbor_vectors = []
        for neighbors in input_neighbors:
            temp_neighbors = self._gather_feature_embedding(neighbors.reshape(-1)).reshape(
                neighbors.shape[0], neighbors.shape[1], -1, self.embedding_size
            )
            sq_neighbor_vectors.append(temp_neighbors)

        for i in range(self.n_hop):
            aggregator = self.aggregators[i]
            for hop in range(self.n_hop - i):
                if hop % 2 == 0:
                    shape = (
                        sq_neighbor_vectors[hop + 1].shape[0],
                        sq_neighbor_vectors[hop + 1].shape[1],
                        -1,
                        self.question_neighbor_num,
                        self.embedding_size,
                    )
                else:
                    shape = (
                        sq_neighbor_vectors[hop + 1].shape[0],
                        sq_neighbor_vectors[hop + 1].shape[1],
                        -1,
                        self.skill_neighbor_num,
                        self.embedding_size,
                    )
                sq_neighbor_vectors[hop] = aggregator(
                    self_vectors=sq_neighbor_vectors[hop],
                    neighbor_vectors=sq_neighbor_vectors[hop + 1].reshape(*shape),
                    question_embeddings=sq_neighbor_vectors[hop],
                )

        return sq_neighbor_vectors

    def hist_neighbor_sampler(self, input_embedding, hist_neighbor_index):
        batch_size = input_embedding.shape[0]
        zero_embeddings = torch.zeros(batch_size, 1, self.hidden_size, device=input_embedding.device, dtype=input_embedding.dtype)
        input_embedding = torch.cat([input_embedding, zero_embeddings], dim=1)
        temp_hist_index = hist_neighbor_index.reshape(batch_size, -1).long().clamp(min=0, max=self.max_step)
        gather_idx = temp_hist_index.unsqueeze(-1).expand(-1, -1, input_embedding.shape[-1])
        gathered = torch.gather(input_embedding, 1, gather_idx)
        return gathered.reshape(batch_size, self.max_step, self.hist_neighbor_num, input_embedding.shape[-1])

    def hist_neighbor_sampler1(self, input_q_emb, next_q_emb, qa_emb):
        if self.hist_neighbor_num == 0:
            return qa_emb.new_zeros(qa_emb.shape[0], self.max_step, 0, qa_emb.shape[-1])
        eps = 1e-12
        next_norm = torch.sqrt(torch.sum(next_q_emb * next_q_emb, dim=-1) + eps)
        input_norm = torch.sqrt(torch.sum(input_q_emb * input_q_emb, dim=-1) + eps)
        q_similarity = torch.sum(next_q_emb.unsqueeze(2) * input_q_emb.unsqueeze(1), dim=-1)
        molds = next_norm.unsqueeze(2) * input_norm.unsqueeze(1) + eps
        q_similarity = q_similarity / molds

        seq_mask = torch.tril(torch.ones(self.max_step, self.max_step, device=q_similarity.device))
        q_similarity = q_similarity * seq_mask.unsqueeze(0)
        q_similarity = torch.where(
            q_similarity > self.args.att_bound,
            q_similarity,
            torch.zeros_like(q_similarity),
        )

        k = min(self.hist_neighbor_num, self.max_step)
        topk_vals, topk_idx = torch.topk(q_similarity, k=k, dim=-1)
        if k < self.hist_neighbor_num:
            pad_size = self.hist_neighbor_num - k
            pad_vals = topk_vals[..., -1:].expand(*topk_vals.shape[:-1], pad_size)
            pad_idx = topk_idx[..., -1:].expand(*topk_idx.shape[:-1], pad_size)
            topk_vals = torch.cat([topk_vals, pad_vals], dim=-1)
            topk_idx = torch.cat([topk_idx, pad_idx], dim=-1)
        minus_one = -torch.ones_like(topk_idx)
        temp_hist_index = torch.where(topk_vals > 0, topk_idx, minus_one)

        batch_size = qa_emb.shape[0]
        zero_embeddings = torch.zeros(batch_size, 1, self.hidden_size, device=qa_emb.device, dtype=qa_emb.dtype)
        qa_emb = torch.cat([qa_emb, zero_embeddings], dim=1)
        temp_hist_index = temp_hist_index.clamp(min=-1, max=self.max_step)
        temp_hist_index = torch.where(
            temp_hist_index < 0,
            torch.full_like(temp_hist_index, self.max_step),
            temp_hist_index,
        )
        gather_idx = temp_hist_index.reshape(batch_size, -1).unsqueeze(-1).expand(-1, -1, qa_emb.shape[-1])
        gathered = torch.gather(qa_emb, 1, gather_idx)
        return gathered.reshape(batch_size, self.max_step, self.hist_neighbor_num, qa_emb.shape[-1])

    def next_neighbor_sampler(self, aggregate_embedding):
        temp_emb = aggregate_embedding[1].reshape(-1, self.question_neighbor_num, self.embedding_size)
        perm = torch.randperm(self.question_neighbor_num, device=temp_emb.device)
        temp_emb = temp_emb[:, perm, :]
        if self.question_neighbor_num >= self.next_neighbor_num:
            selected = temp_emb[:, :self.next_neighbor_num, :]
        else:
            repeat_n = int(math.ceil(float(self.next_neighbor_num) / float(self.question_neighbor_num)))
            selected = temp_emb.repeat(1, repeat_n, 1)[:, :self.next_neighbor_num, :]
        return selected.reshape(-1, self.max_step, self.next_neighbor_num, self.embedding_size)

    def forward(self, features_answer_index, target_answers, seq_lens, hist_neighbor_index):
        select_feature_index = features_answer_index[:, :, self.select_index]
        questions_index = select_feature_index[:, :-1, 1]
        next_questions_index = select_feature_index[:, 1:, 1]
        skill_index = select_feature_index[:, :-1, 0]
        next_skill_index = select_feature_index[:, 1:, 0]

        input_questions_embedding = self._gather_feature_embedding(questions_index)
        next_questions_embedding = self._gather_feature_embedding(next_questions_index)
        input_skills_embedding = self._gather_feature_embedding(skill_index)
        next_skills_embedding = self._gather_feature_embedding(next_skill_index)

        input_answers_embedding = self._gather_feature_embedding(select_feature_index[:, :-1, -1])

        if self.n_hop > 0:
            input_neighbors = self.get_neighbors(self.n_hop, questions_index)
            aggregate_embedding = self.aggregate(input_neighbors)

            next_input_neighbors = self.get_neighbors(self.n_hop, next_questions_index)
            next_aggregate_embedding = self.aggregate(next_input_neighbors)

            feature_trans_embedding = F.relu(self.feature_layer(aggregate_embedding[0].reshape(-1, self.embedding_size))).reshape(
                -1, self.max_step, self.hidden_size
            )
            next_trans_embedding = F.relu(self.feature_layer(next_aggregate_embedding[0].reshape(-1, self.embedding_size))).reshape(
                -1, self.max_step, self.hidden_size
            )
        else:
            feature_trans_embedding = F.relu(self.feature_layer(input_questions_embedding.reshape(-1, self.embedding_size))).reshape(
                -1, self.max_step, self.hidden_size
            )
            next_trans_embedding = F.relu(self.feature_layer(next_questions_embedding.reshape(-1, self.embedding_size))).reshape(
                -1, self.max_step, self.hidden_size
            )
            input_neighbors = self.get_neighbors(1, questions_index)
            next_input_neighbors = self.get_neighbors(1, next_questions_index)
            next_aggregate_embedding = [
                next_trans_embedding,
                self._gather_feature_embedding(next_input_neighbors[-1].reshape(-1)).reshape(
                    -1, self.max_step, self.question_neighbor_num, self.embedding_size
                ),
            ]

        input_fa_embedding = torch.cat([feature_trans_embedding, input_answers_embedding], dim=-1).reshape(
            -1, self.hidden_size + self.embedding_size
        )
        input_trans_embedding = self.input_layer(input_fa_embedding).reshape(-1, self.max_step, self.hidden_size)
        input_trans_embedding = F.dropout(input_trans_embedding, p=1.0 - self.dropout_keep_probs[0], training=self.training)

        batch_size = input_trans_embedding.shape[0]
        device = input_trans_embedding.device
        states = []
        for hidden_size in self.hidden_neurons:
            h = torch.zeros(batch_size, hidden_size, device=device, dtype=input_trans_embedding.dtype)
            c = torch.zeros(batch_size, hidden_size, device=device, dtype=input_trans_embedding.dtype)
            states.append((h, c))

        output_series = []
        for t in range(self.max_step):
            x = input_trans_embedding[:, t, :]
            new_states = []
            for layer_idx, cell in enumerate(self.rnn_cells):
                h, c = states[layer_idx]
                h, c = cell(x, (h, c))
                h = F.dropout(h, p=1.0 - self.dropout_keep_probs[0], training=self.training)
                x = h
                new_states.append((h, c))
            states = new_states
            output_series.append(x)
        output_series = torch.stack(output_series, dim=1)

        if self.args.model == "hssi":
            hist_neighbors_features = self.hist_neighbor_sampler(output_series, hist_neighbor_index)
        elif self.args.model == "hsei":
            hist_neighbors_features = self.hist_neighbor_sampler(input_trans_embedding, hist_neighbor_index)
        elif self.args.model == "ssei":
            if self.args.sim_emb == "skill_emb":
                hist_neighbors_features = self.hist_neighbor_sampler1(input_skills_embedding, next_skills_embedding, input_trans_embedding)
            elif self.args.sim_emb == "question_emb":
                hist_neighbors_features = self.hist_neighbor_sampler1(input_questions_embedding, next_questions_embedding, input_trans_embedding)
            else:
                hist_neighbors_features = self.hist_neighbor_sampler1(feature_trans_embedding, next_trans_embedding, input_trans_embedding)
        else:
            if self.args.sim_emb == "skill_emb":
                hist_neighbors_features = self.hist_neighbor_sampler1(input_skills_embedding, next_skills_embedding, output_series)
            elif self.args.sim_emb == "question_emb":
                hist_neighbors_features = self.hist_neighbor_sampler1(input_questions_embedding, next_questions_embedding, output_series)
            else:
                hist_neighbors_features = self.hist_neighbor_sampler1(feature_trans_embedding, next_trans_embedding, output_series)

        if self.next_neighbor_num != 0:
            nn_emb = self.next_neighbor_sampler(next_aggregate_embedding)
            nn_emb = torch.cat([next_trans_embedding.unsqueeze(2), nn_emb], dim=2)
            next_neighbor_num = self.next_neighbor_num + 1
        else:
            nn_emb = next_trans_embedding.unsqueeze(2)
            next_neighbor_num = 1

        if self.hist_neighbor_num != 0:
            nh_emb = torch.cat([output_series.unsqueeze(2), hist_neighbors_features], dim=2)
            logits = torch.sum(nh_emb.unsqueeze(3) * nn_emb.unsqueeze(2), dim=4)
            logits = logits.reshape(-1, self.max_step, (self.hist_neighbor_num + 1) * next_neighbor_num)
        else:
            nh_emb = output_series.unsqueeze(2)
            logits = torch.sum(nh_emb.unsqueeze(3) * nn_emb.unsqueeze(2), dim=4)
            logits = logits.reshape(-1, self.max_step, next_neighbor_num)

        f1 = self.atn_weights_1(nh_emb).reshape(-1, self.max_step, self.hist_neighbor_num + 1, 1)
        f2 = self.atn_weights_2(nn_emb).reshape(-1, self.max_step, 1, next_neighbor_num)
        coefs = F.softmax(torch.tanh((f1 + f2).reshape(-1, self.max_step, (self.hist_neighbor_num + 1) * next_neighbor_num)), dim=-1)

        weighted_logits = torch.sum(logits * coefs, dim=-1)
        pred = torch.sigmoid(weighted_logits)
        binary_pred = (pred >= 0.5).long()

        mask = (torch.arange(self.max_step, device=seq_lens.device).unsqueeze(0) < (seq_lens - 1).unsqueeze(1)).float()
        bce = F.binary_cross_entropy_with_logits(weighted_logits, target_answers.float(), reduction="none")
        loss = torch.sum(bce * mask)

        return binary_pred, pred, loss
