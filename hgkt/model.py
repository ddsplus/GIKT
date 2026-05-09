import torch
import torch.nn as nn
import torch.nn.functional as F

from .heg import build_assignment_from_skill_matrix, build_exercise_graph


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        out = torch.matmul(adj_norm, x)
        out = self.linear(out)
        out = F.relu(out)
        return self.dropout(out)


class HGKT(nn.Module):
    """
    Practical HGKT implementation adapted to current GIKT project interface.
    Input/Output are kept identical with GIKT.forward for train.py compatibility.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_step = args.max_step - 1
        self.hidden_size = args.hidden_neurons[-1]
        self.embedding_size = args.embedding_size
        self.feature_answer_size = args.feature_answer_size
        self.skill_num = args.skill_num
        self.question_start = args.skill_num
        self.question_num = args.question_num
        self.seq_attn_window = getattr(args, "seq_attn_window", 20)

        dropout = 1.0 - (args.dropout_keep_probs[0] if isinstance(args.dropout_keep_probs, list) else eval(args.dropout_keep_probs)[0])

        # Build HEG structures from current dataset resources.
        adj = build_exercise_graph(args.train_seqs, self.question_start, self.question_num)
        assign = build_assignment_from_skill_matrix(args.skill_matrix, self.question_start)

        adj_t = torch.tensor(adj, dtype=torch.float32)
        deg = torch.sum(adj_t, dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        d_mat = torch.diag(deg_inv_sqrt)
        adj_norm = d_mat @ adj_t @ d_mat

        s_e = torch.tensor(assign, dtype=torch.float32)  # [Q, S]
        schema_adj = s_e.t() @ adj_t @ s_e
        schema_adj = (schema_adj > 0).float()
        schema_adj.fill_diagonal_(1.0)
        sdeg = torch.sum(schema_adj, dim=1)
        sdeg_inv_sqrt = torch.pow(sdeg + 1e-12, -0.5)
        sd_mat = torch.diag(sdeg_inv_sqrt)
        schema_adj_norm = sd_mat @ schema_adj @ sd_mat

        self.register_buffer("adj_norm", adj_norm)
        self.register_buffer("assignment", s_e)
        self.register_buffer("schema_adj_norm", schema_adj_norm)

        self.question_embedding = nn.Embedding(self.question_num, self.embedding_size)
        self.answer_embedding = nn.Embedding(2, self.embedding_size)
        self.skill_embedding = nn.Embedding(self.skill_num, self.embedding_size)

        self.gnn_exer = nn.ModuleList(
            [GraphConv(self.embedding_size, self.embedding_size, dropout) for _ in range(getattr(args, "hgkt_exer_layers", 2))]
        )
        self.gnn_schema = nn.ModuleList(
            [GraphConv(self.embedding_size, self.embedding_size, dropout) for _ in range(getattr(args, "hgkt_schema_layers", 1))]
        )

        self.input_layer = nn.Linear(self.embedding_size * 3, self.hidden_size)
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
        self.hist_key = nn.Linear(self.hidden_size, self.hidden_size)
        self.query_key = nn.Linear(self.hidden_size, self.hidden_size)
        self.schema_gate = nn.Linear(self.hidden_size + self.embedding_size, self.hidden_size)
        self.predictor = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def _question_to_local(self, q_global):
        return (q_global - self.question_start).clamp(min=0, max=self.question_num - 1)

    def _build_schema_embedding(self):
        q_emb = self.question_embedding.weight  # [Q, D]
        h = q_emb
        for layer in self.gnn_exer:
            h = layer(h, self.adj_norm)
        # Aggregate question embeddings to schema level by assignment.
        schema_h = torch.matmul(self.assignment.t(), h)
        norm = torch.sum(self.assignment, dim=0, keepdim=True).t().clamp(min=1.0)
        schema_h = schema_h / norm
        for layer in self.gnn_schema:
            schema_h = layer(schema_h, self.schema_adj_norm)
        return h, schema_h

    def _sequence_attention(self, outputs):
        # Causal local-window attention on hidden states.
        bsz, t_len, h_dim = outputs.shape
        ctx = outputs.new_zeros(bsz, t_len, h_dim)
        for t in range(t_len):
            left = max(0, t - self.seq_attn_window + 1)
            hist = outputs[:, left : t + 1, :]  # [B, W, H]
            q = outputs[:, t : t + 1, :]  # [B, 1, H]
            attn = torch.matmul(self.query_key(q), self.hist_key(hist).transpose(1, 2)) / (h_dim ** 0.5)
            alpha = F.softmax(attn, dim=-1)
            ctx[:, t : t + 1, :] = torch.matmul(alpha, hist)
        return ctx

    def forward(self, features_answer_index, target_answers, seq_lens, hist_neighbor_index):
        del hist_neighbor_index  # not used in HGKT implementation

        q_global = features_answer_index[:, :-1, 1].long()
        s_idx = features_answer_index[:, :-1, 0].long().clamp(min=0, max=self.skill_num - 1)
        ans_raw = features_answer_index[:, :-1, -1].long()
        ans_idx = (ans_raw >= (self.feature_answer_size - 2)).long()
        next_q_global = features_answer_index[:, 1:, 1].long()
        next_q_local = self._question_to_local(next_q_global)

        q_local = self._question_to_local(q_global)
        q_h, schema_h = self._build_schema_embedding()

        q_emb = q_h[q_local]
        skill_emb = self.skill_embedding(s_idx)
        ans_emb = self.answer_embedding(ans_idx)

        x = torch.cat([q_emb, skill_emb, ans_emb], dim=-1)
        x = self.dropout(F.relu(self.input_layer(x)))

        outputs, _ = self.lstm(x)
        seq_ctx = self._sequence_attention(outputs)

        next_schema_prob = self.assignment[next_q_local]  # [B, T, S]
        next_schema_emb = torch.matmul(next_schema_prob, schema_h)  # [B, T, D]

        fusion = torch.cat([seq_ctx, next_schema_emb], dim=-1)
        fusion = torch.tanh(self.schema_gate(fusion))

        logits = self.predictor(fusion).squeeze(-1)
        pred = torch.sigmoid(logits)
        binary_pred = (pred >= 0.5).long()

        mask = (torch.arange(self.max_step, device=seq_lens.device).unsqueeze(0) < (seq_lens - 1).unsqueeze(1)).float()
        bce = F.binary_cross_entropy_with_logits(logits, target_answers.float(), reduction="none")
        loss = torch.sum(bce * mask)
        return binary_pred, pred, loss

