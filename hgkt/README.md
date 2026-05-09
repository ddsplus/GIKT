# HGKT Module

This folder implements a practical HGKT variant based on `HGKT_SIGIR.tex` and is integrated into the existing training pipeline.

## Mapping to Paper Components

- `heg.py`
  - Direct support graph (exercise-level): built from question transition co-occurrence.
  - Indirect support graph (schema-level): built by exercise-to-schema assignment.
  - Assignment matrix `S_e`: derived from `skill_matrix` as a schema proxy.
- `model.py`
  - `GNN_exer`: exercise-level graph convolution.
  - Pooling/assignment: aggregate exercise embeddings into schema embeddings through `S_e`.
  - `GNN_schema`: schema-level graph convolution.
  - Sequence module: LSTM over interaction sequence.
  - Dual attention:
    - Sequence attention: local causal attention over history (`seq_attn_window`).
    - Schema conditioning: fuse next-question schema embedding with sequence context.

## Usage

Run with:

```bash
python main.py --dataset assist2009 --model hgkt
```

Optional HGKT parameters:

- `--seq_attn_window` (default `20`)
- `--hgkt_exer_layers` (default `2`)
- `--hgkt_schema_layers` (default `1`)

