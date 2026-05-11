import argparse
import ast
import os
import json
import numpy as np
import torch

from data_process import data_process, DataGenerator
from model import GIKT
from hgkt import HGKT
from train import compute_global_metrics


def build_parser():
    parser = argparse.ArgumentParser(description="Inference for GIKT/HGKT")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, choices=["gikt", "hgkt"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument(
        "--noise_levels",
        type=str,
        default="",
        help="Comma-separated noise rates. When set, evaluate each rate by flipping input answers with that probability.",
    )
    parser.add_argument(
        "--noise_seed",
        type=int,
        default=42,
        help="Base random seed used to sample noisy answer flips.",
    )

    # Keep aligned with training defaults for reproducibility.
    parser.add_argument("--hidden_neurons", type=str, default="[200,100]")
    parser.add_argument("--dropout_keep_probs", type=str, default="[0.6,0.8,1]")
    parser.add_argument("--aggregator", type=str, default="sum")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.92)
    parser.add_argument("--l2_weight", type=float, default=1e-8)
    parser.add_argument("--limit_max_len", type=int, default=200)
    parser.add_argument("--limit_min_len", type=int, default=3)
    parser.add_argument("--field_size", type=int, default=3)
    parser.add_argument("--embedding_size", type=int, default=100)
    parser.add_argument("--max_step", type=int, default=200)
    parser.add_argument("--input_trans_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--select_index", type=str, default="[0,1,2]")
    parser.add_argument("--n_hop", type=int, default=3)
    parser.add_argument("--skill_neighbor_num", type=int, default=10)
    parser.add_argument("--question_neighbor_num", type=int, default=4)
    parser.add_argument("--hist_neighbor_num", type=int, default=0)
    parser.add_argument("--next_neighbor_num", type=int, default=4)
    parser.add_argument("--att_bound", type=float, default=0.5)
    parser.add_argument("--sim_emb", type=str, default="skill_emb")
    parser.add_argument("--seed", type=int, default=42)

    # HGKT options
    parser.add_argument("--seq_attn_window", type=int, default=20)
    parser.add_argument("--hgkt_exer_layers", type=int, default=2)
    parser.add_argument("--hgkt_schema_layers", type=int, default=1)
    return parser


def parse_noise_levels(noise_levels_text):
    if noise_levels_text is None:
        return []
    if isinstance(noise_levels_text, (list, tuple)):
        return [float(level) for level in noise_levels_text]
    noise_levels_text = str(noise_levels_text).strip()
    if not noise_levels_text:
        return []
    return [float(level.strip()) for level in noise_levels_text.split(",") if level.strip()]


def infer_model_type(args, state):
    checkpoint_model = state.get("model")
    if isinstance(checkpoint_model, str) and checkpoint_model.strip():
        checkpoint_model = checkpoint_model.strip().lower()
        if args.model and args.model.lower() != checkpoint_model:
            raise ValueError(
                "Model type mismatch: checkpoint model='{}' but --model='{}'".format(
                    checkpoint_model, args.model
                )
            )
        return checkpoint_model

    if args.model:
        return args.model.lower()

    return None


def build_model(args, model_name, device):
    if model_name == "hgkt":
        return HGKT(args).to(device)
    return GIKT(args).to(device)


def load_model_from_checkpoint(args, state, device):
    if args.model:
        model_name = args.model.lower()
        model = build_model(args, model_name, device)
        model.load_state_dict(state["model_state_dict"])
        return model_name, model

    for model_name in ("gikt", "hgkt"):
        try:
            trial_model = build_model(args, model_name, device)
            trial_model.load_state_dict(state["model_state_dict"])
            return model_name, trial_model
        except Exception:
            continue

    raise ValueError(
        "Cannot infer model type from checkpoint weights. Pass --model explicitly."
    )


def inject_answer_noise(features_answer_index, seq_lens, noise_rate, rng, feature_base):
    """
    Flip answer feature ids for noise simulation.

    The stored answer ids are raw feature ids where:
      incorrect -> `feature_base`
      correct   -> `feature_base + 1`

    We must flip between these two raw ids (not do `1 - value`).
    """
    if noise_rate <= 0:
        return features_answer_index

    noisy_features = np.array(features_answer_index, copy=True)
    for seq_idx, seq_len in enumerate(seq_lens):
        valid_len = max(0, int(seq_len) - 1)
        if valid_len <= 0:
            continue

        flip_mask = rng.random(valid_len) < noise_rate
        if not np.any(flip_mask):
            continue

        answer_slice = noisy_features[seq_idx, :valid_len, -1]
        base = int(feature_base)

        # If raw-id encoding present (base / base+1), flip those.
        raw_pos = (answer_slice == base) | (answer_slice == base + 1)
        if np.any(raw_pos):
            flip_positions = flip_mask & raw_pos
            if np.any(flip_positions):
                vals = answer_slice[flip_positions].astype(int)
                flipped = (base + 1) - (vals - base)
                answer_slice[flip_positions] = flipped
                noisy_features[seq_idx, :valid_len, -1] = answer_slice
            continue

        # Otherwise, if values are binary 0/1, flip 0<->1.
        if np.all(np.isin(answer_slice, [0, 1])):
            flip_positions = flip_mask
            if np.any(flip_positions):
                vals = answer_slice[flip_positions].astype(int)
                flipped = 1 - vals
                answer_slice[flip_positions] = flipped
                noisy_features[seq_idx, :valid_len, -1] = answer_slice
            continue

        # Fallback: only flip elements that are exactly 0 or 1.
        fallback_pos = (answer_slice == 0) | (answer_slice == 1)
        flip_positions = flip_mask & fallback_pos
        if np.any(flip_positions):
            vals = answer_slice[flip_positions].astype(int)
            answer_slice[flip_positions] = 1 - vals
            noisy_features[seq_idx, :valid_len, -1] = answer_slice

    return noisy_features


def evaluate_once(args, model, device, noise_rate=0.0, noise_seed=42):
    test_generator = DataGenerator(
        args.test_seqs,
        args.max_step,
        batch_size=args.batch_size,
        feature_size=args.feature_answer_size - 2,
        hist_num=args.hist_neighbor_num,
    )
    test_generator.reset()

    preds, binary_preds, targets = [], [], []
    rng = np.random.default_rng(noise_seed)

    with torch.no_grad():
        while not test_generator.end:
            features_answer_index, target_answers, seq_lens, hist_neighbor_index = test_generator.next_batch()
            seq_lens_np = np.asarray(seq_lens)
            features_answer_index = inject_answer_noise(
                features_answer_index, seq_lens_np, noise_rate, rng, args.feature_answer_size - 2
            )

            features_answer_index = torch.LongTensor(features_answer_index).to(device)
            target_answers = torch.FloatTensor(target_answers).to(device)
            seq_lens = torch.LongTensor(seq_lens).to(device)
            hist_neighbor_index = torch.LongTensor(hist_neighbor_index).to(device)

            binary_pred, pred, _ = model(features_answer_index, target_answers, seq_lens, hist_neighbor_index)
            pred_np = pred.cpu().numpy()
            bin_np = binary_pred.cpu().numpy()
            tgt_np = target_answers.cpu().numpy()
            lens_np = seq_lens.cpu().numpy()

            for seq_idx, seq_len in enumerate(lens_np):
                valid_len = max(0, int(seq_len) - 1)
                preds.append(pred_np[seq_idx, 0:valid_len])
                binary_preds.append(bin_np[seq_idx, 0:valid_len])
                targets.append(tgt_np[seq_idx, 0:valid_len])

    return compute_global_metrics(preds, binary_preds, targets)


def evaluate(args):
    if not os.path.exists(args.model_path):
        raise FileNotFoundError("Model checkpoint not found: {}".format(args.model_path))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Explicitly keep current behavior and avoid FutureWarning default flip.
    try:
        state = torch.load(args.model_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(args.model_path, map_location="cpu")

    # Optional: load the exact training config for full reproducibility.
    if args.config_path:
        if not os.path.exists(args.config_path):
            raise FileNotFoundError("Config file not found: {}".format(args.config_path))
        with open(args.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key, val in cfg.items():
            if hasattr(args, key):
                setattr(args, key, val)

    args.hidden_neurons = ast.literal_eval(args.hidden_neurons) if isinstance(args.hidden_neurons, str) else args.hidden_neurons
    args.dropout_keep_probs = ast.literal_eval(args.dropout_keep_probs) if isinstance(args.dropout_keep_probs, str) else args.dropout_keep_probs
    args.select_index = ast.literal_eval(args.select_index) if isinstance(args.select_index, str) else args.select_index

    args = data_process(args)

    # Reuse training-time neighbors if checkpoint provides them.
    has_saved_neighbors = ("question_neighbors" in state and "skill_neighbors" in state)
    if has_saved_neighbors:
        args.question_neighbors = np.asarray(state["question_neighbors"])
        args.skill_neighbors = np.asarray(state["skill_neighbors"])
    else:
        print(
            "warning: checkpoint has no saved neighbors. "
            "Inference will rebuild random neighbors from current data/args; "
            "metrics may drift from training-time best score."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inferred_model = infer_model_type(args, state)
    if inferred_model is not None:
        args.model = inferred_model

    args.model, model = load_model_from_checkpoint(args, state, device)
    model.eval()
    print("dataset={}".format(args.dataset))
    print("model={}".format(args.model))
    print("checkpoint={}".format(args.model_path))
    if args.config_path:
        print("config={}".format(args.config_path))

    noise_levels = parse_noise_levels(getattr(args, "noise_levels", ""))
    if noise_levels:
        print("noise_level\ttest_auc\ttest_accuracy")
        for idx, noise_rate in enumerate(noise_levels):
            auc_value, accuracy, _, _, _ = evaluate_once(
                args,
                model,
                device,
                noise_rate=noise_rate,
                noise_seed=args.noise_seed + idx,
            )
            print("{:.1f}\t{:.6f}\t{:.6f}".format(noise_rate, auc_value, accuracy))
        return

    auc_value, accuracy, precision, recall, f_score = evaluate_once(args, model, device)
    print("test auc={:.6f}".format(auc_value))
    print("test accuracy={:.6f}".format(accuracy))
    print("test precision={:.6f}".format(precision))
    print("test recall={:.6f}".format(recall))
    print("test f1={:.6f}".format(f_score))


if __name__ == "__main__":
    parser = build_parser()
    evaluate(parser.parse_args())
