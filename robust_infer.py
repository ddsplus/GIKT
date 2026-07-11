#!/usr/bin/env python3
"""
Noise inference for GIKT/HGKT.

This script evaluates robustness under grouped perturbation settings:
1) clean baseline
2) drop_0.1 ... drop_0.5
3) c2w_0.1 ... c2w_0.5   (correct -> wrong)
4) w2c_0.1 ... w2c_0.5   (wrong -> correct)

It can also export the summary to TXT and/or CSV.
"""
import argparse
import ast
import csv
import json
import math
import os
import random

import numpy as np
import torch
from sklearn import metrics

from data_process import data_process, format_data
from infer import build_model, infer_model_type, load_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Noise Inference for GIKT/HGKT")
    parser.add_argument("--data_dir", type=str, default="Data")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, choices=["gikt", "hgkt"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])

    parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--noise_seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--output_txt", type=str, default="")

    # Keep aligned with training/inference defaults for reproducibility.
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
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_runtime_args(args):
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
    return args


def normalize_device(device_text):
    if device_text:
        return torch.device(device_text)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_correct_answer(answer_value, feature_base):
    if int(answer_value) in (0, 1):
        return int(answer_value) == 1
    return int(answer_value) == int(feature_base) + 1


def flip_answer_value(answer_value, feature_base):
    answer_value = int(answer_value)
    feature_base = int(feature_base)

    if answer_value in (feature_base, feature_base + 1):
        return feature_base + 1 - (answer_value - feature_base)
    if answer_value in (0, 1):
        return 1 - answer_value
    return answer_value


def perturb_sequence(seq, noise_level, noise_mode, rng, feature_base):
    noise_mode = str(noise_mode).lower().strip()
    if noise_mode == "clean" or float(noise_level) <= 0.0:
        return [list(step) for step in seq]

    seq_copy = [list(step) for step in seq]
    seq_len = len(seq_copy)
    if seq_len == 0:
        return seq_copy

    if noise_mode == "drop":
        if seq_len <= 2:
            return seq_copy

        candidate = np.arange(seq_len)
        num_to_drop = max(1, int(seq_len * float(noise_level)))
        num_to_drop = min(num_to_drop, seq_len - 2)
        if num_to_drop <= 0:
            return seq_copy

        drop_idx = set(rng.choice(candidate, size=num_to_drop, replace=False).tolist())
        kept = [step for idx, step in enumerate(seq_copy) if idx not in drop_idx]
        return kept if len(kept) >= 2 else seq_copy[:2]

    if noise_mode not in {"flip", "c2w", "w2c"}:
        raise ValueError("Unsupported noise_mode: {}".format(noise_mode))

    candidate = []
    for idx, step in enumerate(seq_copy):
        answer_value = step[-1]
        correct = is_correct_answer(answer_value, feature_base)
        if noise_mode == "flip":
            candidate.append(idx)
        elif noise_mode == "c2w" and correct:
            candidate.append(idx)
        elif noise_mode == "w2c" and not correct:
            candidate.append(idx)

    if not candidate:
        return seq_copy

    num_to_flip = max(1, int(len(candidate) * float(noise_level)))
    num_to_flip = min(num_to_flip, len(candidate))
    flip_indices = rng.choice(np.asarray(candidate), size=num_to_flip, replace=False)
    for idx in flip_indices.tolist():
        seq_copy[idx][-1] = flip_answer_value(seq_copy[idx][-1], feature_base)
    return seq_copy


def batch_iter(seqs, batch_size):
    for start in range(0, len(seqs), batch_size):
        yield seqs[start:start + batch_size]


def run_inference_with_noise(model, seqs, args, device, noise_level, seed, noise_mode="clean"):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model.eval()

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch_seqs in batch_iter(seqs, args.batch_size):
            noisy_batch = [
                perturb_sequence(seq, noise_level, noise_mode, rng, args.feature_answer_size - 2)
                for seq in batch_seqs
            ]

            features_answer_index, target_answers, seq_lens, hist_neighbor_index = format_data(
                noisy_batch,
                args.max_step,
                args.feature_answer_size - 2,
                args.hist_neighbor_num,
            )

            features_answer_index = torch.LongTensor(features_answer_index).to(device)
            target_answers = torch.FloatTensor(target_answers).to(device)
            seq_lens = torch.LongTensor(seq_lens).to(device)
            hist_neighbor_index = torch.LongTensor(hist_neighbor_index).to(device)

            binary_pred, pred, _ = model(features_answer_index, target_answers, seq_lens, hist_neighbor_index)

            pred_np = pred.detach().cpu().numpy()
            tgt_np = target_answers.detach().cpu().numpy()
            lens_np = seq_lens.detach().cpu().numpy()

            for seq_idx, seq_len in enumerate(lens_np):
                valid_len = max(0, int(seq_len) - 1)
                if valid_len <= 0:
                    continue
                y_pred_all.append(pred_np[seq_idx, 0:valid_len].reshape(-1))
                y_true_all.append(tgt_np[seq_idx, 0:valid_len].reshape(-1))

    y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.array([])
    return y_true, y_pred


def calc_metrics(y_true, y_pred):
    out = {"auc": float("nan"), "acc": float("nan"), "rmse": float("nan")}
    if y_true.size == 0:
        return out

    try:
        out["auc"] = float(metrics.roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        pass

    try:
        out["acc"] = float(((y_pred >= 0.5) == (y_true >= 0.5)).mean())
    except Exception:
        pass

    try:
        out["rmse"] = float(math.sqrt(np.mean((y_pred - y_true) ** 2)))
    except Exception:
        pass

    return out


def write_results_csv(path, results_list):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "mode", "noise_level", "auc", "acc", "rmse", "num_samples"])
        for result in results_list:
            writer.writerow([
                result["group"],
                result["mode"],
                result["noise_level"],
                "{:.6f}".format(result["metrics"]["auc"]),
                "{:.6f}".format(result["metrics"]["acc"]),
                "{:.6f}".format(result["metrics"]["rmse"]),
                result["num_samples"],
            ])


def write_results_txt(path, results_list, model_path, dataset, split):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("Model: {}\n".format(model_path))
        f.write("Dataset: {}\n".format(dataset))
        f.write("Split: {}\n".format(split))
        f.write("=" * 96 + "\n")
        f.write("{:<18} {:<10} {:<8} {:<14} {:<14} {:<14} {:<8}\n".format(
            "Group", "Mode", "Level", "AUC", "ACC", "RMSE", "Samples"
        ))
        f.write("=" * 96 + "\n")
        for result in results_list:
            m = result["metrics"]
            f.write(
                "{:<18} {:<10} {:<8.1f} {:<14.6f} {:<14.6f} {:<14.6f} {:<8}\n".format(
                    result["group"],
                    result["mode"],
                    result["noise_level"],
                    m["auc"],
                    m["acc"],
                    m["rmse"],
                    result["num_samples"],
                )
            )


def main():
    args = parse_args()
    set_seed(int(args.seed))
    args = load_runtime_args(args)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError("Model checkpoint not found: {}".format(args.model_path))

    try:
        state = torch.load(args.model_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(args.model_path, map_location="cpu")

    args = data_process(args)

    if "question_neighbors" in state and "skill_neighbors" in state:
        args.question_neighbors = np.asarray(state["question_neighbors"])
        args.skill_neighbors = np.asarray(state["skill_neighbors"])
    else:
        print(
            "warning: checkpoint has no saved neighbors. "
            "Inference will rebuild random neighbors from current data/args; "
            "metrics may drift from training-time best score."
        )

    inferred_model = infer_model_type(args, state)
    if inferred_model is not None:
        args.model = inferred_model

    device = normalize_device(args.device)
    _, model = load_model_from_checkpoint(args, state, device)
    model.eval()

    split_seqs = args.test_seqs if args.split == "test" else args.train_seqs

    print("dataset={}".format(args.dataset))
    print("model={}".format(args.model))
    print("checkpoint={}".format(args.model_path))
    print("split={}".format(args.split))
    if args.config_path:
        print("config={}".format(args.config_path))
    print("=" * 88)
    print("{:<18} {:<8} {:<14} {:<14} {:<14} {:<8}".format("Group", "Level", "AUC", "ACC", "RMSE", "Samples"))
    print("=" * 88)

    positive_levels = [float(level) for level in sorted(args.noise_levels) if float(level) > 0.0]
    eval_plan = [("clean", "clean", 0.0)]
    eval_plan.extend(("drop_{:.1f}".format(level), "drop", level) for level in positive_levels)
    eval_plan.extend(("c2w_{:.1f}".format(level), "c2w", level) for level in positive_levels)
    eval_plan.extend(("w2c_{:.1f}".format(level), "w2c", level) for level in positive_levels)

    results_list = []
    for idx, (group_name, noise_mode, noise_level) in enumerate(eval_plan):
        y_true, y_pred = run_inference_with_noise(
            model,
            split_seqs,
            args,
            device,
            noise_level,
            args.noise_seed + idx,
            noise_mode=noise_mode,
        )
        metric_values = calc_metrics(y_true, y_pred)
        result = {
            "group": group_name,
            "mode": noise_mode,
            "noise_level": noise_level,
            "metrics": metric_values,
            "num_samples": int(y_true.size),
        }
        results_list.append(result)
        print(
            "{:<18} {:<8.1f} {:<14.6f} {:<14.6f} {:<14.6f} {:<8}".format(
                group_name,
                noise_level,
                metric_values["auc"],
                metric_values["acc"],
                metric_values["rmse"],
                y_true.size,
            )
        )

    print("=" * 88)

    if args.output_csv:
        write_results_csv(args.output_csv, results_list)
        print("Results saved to: {}".format(args.output_csv))

    if args.output_txt:
        write_results_txt(args.output_txt, results_list, args.model_path, args.dataset, args.split)
        print("TXT report saved to: {}".format(args.output_txt))


if __name__ == "__main__":
    main()
