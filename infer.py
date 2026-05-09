import argparse
import ast
import os
import numpy as np
import torch

from data_process import data_process, DataGenerator
from model import GIKT
from hgkt import HGKT
from train import compute_sequence_macro_metrics


def build_parser():
    parser = argparse.ArgumentParser(description="Inference for GIKT/HGKT")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["gikt", "hgkt"])
    parser.add_argument("--model_path", type=str, required=True)

    # Keep aligned with training defaults for reproducibility.
    parser.add_argument("--hidden_neurons", type=str, default="[200,100]")
    parser.add_argument("--dropout_keep_probs", type=str, default="[0.6,0.8,1]")
    parser.add_argument("--aggregator", type=str, default="sum")
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

    # HGKT options
    parser.add_argument("--seq_attn_window", type=int, default=20)
    parser.add_argument("--hgkt_exer_layers", type=int, default=2)
    parser.add_argument("--hgkt_schema_layers", type=int, default=1)
    return parser


def evaluate(args):
    if not os.path.exists(args.model_path):
        raise FileNotFoundError("Model checkpoint not found: {}".format(args.model_path))

    args.hidden_neurons = ast.literal_eval(args.hidden_neurons)
    args.dropout_keep_probs = ast.literal_eval(args.dropout_keep_probs)
    args.select_index = ast.literal_eval(args.select_index)

    args = data_process(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "hgkt":
        model = HGKT(args).to(device)
    else:
        model = GIKT(args).to(device)

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    test_generator = DataGenerator(
        args.test_seqs,
        args.max_step,
        batch_size=args.batch_size,
        feature_size=args.feature_answer_size - 2,
        hist_num=args.hist_neighbor_num,
    )
    test_generator.reset()

    preds, binary_preds, targets = [], [], []
    with torch.no_grad():
        while not test_generator.end:
            features_answer_index, target_answers, seq_lens, hist_neighbor_index = test_generator.next_batch()
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

    auc_value, accuracy, precision, recall, f_score = compute_sequence_macro_metrics(
        preds, binary_preds, targets
    )
    print("dataset={}".format(args.dataset))
    print("model={}".format(args.model))
    print("checkpoint={}".format(args.model_path))
    print("test auc={:.6f}".format(auc_value))
    print("test accuracy={:.6f}".format(accuracy))
    print("test precision={:.6f}".format(precision))
    print("test recall={:.6f}".format(recall))
    print("test f1={:.6f}".format(f_score))


if __name__ == "__main__":
    parser = build_parser()
    evaluate(parser.parse_args())

