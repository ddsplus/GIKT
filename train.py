import os
import glob
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

from model import GIKT
from hgkt import HGKT
from data_process import DataGenerator


def compute_sequence_macro_metrics(preds, binary_preds, targets):
    auc_list = []
    acc_list = []
    p_list = []
    r_list = []
    f1_list = []

    for p_seq, b_seq, t_seq in zip(preds, binary_preds, targets):
        if len(t_seq) == 0:
            continue
        acc_list.append(accuracy_score(t_seq, b_seq))
        precision, recall, f_score, _ = precision_recall_fscore_support(
            t_seq, b_seq, average="binary", zero_division=0
        )
        p_list.append(precision)
        r_list.append(recall)
        f1_list.append(f_score)
        if len(np.unique(t_seq)) > 1:
            auc_list.append(roc_auc_score(t_seq, p_seq))

    auc_value = float(np.mean(auc_list)) if auc_list else 0.5
    accuracy = float(np.mean(acc_list)) if acc_list else 0.0
    precision = float(np.mean(p_list)) if p_list else 0.0
    recall = float(np.mean(r_list)) if r_list else 0.0
    f_score = float(np.mean(f1_list)) if f1_list else 0.0
    return auc_value, accuracy, precision, recall, f_score


def train(args, train_dkt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.model)
    if args.model.lower() == "hgkt":
        model = HGKT(args).to(device)
    else:
        model = GIKT(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    model_dir = save_model_dir(args)
    best_test_auc = 0.0
    best_epoch = -1
    epochs_without_improve = 0

    if train_dkt:
        for epoch in tqdm(range(args.num_epochs)):
            train_generator = DataGenerator(
                args.train_seqs,
                args.max_step,
                batch_size=args.batch_size,
                feature_size=args.feature_answer_size - 2,
                hist_num=args.hist_neighbor_num,
            )
            test_generator = DataGenerator(
                args.test_seqs,
                args.max_step,
                batch_size=args.batch_size,
                feature_size=args.feature_answer_size - 2,
                hist_num=args.hist_neighbor_num,
            )
            print("epoch:", epoch)
            train_generator.shuffle()
            model.train()

            overall_loss = 0.0
            train_step = 0
            preds, binary_preds, targets = [], [], []

            while not train_generator.end:
                train_step += 1
                features_answer_index, target_answers, seq_lens, hist_neighbor_index = train_generator.next_batch()

                features_answer_index = torch.LongTensor(features_answer_index).to(device)
                target_answers = torch.FloatTensor(target_answers).to(device)
                seq_lens = torch.LongTensor(seq_lens).to(device)
                hist_neighbor_index = torch.LongTensor(hist_neighbor_index).to(device)

                optimizer.zero_grad()
                binary_pred, pred, loss = model(features_answer_index, target_answers, seq_lens, hist_neighbor_index)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
                optimizer.step()

                overall_loss += loss.item()
                pred_np = pred.detach().cpu().numpy()
                bin_np = binary_pred.detach().cpu().numpy()
                tgt_np = target_answers.detach().cpu().numpy()
                lens_np = seq_lens.detach().cpu().numpy()
                for seq_idx, seq_len in enumerate(lens_np):
                    valid_len = max(0, int(seq_len) - 1)
                    preds.append(pred_np[seq_idx, 0:valid_len])
                    binary_preds.append(bin_np[seq_idx, 0:valid_len])
                    targets.append(tgt_np[seq_idx, 0:valid_len])

            train_loss = overall_loss / max(train_step, 1)
            auc_value, accuracy, precision, recall, f_score = compute_sequence_macro_metrics(
                preds, binary_preds, targets
            )
            print("\ntrain loss = {0},auc={1}, accuracy={2}".format(train_loss, auc_value, accuracy))
            print("train precision={0}, recall={1}, f1={2}".format(precision, recall, f_score))
            write_log(args, model_dir, auc_value, accuracy, epoch, name="train_")

            test_generator.reset()
            model.eval()
            preds, binary_preds, targets = [], [], []
            test_step = 0

            with torch.no_grad():
                while not test_generator.end:
                    test_step += 1
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

            test_auc, test_acc, precision, recall, f_score = compute_sequence_macro_metrics(
                preds, binary_preds, targets
            )
            print("\ntest auc={0}, accuracy={1}, precision={2}, recall={3}, f1={4}".format(test_auc, test_acc, precision, recall, f_score))
            write_log(args, model_dir, test_auc, test_acc, epoch, name="test_")

            if test_auc > best_test_auc:
                print("%3.4f to %3.4f" % (best_test_auc, test_auc))
                best_test_auc = test_auc
                best_epoch = epoch
                save_best_checkpoint(args, model_dir, best_epoch, model, optimizer, test_auc, test_acc)
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            print(model_dir + "\tbest_test_auc=" + str(best_test_auc))
            if epochs_without_improve >= args.patience:
                print("Early stopping triggered at epoch {0}, best epoch {1}, best test auc {2}".format(epoch, best_epoch, best_test_auc))
                break
    else:
        checkpoint_path = resolve_checkpoint_path(args, model_dir)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        test_generator = DataGenerator(
            args.test_seqs,
            args.max_step,
            batch_size=args.batch_size,
            feature_size=args.feature_answer_size - 2,
            hist_num=args.hist_neighbor_num,
        )
        test_generator.reset()
        model.eval()

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
        print("\ntest auc={0}, accuracy={1}, precision={2}, recall={3}, f1={4}".format(auc_value, accuracy, precision, recall, f_score))
        write_log(args, model_dir, auc_value, accuracy, state.get("global_step", 0), name="test_")


def save(global_step, model, optimizer, checkpoint_dir, extra_payload=None):
    model_name = "GIKT.pt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra_payload:
        payload.update(extra_payload)
    torch.save(payload, os.path.join(checkpoint_dir, model_name))
    print("Save checkpoint at %d" % global_step)


def save_best_checkpoint(args, model_dir, global_step, model, optimizer, auc_value, acc_value):
    common_payload = {
        "dataset": args.dataset,
        "model": str(args.model).lower(),
        "question_neighbors": np.asarray(args.question_neighbors),
        "skill_neighbors": np.asarray(args.skill_neighbors),
        "seed": int(getattr(args, "seed", 42)),
    }
    # Keep legacy path to avoid breaking existing inference scripts.
    legacy_dir = os.path.join(args.checkpoint_dir, model_dir)
    save(global_step, model, optimizer, legacy_dir, extra_payload=common_payload)

    # New organized path: checkpoint/<dataset>/<model>/<auc_acc_dataset_model_time>.pt
    model_tag = str(args.model).lower()
    dataset_model_dir = os.path.join(args.checkpoint_dir, args.dataset, model_tag)
    os.makedirs(dataset_model_dir, exist_ok=True)
    time_tag = str(args.tag).replace(".", "_")
    model_name = "auc_{:.4f}_acc_{:.4f}_{}_{}_{}.pt".format(
        auc_value,
        acc_value,
        args.dataset,
        model_tag,
        time_tag,
    )
    payload = {
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "dataset": args.dataset,
        "model": model_tag,
        "question_neighbors": np.asarray(args.question_neighbors),
        "skill_neighbors": np.asarray(args.skill_neighbors),
        "seed": int(getattr(args, "seed", 42)),
        "auc": float(auc_value),
        "acc": float(acc_value),
        "time_tag": str(args.tag),
    }
    ckpt_path = os.path.join(dataset_model_dir, model_name)
    torch.save(payload, ckpt_path)
    print("Save best checkpoint:", ckpt_path)


def resolve_checkpoint_path(args, model_dir):
    model_tag = str(args.model).lower()

    # 0) New model-specific path (preferred): checkpoint/<dataset>/<model>/...
    dataset_model_dir = os.path.join(args.checkpoint_dir, args.dataset, model_tag)
    pattern_model = os.path.join(
        dataset_model_dir,
        "auc_*_acc_*_{}_{}_*.pt".format(args.dataset, model_tag),
    )
    model_candidates = glob.glob(pattern_model)
    if model_candidates:
        model_candidates.sort(key=os.path.getmtime, reverse=True)
        return model_candidates[0]

    # 1) Legacy path (original behavior).
    legacy_path = os.path.join(args.checkpoint_dir, model_dir, "GIKT.pt")
    if os.path.exists(legacy_path):
        return legacy_path

    # 2) Backward-compatible dataset path (before model-specific split).
    dataset_dir = os.path.join(args.checkpoint_dir, args.dataset)
    pattern = os.path.join(dataset_dir, "auc_*_acc_*_{}_*.pt".format(args.dataset))
    candidates = glob.glob(pattern)
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]

    return legacy_path


def save_model_dir(args):
    return "{}_{}_{}lr_{}hop_{}sn_{}qn_{}hn_{}nn_{}_{}bound_{}keep_{}".format(
        args.dataset,
        args.model,
        args.lr,
        args.n_hop,
        args.skill_neighbor_num,
        args.question_neighbor_num,
        args.hist_neighbor_num,
        args.next_neighbor_num,
        args.sim_emb,
        args.att_bound,
        args.dropout_keep_probs,
        args.tag,
    )


def write_log(args, model_dir, auc, accuracy, epoch, name="train_"):
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, name + model_dir + ".csv")
    if not os.path.exists(log_path):
        log_file = open(log_path, "w")
        log_file.write("Epoch\tAuc\tAccuracy\n")
    else:
        log_file = open(log_path, "a")
    log_file.write(str(epoch) + "\t" + str(auc) + "\t" + str(accuracy) + "\n")
    log_file.flush()
