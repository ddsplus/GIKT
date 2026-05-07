import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def _safe_int(v):
    try:
        return int(float(v))
    except Exception:
        return None


def _write_gikt_split(path, sequences):
    with open(path, "w", encoding="utf-8") as f:
        for seq in sequences:
            skills = [str(x[0]) for x in seq]
            questions = [str(x[1]) for x in seq]
            answers = [str(x[2]) for x in seq]
            f.write(str(len(seq)) + "\n")
            f.write(",".join(skills) + "\n")
            f.write(",".join(questions) + "\n")
            f.write(",".join(answers) + "\n")


def _to_gikt_sequences(user_steps):
    out = []
    for _, steps in user_steps.items():
        seq = []
        for skill_id, question_id, correct in steps:
            seq.append([int(skill_id), int(question_id), int(correct)])
        if len(seq) >= 3:
            out.append(seq)
    return out


def _finalize_and_save(dataset_name, train_user_steps, test_user_steps, out_root):
    all_train_skills = set()
    all_train_questions = set()
    for seq in train_user_steps.values():
        for s, q, _ in seq:
            all_train_skills.add(s)
            all_train_questions.add(q)

    skill_map = {s: i for i, s in enumerate(sorted(all_train_skills))}
    question_map = {q: i for i, q in enumerate(sorted(all_train_questions))}
    num_skills = len(skill_map)

    def remap_split(split_steps):
        mapped = {}
        for uid, seq in split_steps.items():
            new_seq = []
            for s, q, c in seq:
                if s not in skill_map or q not in question_map:
                    continue
                sid = skill_map[s]
                qid = num_skills + question_map[q]
                ans = num_skills + len(question_map) + int(c)
                new_seq.append((sid, qid, ans))
            if len(new_seq) >= 3:
                mapped[uid] = new_seq
        return mapped

    train_mapped = remap_split(train_user_steps)
    test_mapped = remap_split(test_user_steps)

    out_dir = os.path.join(out_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    train_sequences = _to_gikt_sequences(train_mapped)
    test_sequences = _to_gikt_sequences(test_mapped)
    _write_gikt_split(os.path.join(out_dir, f"{dataset_name}_train.csv"), train_sequences)
    _write_gikt_split(os.path.join(out_dir, f"{dataset_name}_test.csv"), test_sequences)

    question_count = len(question_map)
    skill_matrix = np.zeros((num_skills, num_skills + question_count), dtype=np.int32)
    # Build graph priors strictly from training split to avoid test leakage.
    for seq in train_mapped.values():
        for sid, qid, _ in seq:
            skill_matrix[sid, qid] = 1
    np.savetxt(os.path.join(out_dir, f"{dataset_name}_skill_matrix.txt"), skill_matrix, fmt="%d")

    ques_skill_path = os.path.join(out_dir, "ques_skill.csv")
    with open(ques_skill_path, "w", encoding="utf-8") as f:
        f.write("problem_id,skill_id\n")
        seen_questions = {}
        for seq in train_mapped.values():
            for sid, qid, _ in seq:
                if qid not in seen_questions:
                    seen_questions[qid] = sid
        for qid in sorted(seen_questions.keys()):
            f.write(f"{qid},{seen_questions[qid]}\n")

    print(f"[{dataset_name}] train users: {len(train_mapped)}, test users: {len(test_mapped)}")
    print(f"[{dataset_name}] skills: {num_skills}, questions: {len(question_map)}")
    print(f"[{dataset_name}] output: {out_dir}")


def preprocess_assist2009(data_root, out_root, dataset_name):
    path = os.path.join(data_root, "ASSIST2009", "skill_builder_data.csv")
    df = pd.read_csv(path, index_col=0, encoding="latin1", low_memory=False)
    req = ["user_id", "order_id", "problem_id", "skill_id", "correct"]
    df = df.dropna(subset=req).copy()
    for c in req:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=req).copy()
    for c in req:
        df[c] = df[c].astype(int)
    df = df.sort_values(["user_id", "order_id"])

    users = sorted(df["user_id"].unique().tolist())
    rng = np.random.RandomState(42)
    rng.shuffle(users)
    cut = int(len(users) * 0.8)
    train_u = set(users[:cut])

    train_steps = defaultdict(list)
    test_steps = defaultdict(list)
    for _, r in df.iterrows():
        tup = (int(r["skill_id"]), int(r["problem_id"]), int(r["correct"]))
        uid = int(r["user_id"])
        if uid in train_u:
            train_steps[uid].append(tup)
        else:
            test_steps[uid].append(tup)
    _finalize_and_save(dataset_name, train_steps, test_steps, out_root)


def preprocess_assist2017(data_root, out_root, dataset_name):
    path = os.path.join(data_root, "ASSIST2017", "anonymized_full_release_competition_dataset.csv")
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    req = ["studentId", "startTime", "problemId", "skill", "correct"]
    df = df.dropna(subset=req).copy()
    df["studentId"] = df["studentId"].astype(int)
    df["problemId"] = df["problemId"].astype(int)
    df["skill"] = df["skill"].astype(str).str.strip()
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["studentId", "startTime"])

    skill_vals = sorted(df["skill"].unique().tolist())
    skill_map = {s: i for i, s in enumerate(skill_vals)}

    users = sorted(df["studentId"].unique().tolist())
    rng = np.random.RandomState(42)
    rng.shuffle(users)
    cut = int(len(users) * 0.8)
    train_u = set(users[:cut])

    train_steps = defaultdict(list)
    test_steps = defaultdict(list)
    for _, r in df.iterrows():
        tup = (skill_map[r["skill"]], int(r["problemId"]), int(r["correct"]))
        uid = int(r["studentId"])
        if uid in train_u:
            train_steps[uid].append(tup)
        else:
            test_steps[uid].append(tup)
    _finalize_and_save(dataset_name, train_steps, test_steps, out_root)


def preprocess_statics2011(data_root, out_root, dataset_name):
    path = os.path.join(data_root, "Statics2011", "AllData_student_step_2011F.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    req = ["Anon Student Id", "Problem Name", "Step Name", "First Transaction Time", "First Attempt", "KC (F2011)"]
    df = df.dropna(subset=req).copy()
    df["uid"] = df["Anon Student Id"].astype(str).str.strip()
    df["qid_text"] = (df["Problem Name"].astype(str).str.strip() + "::" + df["Step Name"].astype(str).str.strip())
    df["correct"] = (df["First Attempt"].astype(str).str.lower().str.strip() == "correct").astype(int)
    df["ts"] = pd.to_datetime(df["First Transaction Time"], errors="coerce")
    df = df.sort_values(["uid", "ts"], kind="mergesort")

    def first_skill(v):
        text = str(v).strip()
        if not text or text == "." or text.lower() == "nan":
            return None
        skills = [x.strip() for x in text.split("~~") if x.strip() and x.strip() != "."]
        return skills[0] if skills else None

    df["skill_text"] = df["KC (F2011)"].apply(first_skill)
    df = df.dropna(subset=["skill_text"]).copy()

    qvals = sorted(df["qid_text"].unique().tolist())
    svals = sorted(df["skill_text"].unique().tolist())
    qmap = {q: i for i, q in enumerate(qvals)}
    smap = {s: i for i, s in enumerate(svals)}

    users = sorted(df["uid"].unique().tolist())
    rng = np.random.RandomState(42)
    rng.shuffle(users)
    cut = int(len(users) * 0.8)
    train_u = set(users[:cut])

    train_steps = defaultdict(list)
    test_steps = defaultdict(list)
    for _, r in df.iterrows():
        tup = (smap[r["skill_text"]], qmap[r["qid_text"]], int(r["correct"]))
        uid = r["uid"]
        if uid in train_u:
            train_steps[uid].append(tup)
        else:
            test_steps[uid].append(tup)
    _finalize_and_save(dataset_name, train_steps, test_steps, out_root)


def preprocess_xes3g5m(data_root, out_root, dataset_name):
    def parse_row(row):
        q = [_safe_int(x) for x in row.get("questions", "").split(",")]
        c = [_safe_int(x) for x in row.get("concepts", "").split(",")]
        r = [_safe_int(x) for x in row.get("responses", "").split(",")]
        q = [x for x in q if x is not None]
        c = [x for x in c if x is not None]
        r = [x for x in r if x is not None]
        m = min(len(q), len(c), len(r))
        out = []
        for i in range(m):
            if q[i] <= 0 or c[i] <= 0 or r[i] < 0:
                continue
            out.append((c[i], q[i], int(r[i])))
        return out

    train_csv = os.path.join(data_root, "XES3G5M", "train.csv")
    test_csv = os.path.join(data_root, "XES3G5M", "test.csv")

    train_steps = {}
    with open(train_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row.get("uid", "")
            seq = parse_row(row)
            if len(seq) >= 3:
                train_steps[uid] = seq

    test_steps = {}
    with open(test_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row.get("uid", "")
            seq = parse_row(row)
            if len(seq) >= 3:
                test_steps[uid] = seq

    _finalize_and_save(dataset_name, train_steps, test_steps, out_root)


def main():
    parser = argparse.ArgumentParser(description="Prepare GIKT training files from raw datasets.")
    parser.add_argument("--data_root", type=str, default="Data")
    parser.add_argument("--out_root", type=str, default="data")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["assist2009", "assist2017", "statics2011", "xes3g5m"],
    )
    parser.add_argument("--dataset_name", type=str, default=None, help="Output folder/file prefix name")
    args = parser.parse_args()

    dataset_name = args.dataset_name or args.dataset
    if args.dataset == "assist2009":
        preprocess_assist2009(args.data_root, args.out_root, dataset_name)
    elif args.dataset == "assist2017":
        preprocess_assist2017(args.data_root, args.out_root, dataset_name)
    elif args.dataset == "statics2011":
        preprocess_statics2011(args.data_root, args.out_root, dataset_name)
    elif args.dataset == "xes3g5m":
        preprocess_xes3g5m(args.data_root, args.out_root, dataset_name)


if __name__ == "__main__":
    main()
