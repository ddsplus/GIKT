#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess Statics2011 dataset for DiffuQKT.

Input:
  - ./Statics2011/AllData_student_step_2011F.csv

Output:
  - data/STATICS2011/ques_skill.csv
  - data/STATICS2011/train_question.txt
  - data/STATICS2011/test_question.txt
  - data/STATICS2011/train_skill.txt
  - data/STATICS2011/test_skill.txt
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_skill_text(skill_text: str) -> List[str]:
    value = str(skill_text).strip()
    if not value or value == "." or value.lower() == "nan":
        return []
    return [x.strip() for x in value.split("~~") if x.strip() and x.strip() != "."]


def load_and_process_data(csv_path: str) -> pd.DataFrame:
    print(f"Loading data: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"Raw shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    required_cols = [
        "Anon Student Id",
        "Problem Name",
        "Step Name",
        "First Transaction Time",
        "First Attempt",
        "KC (F2011)",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Keep only rows with required user/question context.
    df = df.dropna(subset=["Anon Student Id", "Problem Name", "Step Name"]).copy()
    df["Anon Student Id"] = df["Anon Student Id"].astype(str).str.strip()
    df["Problem Name"] = df["Problem Name"].astype(str).str.strip()
    df["Step Name"] = df["Step Name"].astype(str).str.strip()

    # Build stable item text key and parse timestamp for sorting.
    df["item_text"] = df["Problem Name"] + "::" + df["Step Name"]
    df["ts"] = pd.to_datetime(df["First Transaction Time"], errors="coerce")

    # Derive binary correctness from First Attempt.
    fa = df["First Attempt"].astype(str).str.strip().str.lower()
    df["correct"] = np.where(fa == "correct", 1, 0)

    # Extract a single skill from KC (F2011) for DiffuQKT mapping.
    def first_skill(x: str):
        skills = split_skill_text(x)
        return skills[0] if skills else None

    df["skill_text"] = df["KC (F2011)"].apply(first_skill)

    # Remove rows without a usable skill.
    before_rows = len(df)
    df = df.dropna(subset=["skill_text"]).copy()
    print(f"Dropped rows without skill: {before_rows - len(df)}")

    # Sort by user and time to preserve sequence order.
    df = df.sort_values(["Anon Student Id", "ts"], kind="mergesort").reset_index(drop=True)
    print(f"Processed shape: {df.shape}")
    return df


def create_id_mappings(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int], pd.DataFrame, pd.DataFrame]:
    print("\nCreating question/skill mappings...")
    unique_items = sorted(df["item_text"].unique())
    unique_skills = sorted(df["skill_text"].unique())

    item_to_id = {item: idx for idx, item in enumerate(unique_items)}
    skill_to_id = {skill: idx for idx, skill in enumerate(unique_skills)}

    df = df.copy()
    df["problem_id"] = df["item_text"].map(item_to_id).astype(int)
    df["skill_id"] = df["skill_text"].map(skill_to_id).astype(int)

    # Keep one mapping per problem for ques_skill.csv.
    ques_skill_map = (
        df[["problem_id", "skill_id"]]
        .drop_duplicates()
        .sort_values(["problem_id", "skill_id"])
        .groupby("problem_id", as_index=False)
        .first()
    )

    print(f"Unique problems: {len(item_to_id)}")
    print(f"Unique skills: {len(skill_to_id)}")
    print(f"Q-S map rows: {len(ques_skill_map)}")
    return item_to_id, skill_to_id, ques_skill_map, df


def group_by_user(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    print("\nGrouping by user...")
    groups: Dict[str, pd.DataFrame] = {}
    for user_id, group in df.groupby("Anon Student Id"):
        # Stable sorting even if some timestamps are missing.
        group = group.sort_values(["ts"], kind="mergesort").reset_index(drop=True)
        groups[user_id] = group

    seq_lengths = [len(g) for g in groups.values()]
    print(f"Total users: {len(groups)}")
    if seq_lengths:
        print(f"Avg seq len: {np.mean(seq_lengths):.2f}")
        print(f"Min seq len: {min(seq_lengths)}")
        print(f"Max seq len: {max(seq_lengths)}")
    return groups


def write_split_files(
    user_groups: Dict[str, pd.DataFrame],
    user_ids: List[str],
    output_dir: str,
    split_name: str,
    min_seq: int = 3,
) -> int:
    q_path = os.path.join(output_dir, f"{split_name}_question.txt")
    s_path = os.path.join(output_dir, f"{split_name}_skill.txt")
    kept = 0

    with open(q_path, "w", encoding="utf-8") as fq, open(s_path, "w", encoding="utf-8") as fs:
        for user_id in user_ids:
            g = user_groups[user_id]
            if len(g) < min_seq:
                continue

            problem_seq = g["problem_id"].astype(int).tolist()
            skill_seq = g["skill_id"].astype(int).tolist()
            answer_seq = g["correct"].astype(int).tolist()

            fq.write(f"{user_id}\n")
            fq.write(",".join(map(str, problem_seq)) + "\n")
            fq.write(",".join(map(str, answer_seq)) + "\n")

            fs.write(f"{user_id}\n")
            fs.write(",".join(map(str, skill_seq)) + "\n")
            fs.write(",".join(map(str, answer_seq)) + "\n")

            kept += 1

    return kept


def generate_dataset_files(
    user_groups: Dict[str, pd.DataFrame],
    output_dir: str = "data/STATICS2011",
    test_size: float = 0.2,
    random_state: int = 42,
    min_seq: int = 3,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    user_ids = list(user_groups.keys())
    train_ids, test_ids = train_test_split(user_ids, test_size=test_size, random_state=random_state)

    print(f"\nTrain users: {len(train_ids)}")
    print(f"Test users: {len(test_ids)}")

    train_kept = write_split_files(user_groups, train_ids, output_dir, "train", min_seq=min_seq)
    test_kept = write_split_files(user_groups, test_ids, output_dir, "test", min_seq=min_seq)

    print(f"Kept train users (seq_len >= {min_seq}): {train_kept}")
    print(f"Kept test users (seq_len >= {min_seq}): {test_kept}")


def save_ques_skill_mapping(ques_skill_map: pd.DataFrame, output_dir: str = "data/STATICS2011") -> None:
    output_file = os.path.join(output_dir, "ques_skill.csv")
    ques_skill_map.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")


def main() -> None:
    csv_path = "./Statics2011/AllData_student_step_2011F.csv"
    output_dir = "data/STATICS2011"

    if not os.path.exists(csv_path):
        print(f"Error: input file not found: {csv_path}")
        return

    df = load_and_process_data(csv_path)
    _, _, ques_skill_map, df = create_id_mappings(df)
    user_groups = group_by_user(df)
    generate_dataset_files(user_groups, output_dir=output_dir)
    save_ques_skill_mapping(ques_skill_map, output_dir=output_dir)

    print("\nDone.")
    print(f"Output dir: {output_dir}")
    print("Files:")
    print("  - ques_skill.csv")
    print("  - train_question.txt")
    print("  - test_question.txt")
    print("  - train_skill.txt")
    print("  - test_skill.txt")


if __name__ == "__main__":
    main()
