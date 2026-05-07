#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict


def _parse_int(token):
    token = token.strip()
    if not token:
        return None
    try:
        return int(float(token))
    except ValueError:
        return None


def _parse_list(raw_value):
    items = []
    for token in raw_value.split(','):
        value = _parse_int(token)
        if value is None:
            continue
        items.append(value)
    return items


def read_xes3g5m_csv(csv_path):
    users = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get('uid')
            if uid is None:
                continue

            questions = _parse_list(row.get('questions', ''))
            concepts = _parse_list(row.get('concepts', ''))
            responses = _parse_list(row.get('responses', ''))

            if not questions or not concepts or not responses:
                continue

            min_len = min(len(questions), len(concepts), len(responses))
            questions = questions[:min_len]
            concepts = concepts[:min_len]
            responses = responses[:min_len]

            filtered_q, filtered_c, filtered_r = [], [], []
            for q, c, r in zip(questions, concepts, responses):
                if q <= 0 or c <= 0 or r < 0:
                    continue
                filtered_q.append(q)
                filtered_c.append(c)
                filtered_r.append(r)

            if not filtered_q:
                continue

            users[uid] = (filtered_q, filtered_c, filtered_r)

    return users


# ✅ 只用 train 构建
def build_maps(train_users):
    all_q = set()
    all_c = set()
    q2c_counts = defaultdict(lambda: defaultdict(int))

    for questions, concepts, responses in train_users.values():
        for q, c, r in zip(questions, concepts, responses):
            all_q.add(q)
            all_c.add(c)
            q2c_counts[q][c] += 1

    qid_map = {q: i for i, q in enumerate(sorted(all_q))}
    cid_map = {c: i for i, c in enumerate(sorted(all_c))}

    # 单 skill 映射（DiffuQKT需要）
    q2c_final = {}
    for q, c_counts in q2c_counts.items():
        best_c = sorted(c_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        q2c_final[q] = best_c

    return qid_map, cid_map, q2c_final


def write_ques_skill_csv(out_path, qid_map, cid_map, q2c_final):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('problem_id,skill_id\n')
        for q in q2c_final:
            if q not in qid_map:
                continue
            c = q2c_final[q]
            if c not in cid_map:
                continue
            f.write(f"{qid_map[q]},{cid_map[c]}\n")


def write_ques_skill_files(out_ques, out_skill, users, qid_map, cid_map):
    # ✅ UNK ID
    UNK_Q = len(qid_map)
    UNK_C = len(cid_map)

    def sort_key(item):
        uid = item[0]
        return int(uid) if uid.isdigit() else uid

    with open(out_ques, 'w', encoding='utf-8') as fq, \
         open(out_skill, 'w', encoding='utf-8') as fs:

        for uid, (questions, concepts, responses) in sorted(users.items(), key=sort_key):
            mapped_q, mapped_c, mapped_r = [], [], []

            for q, c, r in zip(questions, concepts, responses):
                mq = qid_map.get(q, UNK_Q)
                mc = cid_map.get(c, UNK_C)

                mapped_q.append(str(mq))
                mapped_c.append(str(mc))
                mapped_r.append(str(r))

            if not mapped_q:
                continue

            fq.write(f"{uid}\n")
            fq.write(','.join(mapped_q) + '\n')
            fq.write(','.join(mapped_r) + '\n')

            fs.write(f"{uid}\n")
            fs.write(','.join(mapped_c) + '\n')
            fs.write(','.join(mapped_r) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', default='./XES3G5M/train.csv')
    parser.add_argument('--test-csv', default='./XES3G5M/test.csv')
    parser.add_argument('--out-dir', default='./data/XES3G5M')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading train CSV: {args.train_csv}")
    train_users = read_xes3g5m_csv(args.train_csv)

    print(f"Loading test CSV: {args.test_csv}")
    test_users = read_xes3g5m_csv(args.test_csv)

    print(f"Train users: {len(train_users)}, Test users: {len(test_users)}")

    # ✅ 只用 train 构图
    qid_map, cid_map, q2c_final = build_maps(train_users)

    print(f"Train-only mapped questions: {len(qid_map)}, concepts: {len(cid_map)}")

    write_ques_skill_files(
        os.path.join(args.out_dir, 'train_question.txt'),
        os.path.join(args.out_dir, 'train_skill.txt'),
        train_users,
        qid_map,
        cid_map,
    )

    write_ques_skill_files(
        os.path.join(args.out_dir, 'test_question.txt'),
        os.path.join(args.out_dir, 'test_skill.txt'),
        test_users,
        qid_map,
        cid_map,
    )

    write_ques_skill_csv(
        os.path.join(args.out_dir, 'ques_skill.csv'),
        qid_map,
        cid_map,
        q2c_final,
    )

    print('Done.')


if __name__ == '__main__':
    main()