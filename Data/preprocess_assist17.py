#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预处理ASSIST2017数据集
将原始CSV文件转换为DiffuQKT模型所需的格式

输入: DiffuQKT-main/ASSIST2017/anonymized_full_release_competition_dataset.csv
输出: 
  - data/ASSIST17/ques_skill.csv      (问题-技能映射)
  - data/ASSIST17/train_question.txt  (训练集问题数据)
  - data/ASSIST17/test_question.txt   (测试集问题数据)
  - data/ASSIST17/train_skill.txt     (训练集技能数据)
  - data/ASSIST17/test_skill.txt      (测试集技能数据)
  - data/ASSIST17/problem_map.txt     (问题ID映射)
  - data/ASSIST17/skill_map.txt       (技能ID映射)
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

def load_and_process_data(csv_path):
    """
    加载CSV数据并处理
    """
    print(f"加载数据: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # ASSIST2017数据按startTime排序（时间戳）
    df = df.sort_values(['studentId', 'startTime'])
    
    return df

def create_ques_skill_mapping(df):
    """
    创建问题-技能映射关系
    """
    print("\n创建问题-技能映射...")
    
    # 获取唯一的问题和技能ID
    ques_skill_map = df[['problemId', 'skill']].drop_duplicates()
    ques_skill_map = ques_skill_map.sort_values('problemId')
    
    print(f"独特问题数: {ques_skill_map['problemId'].nunique()}")
    print(f"独特技能数: {ques_skill_map['skill'].nunique()}")
    
    return ques_skill_map

def group_by_user(df):
    """
    按用户组织数据，保持时间顺序
    """
    print("\n按用户分组数据...")
    user_interactions = defaultdict(list)
    
    for idx, row in df.iterrows():
        user_id = int(row['studentId'])
        problem_id = int(row['problemId'])
        skill = str(row['skill']).strip()
        correct = int(row['correct'])
        
        user_interactions[user_id].append({
            'problem_id': problem_id,
            'skill': skill,
            'correct': correct
        })
    
    print(f"独特用户数: {len(user_interactions)}")
    
    return user_interactions

def generate_dataset_files(output_dir, user_interactions, ques_skill_map, train_ratio=0.8):
    """
    生成训练/测试文件，按用户分割
    
    格式：
    Line 1: skip (保留)
    Line 2: problem_id序列 (逗号分割)
    Line 3: correct答案序列 (逗号分割)
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建ID映射
    unique_problems = ques_skill_map['problemId'].unique()
    problem_to_idx = {pid: idx for idx, pid in enumerate(sorted(unique_problems))}
    
    unique_skills = ques_skill_map['skill'].unique()
    skill_to_idx = {skill: idx for idx, skill in enumerate(sorted(unique_skills))}
    
    print(f"\n创建ID映射...")
    print(f"问题映射数: {len(problem_to_idx)}")
    print(f"技能映射数: {len(skill_to_idx)}")
    
    # 按用户分割
    user_ids = list(user_interactions.keys())
    train_users, test_users = train_test_split(user_ids, test_size=1-train_ratio, random_state=42)
    
    print(f"\n训练用户数: {len(train_users)}")
    print(f"测试用户数: {len(test_users)}")
    
    # 写入训练和测试文件
    with open(os.path.join(output_dir, 'train_question.txt'), 'w') as f_train_q, \
         open(os.path.join(output_dir, 'train_skill.txt'), 'w') as f_train_s, \
         open(os.path.join(output_dir, 'test_question.txt'), 'w') as f_test_q, \
         open(os.path.join(output_dir, 'test_skill.txt'), 'w') as f_test_s:
        
        for user_id in train_users:
            interactions = user_interactions[user_id]
            
            if len(interactions) >= 3:  # 至少3个交互
                problem_ids = [str(problem_to_idx[inter['problem_id']]) for inter in interactions]
                skills = [str(skill_to_idx[inter['skill']]) for inter in interactions]
                corrects = [str(inter['correct']) for inter in interactions]
                
                f_train_q.write('\n')
                f_train_q.write(','.join(problem_ids) + '\n')
                f_train_q.write(','.join(corrects) + '\n')
                
                f_train_s.write('\n')
                f_train_s.write(','.join(skills) + '\n')
                f_train_s.write(','.join(corrects) + '\n')
        
        for user_id in test_users:
            interactions = user_interactions[user_id]
            
            if len(interactions) >= 3:  # 至少3个交互
                problem_ids = [str(problem_to_idx[inter['problem_id']]) for inter in interactions]
                skills = [str(skill_to_idx[inter['skill']]) for inter in interactions]
                corrects = [str(inter['correct']) for inter in interactions]
                
                f_test_q.write('\n')
                f_test_q.write(','.join(problem_ids) + '\n')
                f_test_q.write(','.join(corrects) + '\n')
                
                f_test_s.write('\n')
                f_test_s.write(','.join(skills) + '\n')
                f_test_s.write(','.join(corrects) + '\n')
    
    # 写入问题-技能映射文件
    with open(os.path.join(output_dir, 'ques_skill.csv'), 'w') as f:
        f.write('problem_id,skill_id\n')
        for orig_problem in sorted(problem_to_idx.keys()):
            problem_idx = problem_to_idx[orig_problem]
            # 找到该问题对应的技能
            skill_mask = ques_skill_map['problemId'] == orig_problem
            skills = ques_skill_map[skill_mask]['skill'].unique()
            if len(skills) > 0:
                skill = str(skills[0])
                skill_idx = skill_to_idx[skill]
                f.write(f'{problem_idx},{skill_idx}\n')
    
    # 写入映射关系文件
    with open(os.path.join(output_dir, 'problem_map.txt'), 'w') as f:
        f.write('problem_id,original_problem_id\n')
        for orig_id, new_id in sorted(problem_to_idx.items(), key=lambda x: x[1]):
            f.write(f'{new_id},{orig_id}\n')
    
    with open(os.path.join(output_dir, 'skill_map.txt'), 'w') as f:
        f.write('skill_id,original_skill\n')
        for orig_skill, new_id in sorted(skill_to_idx.items(), key=lambda x: x[1]):
            f.write(f'{new_id},{orig_skill}\n')
    
    print(f"\n生成文件完成:")
    print(f"  - train_question.txt ({len(train_users)} 用户)")
    print(f"  - train_skill.txt ({len(train_users)} 用户)")
    print(f"  - test_question.txt ({len(test_users)} 用户)")
    print(f"  - test_skill.txt ({len(test_users)} 用户)")
    print(f"  - ques_skill.csv")
    print(f"  - problem_map.txt")
    print(f"  - skill_map.txt")
    
    return len(problem_to_idx), len(skill_to_idx)

def main():
    """
    主处理流程
    """
    csv_path = './ASSIST2017/anonymized_full_release_competition_dataset.csv'
    output_dir = './data/ASSIST17'
    
    # 加载数据
    df = load_and_process_data(csv_path)
    
    # 创建问题-技能映射
    ques_skill_map = create_ques_skill_mapping(df)
    
    # 按用户分组
    user_interactions = group_by_user(df)
    
    # 生成数据文件
    pro_max, skill_max = generate_dataset_files(output_dir, user_interactions, ques_skill_map)
    
    print(f"\n========================================")
    print(f"数据预处理完成!")
    print(f"问题数量: {pro_max}")
    print(f"技能数量: {skill_max}")
    print(f"========================================")

if __name__ == '__main__':
    main()
