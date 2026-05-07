#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预处理ASSIST2009数据集
将原始CSV文件转换为DiffuQKT模型所需的格式

输入: DiffuQKT-main/ASSIST2009/skill_builder_data.csv
输出: 
  - data/ASSIST09/ques_skill.csv      (问题-技能映射)
  - data/ASSIST09/train_question.txt  (训练集问题数据)
  - data/ASSIST09/test_question.txt   (测试集问题数据)
  - data/ASSIST09/train_skill.txt     (训练集技能数据)
  - data/ASSIST09/test_skill.txt      (测试集技能数据)
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
    df = pd.read_csv(csv_path, index_col=0, encoding='latin1')
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    required_cols = ['user_id', 'order_id', 'problem_id', 'skill_id', 'correct']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列: {missing_cols}")

    # 统一转为数值并处理空值，避免生成小数或非法ID
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    before_rows = len(df)
    df = df.dropna(subset=required_cols).copy()
    dropped_rows = before_rows - len(df)
    if dropped_rows > 0:
        print(f"丢弃含空值的行数: {dropped_rows}")

    # 转为整数，避免输出如 317.0
    for col in required_cols:
        df[col] = df[col].astype(int)

    # 按user_id和时间顺序排序
    df = df.sort_values(['user_id', 'order_id'])
    
    return df

def create_ques_skill_mapping(df):
    """
    创建问题-技能映射关系
    """
    print("\n创建问题-技能映射...")
    
    # 获取唯一的问题和技能ID
    ques_skill_map = df[['problem_id', 'skill_id']].drop_duplicates()
    ques_skill_map = ques_skill_map.sort_values('problem_id')
    
    print(f"独特问题数: {ques_skill_map['problem_id'].nunique()}")
    print(f"独特技能数: {ques_skill_map['skill_id'].nunique()}")
    
    return ques_skill_map

def group_by_user(df):
    """
    按用户ID分组数据
    返回: 字典 {user_id: user_data}
    """
    print("\n按用户分组...")
    user_groups = {}
    
    for user_id, group in df.groupby('user_id'):
        # 按order_id排序确保时间顺序
        group = group.sort_values('order_id')
        user_groups[user_id] = group.reset_index(drop=True)
    
    print(f"总用户数: {len(user_groups)}")
    
    # 统计用户序列长度分布
    seq_lengths = [len(group) for group in user_groups.values()]
    print(f"平均序列长度: {np.mean(seq_lengths):.2f}")
    print(f"最小序列长度: {min(seq_lengths)}")
    print(f"最大序列长度: {max(seq_lengths)}")
    
    return user_groups

def generate_dataset_files(user_groups, ques_skill_map, output_dir='data/ASSIST09'):
    """
    生成训练、测试数据文件 (8:2 分割)
    
    文件格式（3行一组）：
    行1: 跳过行（任意内容）
    行2: 逗号分隔的问题ID列表
    行3: 逗号分隔的答案列表（0或1）
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    # 将问题-技能映射转换为字典以便快速查询
    ques_skill_dict = dict(zip(ques_skill_map['problem_id'], ques_skill_map['skill_id'].astype(int)))
    
    # 分割数据为train/test (8:2)
    user_ids = list(user_groups.keys())
    train_ids, test_ids = train_test_split(user_ids, test_size=0.2, random_state=42)
    
    print(f"训练集用户数: {len(train_ids)}")
    print(f"测试集用户数: {len(test_ids)}")
    
    # 生成数据文件
    splits = {
        'train': train_ids,
        'test': test_ids
    }
    
    stats = {'train': {'total': 0, 'filtered': 0},
             'test': {'total': 0, 'filtered': 0}}
    
    for split_name, user_id_list in splits.items():
        ques_file = os.path.join(output_dir, f'{split_name}_question.txt')
        skill_file = os.path.join(output_dir, f'{split_name}_skill.txt')
        
        with open(ques_file, 'w') as f_ques, open(skill_file, 'w') as f_skill:
            for user_id in user_id_list:
                user_data = user_groups[user_id]
                
                # 获取问题、技能和答案序列
                problem_seq = user_data['problem_id'].astype(int).tolist()
                skill_seq = [int(ques_skill_dict.get(p, -1)) for p in problem_seq]
                answer_seq = user_data['correct'].astype(int).tolist()
                
                # 过滤掉有无效技能ID的用户
                if -1 in skill_seq:
                    stats[split_name]['filtered'] += 1
                    continue
                
                stats[split_name]['total'] += 1
                
                # 写入3行格式
                f_ques.write(f"{user_id}\n")  # 行1: 跳过行（用户ID作为标记）
                f_ques.write(','.join(map(str, problem_seq)) + '\n')  # 行2: 问题ID
                f_ques.write(','.join(map(str, answer_seq)) + '\n')   # 行3: 答案
                
                f_skill.write(f"{user_id}\n")  # 行1: 跳过行
                f_skill.write(','.join(map(str, skill_seq)) + '\n')   # 行2: 技能ID
                f_skill.write(','.join(map(str, answer_seq)) + '\n')  # 行3: 答案
        
        print(f"{split_name}: {stats[split_name]['total']} 用户 ({stats[split_name]['filtered']} 个无效用户已过滤)")
    
    return stats

def save_ques_skill_mapping(ques_skill_map, output_dir='data/ASSIST09'):
    """
    保存问题-技能映射为CSV
    """
    output_file = os.path.join(output_dir, 'ques_skill.csv')
    ques_skill_map.to_csv(output_file, index=False)
    print(f"\n问题-技能映射保存到: {output_file}")
    print(f"映射记录数: {len(ques_skill_map)}")

def main():
    # 数据路径
    csv_path = './ASSIST2009/skill_builder_data.csv'
    output_dir = 'data/ASSIST09'
    
    # 检查输入文件
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return
    
    # 1. 加载数据
    df = load_and_process_data(csv_path)
    
    # 2. 创建问题-技能映射
    ques_skill_map = create_ques_skill_mapping(df)
    
    # 3. 按用户分组
    user_groups = group_by_user(df)
    
    # 4. 生成数据集文件
    stats = generate_dataset_files(user_groups, ques_skill_map, output_dir)
    
    # 5. 保存问题-技能映射
    save_ques_skill_mapping(ques_skill_map, output_dir)
    
    # 打印总体统计信息
    print("\n" + "="*50)
    print("处理完成!")
    print("="*50)
    total_samples = sum(stats[k]['total'] for k in stats)
    print(f"总有效用户数: {total_samples}")
    print(f"总过滤用户数: {sum(stats[k]['filtered'] for k in stats)}")
    print(f"\n输出文件位置: {output_dir}/")
    print("  - ques_skill.csv")
    print("  - train_question.txt, train_skill.txt")
    print("  - test_question.txt, test_skill.txt")

if __name__ == '__main__':
    main()
