# GIKT / HGKT (PyTorch)

本项目支持两类知识追踪模型：

- `gikt`：原始 GIKT 模型（项目已有实现）
- `hgkt`：按 `HGKT_SIGIR.tex` 落地的 HGKT 实现（新增 `hgkt/` 目录）

## 1. 环境要求

- Python 3.9+
- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `tqdm`

安装依赖：

```bash
pip install torch numpy pandas scikit-learn tqdm
```

## 2. 目录结构

- `main.py`：训练入口（也支持 `--train false` 走旧推理流程）
- `infer.py`：新推理入口（显式输入数据集、模型类型、模型路径）
- `train.py`：训练、评估、保存 best checkpoint
- `model.py`：GIKT 模型
- `hgkt/`：HGKT 模型与 HEG 构建代码
- `data_process.py`：数据读取、切分、batch 构建
- `prepare_gikt_data.py`：原始数据预处理脚本

## 3. 数据准备

### 3.1 原始数据放置

将原始数据放在 `Data/` 目录下，参考：

- `Data/ASSIST2009/skill_builder_data.csv`
- `Data/ASSIST2017/anonymized_full_release_competition_dataset.csv`
- `Data/Statics2011/AllData_student_step_2011F.csv`
- `Data/XES3G5M/train.csv`
- `Data/XES3G5M/test.csv`

### 3.2 生成训练/测试数据

执行（示例）：

```bash
python prepare_gikt_data.py --dataset assist2009
python prepare_gikt_data.py --dataset assist2017
python prepare_gikt_data.py --dataset statics2011
python prepare_gikt_data.py --dataset xes3g5m
```

生成结果在 `data/<dataset>/`：

- `<dataset>_train.csv`
- `<dataset>_test.csv`
- `<dataset>_skill_matrix.txt`
- `ques_skill.csv`

## 4. 训练操作（详细）

### 4.1 训练 GIKT

```bash
python main.py --dataset assist2009 --model gikt --num_epochs 150 --batch_size 32 --lr 0.001
```

### 4.2 训练 HGKT

```bash
python main.py --dataset assist2009 --model hgkt --num_epochs 150 --batch_size 32 --lr 0.001
```

HGKT常用参数：

- `--seq_attn_window`：序列注意力窗口，默认 `20`
- `--hgkt_exer_layers`：exercise 图卷积层数，默认 `2`
- `--hgkt_schema_layers`：schema 图卷积层数，默认 `1`

### 4.3 训练输出说明

每个 epoch 会输出：

- `train loss / train auc / train accuracy`
- `train precision / recall / f1`
- `test auc / test accuracy / precision / recall / f1`

早停策略：

- 参数 `--patience` 控制连续多少个 epoch 无提升后停止训练（默认 `10`）

日志文件：

- `logs/train_*.csv`
- `logs/test_*.csv`
- `logs/<timestamp>_config.json`

### 4.4 best 模型保存规则（已按模型隔离）

#### 兼容旧逻辑路径

- `checkpoint/<model_dir>/GIKT.pt`

#### 新 best 路径（推荐）

- `checkpoint/<dataset>/<model>/auc_<auc>_acc_<acc>_<dataset>_<model>_<time>.pt`

例如：

- `checkpoint/assist2009/gikt/auc_0.8123_acc_0.7456_assist2009_gikt_*.pt`
- `checkpoint/assist2009/hgkt/auc_0.8268_acc_0.7580_assist2009_hgkt_*.pt`

这样 `gikt` 与 `hgkt` 不会覆盖彼此最佳模型。

## 5. 推理操作（详细）

项目有两种推理方式，推荐使用 `infer.py`。

### 5.1 推荐：使用 `infer.py`（显式传模型路径）

命令格式：

```bash
python infer.py --dataset <dataset> --model <gikt|hgkt> --model_path <checkpoint_path>
```

示例（GIKT）：

```bash
python infer.py --dataset assist2009 --model gikt --model_path checkpoint/assist2009/gikt/auc_xxx_acc_xxx_assist2009_gikt_xxx.pt
```

示例（HGKT）：

```bash
python infer.py --dataset assist2009 --model hgkt --model_path checkpoint/assist2009/hgkt/auc_xxx_acc_xxx_assist2009_hgkt_xxx.pt
```

输出指标：

- `test auc`
- `test accuracy`
- `test precision`
- `test recall`
- `test f1`

如果需要做鲁棒性测试，可以直接使用 `robust_infer.py`，它会在推理阶段按噪声强度 `0 / 0.1 / 0.2 / 0.3 / 0.4 / 0.5` 依次随机翻转输入答案特征中的一部分 `0 ↔ 1`，并只输出每个噪声强度下的 `AUC` 和 `ACC`：

```bash
python robust_infer.py --dataset assist2009 --model_path checkpoint/assist2009/gikt/auc_xxx_acc_xxx_assist2009_gikt_xxx.pt
```

说明：

- `robust_infer.py` 会优先从 checkpoint 中读取模型类型；如果 checkpoint 里没有保存模型名，也可以显式加 `--model gikt` 或 `--model hgkt`。
- 噪声只作用于推理输入里的答案特征，不修改真实标签，因此输出的是在噪声观测下的模型鲁棒性表现。

说明：

- `infer.py` 会优先复用 checkpoint 内保存的 `question_neighbors/skill_neighbors`（若存在），避免推理时重采样邻居导致结果漂移。
- 若 checkpoint 较老不含上述字段，则会按当前随机种子重建邻居图。可通过 `--seed` 固定结果复现。

### 5.2 兼容方式：`main.py --train false`

```bash
python main.py --train false --dataset assist2009 --model gikt
python main.py --train false --dataset assist2009 --model hgkt
```

该方式会按内置规则自动找 checkpoint（优先模型隔离目录）。

## 6. 关键参数说明

- `--dataset`：数据集名（如 `assist2009`）
- `--model`：`gikt` 或 `hgkt`
- `--batch_size`：batch 大小
- `--max_step`：序列最大长度
- `--lr`：学习率
- `--num_epochs`：训练轮数
- `--patience`：早停耐心值

GIKT相关：

- `--n_hop`
- `--skill_neighbor_num`
- `--question_neighbor_num`
- `--hist_neighbor_num`
- `--next_neighbor_num`
- `--sim_emb`
- `--att_bound`

HGKT相关：

- `--seq_attn_window`
- `--hgkt_exer_layers`
- `--hgkt_schema_layers`

## 7. 常见问题

### 7.1 旧的 GIKT 模型还能用吗？

能用。推理时传 `--model gikt` 并指定对应权重即可。

### 7.2 为什么会加载失败（shape mismatch）？

通常是模型类型和权重不匹配：

- `gikt` 权重必须配 `--model gikt`
- `hgkt` 权重必须配 `--model hgkt`

### 7.3 如何避免模型互相覆盖？

已按 `checkpoint/<dataset>/<model>/` 分目录保存 best 模型，不会覆盖。
