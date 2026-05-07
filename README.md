# GIKT (PyTorch)

本仓库已改为 PyTorch 训练版本，并支持从 `Data/` 下的 4 个原始数据集一键生成本模型训练格式。

## 1. 环境要求

- Python 3.9+
- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `tqdm`

安装示例：

```bash
pip install torch numpy pandas scikit-learn tqdm
```

## 2. 数据目录约定

原始数据放在 `Data/`（已存在）：

- `Data/ASSIST2009/skill_builder_data.csv`
- `Data/ASSIST2017/anonymized_full_release_competition_dataset.csv`
- `Data/Statics2011/AllData_student_step_2011F.csv`
- `Data/XES3G5M/train.csv`
- `Data/XES3G5M/test.csv`

预处理输出到 `data/`：

- `data/<dataset>/<dataset>_train.csv`
- `data/<dataset>/<dataset>_test.csv`
- `data/<dataset>/<dataset>_skill_matrix.txt`
- `data/<dataset>/ques_skill.csv`

## 3. 预处理（本模型专用）

使用统一脚本 `prepare_gikt_data.py`：

```bash
python prepare_gikt_data.py --dataset assist2009
python prepare_gikt_data.py --dataset assist2017
python prepare_gikt_data.py --dataset statics2011
python prepare_gikt_data.py --dataset xes3g5m
```

可选参数：

- `--data_root`：原始数据目录，默认 `Data`
- `--out_root`：输出目录，默认 `data`
- `--dataset_name`：输出子目录和文件名前缀（默认和 `--dataset` 相同）

示例：

```bash
python prepare_gikt_data.py --dataset assist2009 --dataset_name assist2009
```

## 4. 训练

训练脚本为 `main.py`，当前策略：

- 无验证集
- 每个 epoch 在测试集上输出 `AUC / ACC`（同时输出 precision/recall/f1）
- 按测试集 AUC 保存最佳模型
- 早停耐心 `patience=10`

示例：

```bash
python main.py --dataset assist2009 --num_epochs 150 --patience 10 --batch_size 32 --lr 0.001
```

可用数据集名（对应你预处理时的 `--dataset_name`）：

- `assist2009`
- `assist2017`
- `statics2011`
- `xes3g5m`

## 5. 测试集监控与最佳模型保存

训练时每轮会打印：

- `train loss / train auc / train accuracy`
- `test auc / test accuracy`

最佳模型保存路径：

- `checkpoint/<model_dir>/GIKT.pt`

其中 `<model_dir>` 含数据集名、超参数和时间戳。

## 6. 日志与配置文件

日志目录：

- `logs/train_*.csv`
- `logs/test_*.csv`

启动配置（含超参数）：

- `logs/<timestamp>_config.json`

## 7. 仅测试模式

若只跑测试（加载已保存模型）：

```bash
python main.py --train false --dataset assist2009
```

要求对应的 checkpoint 已存在于：

- `checkpoint/<model_dir>/GIKT.pt`

## 8. 主要脚本说明

- `prepare_gikt_data.py`：四个原始数据集到 GIKT 训练格式转换
- `data_process.py`：读取 `data/<dataset>/` 下训练/测试文件并构图
- `model.py`：PyTorch GIKT 模型定义
- `train.py`：训练、测试监控、早停、最佳模型保存
- `main.py`：训练入口和参数管理
