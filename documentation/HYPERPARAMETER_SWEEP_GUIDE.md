# Hyperparameter Sweep Report Generator

超参数扫描训练报告生成工具，基于真实 EfficientNet 两阶段训练流程，自动执行多组超参数配置的训练并生成完整的 HTML 对比报告。

## 前置条件

- Python 3.10+
- TensorFlow 2.x（建议 GPU 环境）
- 已准备好的数据集（`train/dataset_merged/` 下包含 `Train/`、`Val/`、`Test/` 目录）
- 依赖包：`numpy`、`matplotlib`、`scikit-learn`、`pyyaml`

## 快速开始

```bash
cd src/efficientnet_lite_gpu

# 1. 运行全部 8 组实验
python -m tools.generate_training_report

# 2. 训练完成后，仅重新生成报告（不重新训练）
python -m tools.generate_training_report --report
```

## 命令参数

| 参数 | 说明 |
|------|------|
| （无参数） | 运行全部实验 + 生成报告 |
| `--report` | 跳过训练，仅从已有结果生成报告 |
| `--force` | 强制重跑已完成的实验（默认会跳过） |
| `--experiments A,B,E` | 只运行指定 ID 的实验（逗号分隔） |

## 实验配置

脚本预定义了 8 组超参数配置：

| ID | 名称 | Batch Size | S1 LR | S2 LR | S1 Epochs | S2 Epochs | Model | 说明 |
|----|------|-----------|-------|-------|-----------|-----------|-------|------|
| A | Baseline | 8 | 1e-3 | 2e-5 | 12 | 6 | B0 | 基准配置 |
| B | Large Batch | 32 | 1e-3 | 2e-5 | 12 | 6 | B0 | 大 batch size |
| C | High LR | 8 | 1e-2 | 1e-4 | 12 | 6 | B0 | 高学习率 |
| D | Low LR + Long | 8 | 5e-4 | 1e-5 | 20 | 8 | B0 | 低学习率 + 长训练 |
| E | EfficientNet-B1 | 8 | 1e-3 | 2e-5 | 12 | 6 | B1 | 更大模型 |
| F | No Fine-tune | 8 | 1e-3 | 0 | 20 | 0 | B0 | 仅 Stage 1 |
| G | Batch 16 + Mid LR | 16 | 2e-3 | 5e-5 | 15 | 6 | B0 | 中等配置 |
| H | Aggressive Fine-tune | 8 | 1e-3 | 1e-4 | 10 | 12 | B0 | 长时间微调 |

### 自定义实验

编辑 `tools/generate_training_report.py` 中的 `EXPERIMENTS` 列表，每个实验包含：

```python
{
    "id": "X",                    # 唯一标识（用于目录命名）
    "name": "X: My Experiment",   # 显示名称
    "overrides": {                # 覆盖 train_config 中的字段
        "batch_size": 8,
        "stage1_learning_rate": 1e-3,
        "stage2_learning_rate": 2e-5,
        "stage1_epochs": 12,
        "stage2_epochs": 6,
        "fine_tune": True,
    },
    "model_overrides": {          # 覆盖 model_config 中的字段
        "model_name": "efficientnet-b0",
    },
}
```

## 输出文件

训练结果保存在 `train/results_sweep/` 下：

```
train/results_sweep/
├── exp_A/                              # 每组实验独立目录
│   ├── data_exploration/
│   │   ├── class_distribution.png
│   │   ├── dataset_statistics.png
│   │   └── sample_images.png
│   ├── evaluation_results/
│   │   ├── test_metrics.json           # 测试集指标
│   │   ├── test_class_report.json      # 逐类指标
│   │   ├── test_confusion_matrix.npy   # 混淆矩阵
│   │   ├── confusion_matrix.png
│   │   └── class_performance.png
│   ├── training_logs/
│   │   ├── training_history.json       # 训练曲线数据
│   │   └── best_metrics.json           # 最佳指标
│   └── training_results/
│       ├── training_config.json        # 训练配置
│       └── training_history.png
├── exp_B/
│   └── ...
├── hyperparameter_sweep_report.html    # 最终 HTML 报告
└── hyperparameter_sweep_metrics.json   # 汇总指标 JSON
```

## 报告内容

生成的 HTML 报告包含 6 个板块：

1. **数据集统计** — 类别分布柱状图、Train/Val/Test 比例饼图、统计卡片
2. **超参数对比** — 准确率横向对比、F1 对比、Precision-Recall 散点图、参数表格
3. **逐类 F1 对比** — 所有配置的逐类 F1 分组柱状图
4. **雷达图** — Accuracy / Precision / Recall / F1(W) / F1(M) 多维对比
5. **逐配置详情** — 每组的训练曲线（Stage1/Stage2 分段）、混淆矩阵热力图、逐类指标表
6. **汇总表** — 所有配置的完整指标对比，最优配置高亮

报告为自包含 HTML 文件（图表以 base64 内嵌），直接浏览器打开即可查看。

## 断点续跑

脚本会自动检测已完成的实验（通过检查结果目录中的关键 JSON 文件），默认跳过已完成的实验。如果训练中途中断，重新运行即可从断点继续：

```bash
# 自动跳过已完成的实验，只跑剩余的
python -m tools.generate_training_report

# 如果某组实验结果有问题，强制重跑
python -m tools.generate_training_report --experiments C --force
```

## 使用建议

- 建议在 GPU 环境运行，CPU 训练每组可能需要 30-60+ 分钟
- 可以先跑 2-3 组关键配置（如 `--experiments A,E,F`），确认流程正常后再跑全部
- 训练完成后用 `--report` 可以反复调整报告样式而无需重新训练
