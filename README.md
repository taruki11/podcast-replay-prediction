# 播客重复播放预测

算法笔试题，基于用户行为数据预测播客重复播放概率。

## 结果

- **模型**：LightGBM（88特征，5-Fold CV）
- **AUC**：0.8094
- **Accuracy**：0.7287

## 文件说明

```
src/
  features.py          # 特征工程（41个基础特征）
  cross_features.py    # 交叉特征生成（41→88个）
  train_model.py      # LightGBM训练
  train_xgboost.py   # XGBoost GPU训练
  train_deepfm.py    # DeepFM训练
  train_fm.py        # FM训练
  ensemble_final.py   # 模型融合
  plot_features.py    # 特征分布可视化
  eda.py             # 探索性数据分析

docs/
  analysis.md         # 分析报告

figs/
  fig1_label.png      # 标签分布
  fig2_age.png       # 年龄分布
  fig3_signals.png   # 特征重要性
  fig4_model_compare.png  # 模型对比
  fig5_errors.png     # 误差分析
```

## 运行方式

### 1. 下载数据

从ModelScope下载数据集`Iftech/dataset_algo2025`，放在`data/Iftech/dataset_algo2025/`。

### 2. 特征工程

```bash
# 基础特征（41个）
python src/features.py

# 交叉特征（41→88个）
python src/cross_features.py
```

### 3. 训练模型

```bash
# LightGBM（推荐）
python src/train_model.py

# XGBoost GPU（快速）
python src/train_xgboost.py
```

### 4. 生成提交文件

训练完成后自动生成`result.csv`。

## 特征工程

共构建88个特征，分为6组：
1. 用户画像特征（8个）
2. 播客内容特征（16个）
3. 上下文特征（5个）
4. 用户历史行为特征（6个）
5. 播客热度特征（6个）
6. 交叉特征（47个）

**创新点**：将数据质量问题的"缺陷"转化为"信号"（`profile_complete`、`is_default_host`等）。

## 模型训练

- **验证策略**：5-Fold Stratified CV
- **防过拟合**：Early Stopping + L1/L2正则化 + 特征/数据抽样
- **超参调优**：默认→增加复杂度→正则化，AUC从0.8019提升到0.8094

## 实验结果

| 模型 | AUC | Accuracy | F1 |
|------|-----|----------|----|
| LightGBM（88特征） | **0.8094** | 0.7287 | 0.7301 |
| XGBoost GPU | 0.8076 | 0.7265 | 0.7278 |
| DeepFM | 0.8011 | 0.7198 | 0.7212 |

## 分析报告

详见`docs/analysis.md`，包含：
- 数据理解和特征处理
- 模型训练过程
- 结果分析对比
- 工作亮点
- AI工具使用说明

## 依赖

```
pandas
numpy
scikit-learn
lightgbm
xgboost  # 可选
torch     # 可选
pyarrow
```

安装：`pip install -r requirements.txt`

---
*最后更新：2026-06-26*
