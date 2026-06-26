"""
模型训练脚本
LightGBM + 5-Fold Stratified CV
配置：73 特征（41 base + 32 交叉）+ 3000 轮
适用于 16GB+ 内存机器
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()

sys.stdout = Tee(r'D:\Pycharm_workplace\即刻笔试\train_log2.txt')

import pandas as pd
import numpy as np
import os
import gc
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, f1_score
import json

OUTPUT_DIR = r'D:\Pycharm_workplace\即刻笔试'
DATA_DIR = r'D:\Pycharm_workplace\即刻笔试\data\Iftech\dataset_algo2025'


def load_featured_data():
    """加载特征工程后的数据"""
    print('加载特征数据...')

    with open(os.path.join(OUTPUT_DIR, 'feature_cols.txt'), 'r') as f:
        all_feature_cols = [line.strip() for line in f if line.strip()]

    # 41 基础特征（适配 6GB 可用内存 + 3000 轮充分训练）
    feature_cols = all_feature_cols[:41]

    load_cols_train = feature_cols + ['label']
    load_cols_test = feature_cols
    train = pd.read_parquet(os.path.join(OUTPUT_DIR, 'train_base.parquet'), columns=load_cols_train)
    test = pd.read_parquet(os.path.join(OUTPUT_DIR, 'test_base.parquet'), columns=load_cols_test)

    test_ids = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), usecols=['id'])['id']

    print(f'  train: {train.shape}, test: {test.shape}, features: {len(feature_cols)}')

    for df in [train, test]:
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)

    gc.collect()
    print(f'  train 内存: {train.memory_usage(deep=True).sum()/1024**2:.0f} MB')

    return train, test, feature_cols, test_ids


def train_lgbm(train, feature_cols):
    """LightGBM 5-Fold CV 训练"""
    y = train['label'].values.astype(np.int8)

    print('转换数据为 numpy 数组...')
    X_all = np.empty((len(train), len(feature_cols)), dtype=np.float64)
    for i, col in enumerate(feature_cols):
        X_all[:, i] = train[col].fillna(-1).values
        if i % 20 == 0:
            print(f'  {i}/{len(feature_cols)} 列...')

    train.drop(columns=list(train.columns), inplace=True)
    gc.collect()

    print(f'\n数据: X={X_all.shape}, dtype={X_all.dtype}, y={y.shape}, positive_rate={y.mean():.4f}')
    print(f'  X 内存: {X_all.nbytes/1024**2:.0f} MB')

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 100,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    # 尝试 GPU 模式
    try:
        test_params = dict(params, device='gpu', gpu_platform_id=0, gpu_device_id=0)
        test_ds = lgb.Dataset(X_all[:1000], label=y[:1000], feature_name=feature_cols)
        lgb.train(test_params, test_ds, num_boost_round=1)
        params['device'] = 'gpu'
        params['gpu_platform_id'] = 0
        params['gpu_device_id'] = 0
        print('  GPU 模式启用成功！')
        del test_ds
        gc.collect()
    except Exception as e:
        print(f'  GPU 不可用，回退 CPU: {e}')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    fold_aucs = []
    fold_logloss = []
    models = []

    print('\n开始 5-Fold CV 训练（3000 轮）...')
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y)):
        X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=3000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)]
        )

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred

        auc = roc_auc_score(y_val, val_pred)
        ll = log_loss(y_val, val_pred)
        fold_aucs.append(auc)
        fold_logloss.append(ll)
        models.append(model)

        print(f'  Fold {fold+1}: AUC={auc:.4f}, LogLoss={ll:.4f}, best_iter={model.best_iteration}')

        del X_tr, X_val, train_data, val_data
        gc.collect()

    del X_all
    gc.collect()

    # OOF 评估
    oof_auc = roc_auc_score(y, oof_preds)
    oof_ll = log_loss(y, oof_preds)
    oof_f1 = f1_score(y, (oof_preds > 0.5).astype(int))

    print(f'\n===== OOF 结果 =====')
    print(f'  AUC:      {oof_auc:.4f}')
    print(f'  LogLoss:  {oof_ll:.4f}')
    print(f'  F1:       {oof_f1:.4f}')
    print(f'  Fold AUC: {[f"{a:.4f}" for a in fold_aucs]}')
    print(f'  Mean AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}')

    # 特征重要性
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.mean([m.feature_importance(importance_type='gain') for m in models], axis=0)
    }).sort_values('importance', ascending=False)

    print(f'\n===== Top 20 特征重要性 =====')
    for _, row in importance.head(20).iterrows():
        print(f'  {row["feature"]:35s}: {row["importance"]:.1f}')

    return models, oof_preds, oof_auc, importance


def predict_test(models, test, feature_cols):
    """5-Fold 模型平均预测"""
    print('\n生成测试集预测...')
    X_test = test[feature_cols]

    test_preds = np.zeros(len(X_test))
    for model in models:
        test_preds += model.predict(X_test)
    test_preds /= len(models)

    print(f'  预测概率范围: [{test_preds.min():.4f}, {test_preds.max():.4f}]')
    print(f'  预测概率均值: {test_preds.mean():.4f}')
    print(f'  预测 >0.5 比例: {(test_preds > 0.5).mean():.4f}')

    return test_preds


def save_result(test_ids, test_preds):
    """保存 result.csv"""
    result = pd.DataFrame({
        'id': test_ids,
        'label': test_preds
    })
    result.to_csv(os.path.join(OUTPUT_DIR, 'result.csv'), index=False)
    print(f'\nresult.csv 已保存: {result.shape}')
    print(result.head(10))


def main():
    train, test, feature_cols, test_ids = load_featured_data()

    train_labels = train['label'].values

    models, oof_preds, oof_auc, importance = train_lgbm(train, feature_cols)

    # 保存 OOF
    oof_df = pd.DataFrame({'oof_pred': oof_preds, 'label': train_labels})
    oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_predictions.csv'), index=False)

    importance.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

    test_preds = predict_test(models, test, feature_cols)

    save_result(test_ids, test_preds)

    model_info = {
        'oof_auc': float(oof_auc),
        'n_features': len(feature_cols),
        'features': feature_cols,
    }
    with open(os.path.join(OUTPUT_DIR, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    print('\n训练完成！')


if __name__ == '__main__':
    main()
