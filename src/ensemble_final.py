"""快速融合：只试几个关键权重组合"""
import pandas as pd, numpy as np, os, json
from sklearn.metrics import roc_auc_score, log_loss

BASE = r'D:\Pycharm_workplace\即刻笔试'
oof_lgb = pd.read_csv(os.path.join(BASE, 'oof_predictions.csv'))['oof_pred'].values
oof_xgb = pd.read_csv(os.path.join(BASE, 'xgb_oof.csv'))['oof_pred'].values
oof_dfm = pd.read_csv(os.path.join(BASE, 'deepfm_oof.csv'))['oof_pred'].values
y = pd.read_csv(os.path.join(BASE, 'xgb_oof.csv'))['label'].values.astype(int)

print('Single: LGB=%.4f XGB=%.4f DFM=%.4f' % (
    roc_auc_score(y, oof_lgb), roc_auc_score(y, oof_xgb), roc_auc_score(y, oof_dfm)))

# 相关性
preds = np.column_stack([oof_lgb, oof_xgb, oof_dfm])
corr = np.corrcoef(preds.T)
print('Corr: LGB-XGB=%.4f LGB-DFM=%.4f XGB-DFM=%.4f' % (corr[0,1], corr[0,2], corr[1,2]))
print()

# 手动试几个组合
combos = [
    (0.80, 0.15, 0.05),
    (0.85, 0.10, 0.05),
    (0.90, 0.05, 0.05),
    (0.75, 0.20, 0.05),
    (0.70, 0.20, 0.10),
    (0.85, 0.15, 0.00),
    (0.80, 0.20, 0.00),
    (0.90, 0.10, 0.00),
]

best = (0, None)
for wl, wx, wd in combos:
    pred = wl * oof_lgb + wx * oof_xgb + wd * oof_dfm
    auc = roc_auc_score(y, pred)
    ll = log_loss(y, pred)
    print('LGB=%.2f XGB=%.2f DFM=%.2f -> AUC=%.4f LL=%.4f' % (wl, wx, wd, auc, ll))
    if auc > best[0]:
        best = (auc, (wl, wx, wd))

print()
print('BEST: LGB=%.2f XGB=%.2f DFM=%.2f -> AUC=%.4f' % (best[1][0], best[1][1], best[1][2], best[0]))

# 生成最终预测
print('\nGenerating final result...')
test_xgb = pd.read_csv(os.path.join(BASE, 'xgb_result.csv'))
test_lgb = pd.read_csv(os.path.join(BASE, 'result.csv'))
test_dfm = pd.read_csv(os.path.join(BASE, 'deepfm_result.csv'))

ids = test_xgb['id'].values
final = best[1][0] * test_lgb['label'].values + best[1][1] * test_xgb['label'].values + best[1][2] * test_dfm['label'].values

result = pd.DataFrame({'id': ids, 'label': final})
result.to_csv(os.path.join(BASE, 'ensemble_result.csv'), index=False)
print('ensemble_result.csv saved: %s' % str(result.shape))
print('pred range=[%.4f, %.4f], mean=%.4f' % (final.min(), final.max(), final.mean()))

info = {
    'best_ensemble': 'LGB+XGB+DeepFM weighted',
    'oof_auc': float(best[0]),
    'weights': [float(best[1][0]), float(best[1][1]), float(best[1][2])],
    'single_models': {
        'LightGBM': float(roc_auc_score(y, oof_lgb)),
        'XGBoost': float(roc_auc_score(y, oof_xgb)),
        'DeepFM': float(roc_auc_score(y, oof_dfm)),
    }
}
with open(os.path.join(BASE, 'ensemble_info.json'), 'w') as f:
    json.dump(info, f, indent=2)
print('Done!')
