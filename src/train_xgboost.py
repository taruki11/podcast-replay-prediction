import sys, os, warnings, json, time, gc
warnings.filterwarnings('ignore')

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

sys.stdout = Tee(r'D:\Pycharm_workplace\即刻笔试\xgb_train_log.txt')

import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, f1_score

OUTPUT_DIR = r'D:\Pycharm_workplace\即刻笔试'
DATA_DIR = r'D:\Pycharm_workplace\即刻笔试\data\Iftech\dataset_algo2025'

def log(msg):
    print('[%s] %s' % (datetime.now().strftime('%H:%M:%S'), msg))

def main():
    log('=' * 60)
    log('XGBoost training started')
    log('=' * 60)

    # Load feature cols
    with open(os.path.join(OUTPUT_DIR, 'feature_cols.txt'), 'r') as f:
        all_feature_cols = [line.strip() for line in f if line.strip()]
    feature_cols = all_feature_cols[:51]
    log('Features: %d (with cross features)' % len(feature_cols))

    # Load data (with cross features)
    load_cols = feature_cols + ['label']
    train = pd.read_parquet(os.path.join(OUTPUT_DIR, 'train_featured.parquet'), columns=load_cols)
    test = pd.read_parquet(os.path.join(OUTPUT_DIR, 'test_featured.parquet'), columns=feature_cols)
    test_ids = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), usecols=['id'])['id']

    log('Train: %s, positive: %.4f' % (str(train.shape), train['label'].mean()))
    log('Test: %s' % str(test.shape))

    # To numpy
    y = train['label'].values.astype(np.int8)
    train.drop(columns=['label'], inplace=True)
    gc.collect()

    for col in train.select_dtypes(include=['float64']).columns:
        train[col] = train[col].astype(np.float32)
    for col in test.select_dtypes(include=['float64']).columns:
        test[col] = test[col].astype(np.float32)

    X = train[feature_cols].values.astype(np.float32)
    X_test = test[feature_cols].values.astype(np.float32)

    del train, test
    gc.collect()

    log('X: %s, X_test: %s' % (str(X.shape), str(X_test.shape)))
    log('Memory: %.0f MB' % (X.nbytes / 1024**2))

    # XGBoost params - XGBoost 3.x uses 'device' param for GPU
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 10,
        'subsample': 0.6,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'gamma': 0.1,
        'tree_method': 'hist',
        'device': 'cuda:0',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'max_bin': 128,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    fold_aucs = []
    models = []

    log('Starting 5-Fold CV...')
    start = time.time()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        log('--- Fold %d/5 ---' % (fold+1))
        t0 = time.time()

        dtrain = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
        dval = xgb.DMatrix(X[val_idx], label=y[val_idx])

        model = xgb.train(
            params, dtrain,
            num_boost_round=3000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=150,
            verbose_eval=False,
        )

        val_pred = model.predict(dval)
        oof_preds[val_idx] = val_pred

        auc = roc_auc_score(y[val_idx], val_pred)
        ll = log_loss(y[val_idx], val_pred)
        fold_aucs.append(auc)

        log('  AUC=%.4f, LogLoss=%.4f' % (auc, ll))
        log('  Best iter: %d, time: %.1fs' % (model.best_iteration, time.time() - t0))

        models.append(model)
        del dtrain, dval
        gc.collect()

    # OOF eval
    oof_auc = roc_auc_score(y, oof_preds)
    oof_ll = log_loss(y, oof_preds)
    oof_f1 = f1_score(y, (oof_preds > 0.5).astype(int))

    log('\n' + '=' * 60)
    log('OOF AUC: %.4f' % oof_auc)
    log('OOF LogLoss: %.4f' % oof_ll)
    log('OOF F1: %.4f' % oof_f1)
    log('Fold AUCs: %s' % str(['%.4f' % a for a in fold_aucs]))
    log('Mean: %.4f +/- %.4f' % (np.mean(fold_aucs), np.std(fold_aucs)))
    log('Total time: %.1fs' % (time.time() - start))

    # Feature importance
    imp = {}
    for col in feature_cols:
        imp[col] = 0.0
    for model in models:
        for k, v in model.get_score(importance_type='gain').items():
            imp[k] = imp.get(k, 0) + v
    for k in imp:
        imp[k] /= len(models)

    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    log('\nTop 20 features:')
    for i, (col, val) in enumerate(sorted_imp[:20]):
        log('  %2d. %-35s: %.1f' % (i+1, col, val))

    # Save feature importance
    imp_df = pd.DataFrame({'feature': list(imp.keys()), 'importance': list(imp.values())})
    imp_df = imp_df.sort_values('importance', ascending=False)
    imp_df.to_csv(os.path.join(OUTPUT_DIR, 'xgb_feature_importance.csv'), index=False)

    # Predict test
    log('\nPredicting test set...')
    test_preds = np.zeros(len(X_test))
    for model in models:
        dtest = xgb.DMatrix(X_test)
        test_preds += model.predict(dtest)
    test_preds /= len(models)

    log('Test pred range: [%.4f, %.4f]' % (test_preds.min(), test_preds.max()))
    log('Test pred mean: %.4f' % test_preds.mean())

    result = pd.DataFrame({'id': test_ids, 'label': test_preds})
    result.to_csv(os.path.join(OUTPUT_DIR, 'xgb_result.csv'), index=False)
    log('xgb_result.csv saved: %s' % str(result.shape))

    # Save OOF
    oof_df = pd.DataFrame({'oof_pred': oof_preds, 'label': y})
    oof_df.to_csv(os.path.join(OUTPUT_DIR, 'xgb_oof.csv'), index=False)

    model_info = {'model': 'XGBoost', 'oof_auc': float(oof_auc), 'n_features': len(feature_cols)}
    with open(os.path.join(OUTPUT_DIR, 'xgb_model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)

    log('\n' + '=' * 60)
    log('Done! OOF AUC: %.4f' % oof_auc)
    log('=' * 60)

if __name__ == '__main__':
    main()
