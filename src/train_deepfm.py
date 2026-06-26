"""
DeepFM 训练脚本
推荐系统 CTR 预估主流模型（FM + DNN 并联）

架构：
  - Wide 部分：FM（一阶 + 二阶特征交叉）
  - Deep 部分：DNN（高阶特征交叉）
  - 输出：sigmoid(wide + deep)

工业界常用，拟合能力远强于纯 FM。
"""
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

log_path = r'D:\Pycharm_workplace\即刻笔试\deepfm_train_log.txt'
sys.stdout = Tee(log_path)

import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

BASE     = r'D:\Pycharm_workplace\即刻笔试'
DATA_DIR = r'D:\Pycharm_workplace\即刻笔试\data\Iftech\dataset_algo2025'


# ============================================================
#  DeepFM 模型定义
# ============================================================
class DeepFM(nn.Module):
    """
    DeepFM: A Factorization-Machine based Neural Network
    
    架构：
      Input → [FM 部分] ──┐
                       [DNN 部分] ──┤ → Sigmoid → Output
                                        
      FM 部分（Wide）：
        - 一阶：w0 + sum(wi * xi)
        - 二阶：0.5 * sum_k[(sum_i vi_k * xi)^2 - sum_i vi_k^2 * xi^2]
        
      DNN 部分（Deep）：
        - 输入：x（标准化后的稠密特征）
        - 隐藏层：MLP（ReLU + Dropout）
        - 输出：1 个神经元（logit）
        
    输出：
        y = sigmoid( FM_logit + DNN_logit )
    """
    def __init__(self, n_features, k=16, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        self.n_features = n_features
        self.k = k
        
        # ========== FM 部分（Wide）==========
        self.fm_w0 = nn.Parameter(torch.zeros(1))          # w0
        self.fm_w  = nn.Embedding(n_features, 1)          # wi（一阶权重）
        self.fm_v  = nn.Embedding(n_features, k)          # vi（二阶隐向量）
        nn.init.xavier_uniform_(self.fm_w.weight)
        nn.init.xavier_uniform_(self.fm_v.weight)
        
        # ========== DNN 部分（Deep）==========
        layers = []
        prev_dim = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))   # DNN 输出 1 个 logit
        self.dnn = nn.Sequential(*layers)
        
        # ========== 输出层 ==========
        # FM logit + DNN logit（在 forward 里加）
        
    def forward(self, x):
        """
        x: (batch, n_features) 稠密特征值（已标准化）
        """
        # --- FM 部分 ---
        # 一阶
        fm_first = self.fm_w0 + (self.fm_w.weight.squeeze(-1) * x).sum(dim=1)
        
        # 二阶（简化公式）
        vx   = x @ self.fm_v.weight                # (batch, k)
        vx2  = (x ** 2) @ (self.fm_v.weight ** 2)  # (batch, k)
        fm_second = 0.5 * (vx ** 2 - vx2).sum(dim=1)
        
        fm_logit = fm_first + fm_second   # (batch,)
        
        # --- DNN 部分 ---
        dnn_logit = self.dnn(x).squeeze(-1)   # (batch,)
        
        # --- 合并 ---
        logits = fm_logit + dnn_logit   # (batch,)
        return logits



# ============================================================
#  训练单折
# ============================================================
def train_one_fold(X_tr, y_tr, X_val, y_val, params, device, fold):
    model = DeepFM(
        X_tr.shape[1],
        k=params['k'],
        hidden_dims=params['hidden_dims'],
        dropout=params['dropout']
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params['n_epochs']
    )
    criterion = nn.BCEWithLogitsLoss()
    
    X_tr_t  = torch.from_numpy(X_tr).to(device)
    y_tr_t  = torch.from_numpy(y_tr).float().to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)
    
    best_auc = 0.0
    best_state = None
    patience = 0
    
    for epoch in range(params['n_epochs']):
        # — train —
        model.train()
        perm = torch.randperm(X_tr_t.size(0), device=device)
        n_batches = 0
        total_loss = 0.0
        
        for i in range(0, X_tr_t.size(0), params['batch_size']):
            idx = perm[i:i+params['batch_size']]
            bx = X_tr_t[idx]
            by = y_tr_t[idx]
            
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # — eval —
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs  = torch.sigmoid(val_logits)
            val_loss   = criterion(val_logits, y_val_t).item()
            val_auc    = roc_auc_score(y_val, val_probs.cpu().numpy())
        
        if (epoch + 1) % 5 == 0 or val_auc > best_auc:
            log(f'  Fold {fold} | Epoch {epoch+1:3d} | '
                f'TrLoss {total_loss/n_batches:.4f} | '
                f'ValLoss {val_loss:.4f} | ValAUC {val_auc:.4f}')
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        
        if patience >= params['early_stopping']:
            log(f'  Fold {fold} | Early stopping at epoch {epoch+1}')
            break
    
    model.load_state_dict(best_state)
    return model, best_auc


# ============================================================
#  预测
# ============================================================
def predict(model, X, device, batch_size=16384):
    model.eval()
    preds = []
    X_t = torch.from_numpy(X).to(device)
    with torch.no_grad():
        for i in range(0, X_t.size(0), batch_size):
            batch = X_t[i:i+batch_size]
            logits = model(batch)
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())
    return np.concatenate(preds)


# ============================================================
#  主流程
# ============================================================
def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}')

def main():
    log('=' * 62)
    log('  DeepFM — RecSys CTR Model (FM + DNN)')
    log('=' * 62)
    
    # --- 特征名 ---
    with open(os.path.join(BASE, 'feature_cols.txt')) as f:
        feature_cols = [l.strip() for l in f if l.strip()][:41]
    log(f'Features: {len(feature_cols)}')
    
    # --- 加载数据 ---
    train = pd.read_parquet(
        os.path.join(BASE, 'train_base.parquet'),
        columns=feature_cols + ['label']
    )
    test  = pd.read_parquet(
        os.path.join(BASE, 'test_base.parquet'),
        columns=feature_cols
    )
    test_ids = pd.read_csv(
        os.path.join(DATA_DIR, 'test.csv'), usecols=['id']
    )['id']
    
    log(f'Train: {train.shape}, pos_rate={train["label"].mean():.4f}')
    log(f'Test:  {test.shape}')
    
    # --- 特征标准化 ---
    log('\n[Preprocess] StandardScaler...')
    X_tr = train[feature_cols].values.astype(np.float32)
    X_te = test[feature_cols].values.astype(np.float32)
    X_tr = np.nan_to_num(X_tr, nan=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0)
    
    mean = X_tr.mean(axis=0, keepdims=True)
    std  = X_tr.std(axis=0, keepdims=True) + 1e-8
    X     = (X_tr - mean) / std
    X_test = (X_te - mean) / std
    y = train['label'].values.astype(np.float32)
    
    log(f'X: {X.shape}, X_test: {X_test.shape}')
    log(f'X mean={X.mean():.3f}, std={X.std():.3f}')
    
    del train, test
    gc.collect()
    
    # --- 设备 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'\n[Device] {device}')
    if device == 'cuda':
        log(f'  GPU: {torch.cuda.get_device_name(0)}')
        log(f'  Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB')
    
    # --- 超参 ---
    params = dict(
        k=16,                 # FM 隐向量维度
        hidden_dims=[256, 128, 64],  # DNN 隐藏层
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=16384,
        n_epochs=80,
        early_stopping=15,
    )
    log(f'[Params] {params}')
    
    # --- 5-Fold ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    fold_aucs = []
    models = []
    
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        log(f'\n--- Fold {fold+1}/5 ---')
        X_tr_f, X_val_f = X[tr_idx], X[val_idx]
        y_tr_f, y_val_f = y[tr_idx], y[val_idx]
        
        model, best_auc = train_one_fold(
            X_tr_f, y_tr_f, X_val_f, y_val_f, params, device, fold+1
        )
        
        val_preds = predict(model, X_val_f, device)
        oof[val_idx] = val_preds
        
        auc = roc_auc_score(y_val_f, val_preds)
        fold_aucs.append(auc)
        models.append(model)
        log(f'  Fold {fold+1} AUC = {auc:.4f}')
        
        torch.save(model.state_dict(),
                   os.path.join(BASE, f'deepfm_fold{fold+1}.pt'))
    
    # --- OOF 评估 ---
    oof_auc = roc_auc_score(y, oof)
    oof_ll  = log_loss(y, oof)
    
    elapsed = time.time() - t0
    log('\n' + '=' * 62)
    log(f'  OOF AUC    = {oof_auc:.4f}')
    log(f'  OOF LogLoss = {oof_ll:.4f}')
    log(f'  Fold AUCs   = {[f"{a:.4f}" for a in fold_aucs]}')
    log(f'  Mean AUC    = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}')
    log(f'  Time        = {elapsed/60:.1f} min')
    log('=' * 62)
    
    # --- 测试集预测 ---
    log('\n[Predict] Test set (5-fold avg)...')
    test_preds = np.zeros(len(X_test))
    for i, model in enumerate(models):
        model.to(device)
        pred = predict(model, X_test, device)
        test_preds += pred / 5
        log(f'  Fold {i+1} test_pred mean = {pred.mean():.4f}')
    
    log(f'  Test pred range = [{test_preds.min():.4f}, {test_preds.max():.4f}]')
    log(f'  Test pred mean = {test_preds.mean():.4f}')
    
    # --- 保存 ---
    result = pd.DataFrame({'id': test_ids, 'label': test_preds})
    result.to_csv(os.path.join(BASE, 'deepfm_result.csv'), index=False)
    log(f'\n  deepfm_result.csv saved ({result.shape})')
    
    oof_df = pd.DataFrame({'oof_pred': oof, 'label': y})
    oof_df.to_csv(os.path.join(BASE, 'deepfm_oof.csv'), index=False)
    
    info = dict(
        model='DeepFM (FM + DNN)',
        oof_auc=float(oof_auc),
        n_features=X.shape[1],
        k=params['k'],
        hidden_dims=params['hidden_dims'],
    )
    with open(os.path.join(BASE, 'deepfm_model_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    log('\n' + '=' * 62)
    log(f'  DONE — OOF AUC = {oof_auc:.4f}')
    log('=' * 62)


if __name__ == '__main__':
    main()
