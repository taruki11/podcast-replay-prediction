"""
FM (Factorization Machines) 训练脚本 v2
推荐系统 CTR 预估经典模型

修正：
- 特征标准化（StandardScaler）
- FM forward 数值稳定性修正
- 用 GPU 加速（若可用）
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

log_path = r'D:\Pycharm_workplace\即刻笔试\fm_train_log.txt'
sys.stdout = Tee(log_path)

import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss

BASE     = r'D:\Pycharm_workplace\即刻笔试'
DATA_DIR = r'D:\Pycharm_workplace\即刻笔试\data\Iftech\dataset_algo2025'

# ============================================================
#  FM 模型（修正版 forward）
# ============================================================
class FM(nn.Module):
    """
    Factorization Machines (FM)
    
    公式（稠密输入）:
      y(x) = w0 + sum_i(w_i * x_i) 
              + 0.5 * sum_f [ (sum_i v_{i,f} * x_i)^2 - sum_i v_{i,f}^2 * x_i^2 ]
    
    实现要点：
    - 一阶项：w0 + <w, x>
    - 二阶项：用「简化公式」O(kn) 计算
    - 数值稳定：先做 sum，再平方，避免大数
    """
    def __init__(self, n_features, k=16):
        super().__init__()
        self.n_features = n_features
        self.k = k
        
        # 一阶
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w  = nn.Embedding(n_features, 1)
        
        # 二阶（隐向量）
        self.v  = nn.Embedding(n_features, k)
        
        # 初始化（Xavier 适合 sigmoid 输出）
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.xavier_uniform_(self.v.weight)
        
    def forward(self, x):
        """
        x: (batch, n_features) 稠密特征值（已标准化）
        
        二阶项用 einsum 高效计算（避免大张量广播）:
          second = 0.5 * sum_k [ (sum_i v_{i,k} x_i)^2 - sum_i v_{i,k}^2 x_i^2 ]
        """
        # 一阶：w0 + <w, x>
        first = self.w0 + (self.w.weight.squeeze(-1) * x).sum(dim=1)
        
        # 二阶：用 einsum 避免显式广播
        # vx[i,k] = sum_j v_{j,k} * x_{j,i}  (这里 j=特征,i=样本 → 转置处理)
        # 正确写法：对每个样本 i，计算 sum_k (sum_j v_{j,k}*x_{i,j})^2 - sum_j v_{j,k}^2 * x_{i,j}^2
        # 用矩阵运算：(x @ v) ^ 2 - sum_j (v^2 * x^2)
        vx = x @ self.v.weight                    # (batch, k)
        vx2 = (x ** 2) @ (self.v.weight ** 2)  # (batch, k)
        second = 0.5 * (vx ** 2 - vx2).sum(dim=1)
        
        return first + second


# ============================================================
#  训练单折
# ============================================================
def train_one_fold(X_tr, y_tr, X_val, y_val, params, device, fold):
    model = FM(X_tr.shape[1], k=params['k']).to(device)
    
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
def predict(model, X, device, batch_size=8192):
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
    log('  FM (Factorization Machines) — RecSys CTR Model')
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
    
    # --- 特征标准化（FM 必须做！）---
    log('\n[Preprocess] Manual StandardScaler (handle NaN + zero-var)...')
    
    X_tr = train[feature_cols].values.astype(np.float32)
    X_te = test[feature_cols].values.astype(np.float32)
    
    # 填 NaN（只有 2 行有 NaN，来自 is_default_* 等特征）
    X_tr = np.nan_to_num(X_tr, nan=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0)
    
    # 手动标准化
    mean = X_tr.mean(axis=0, keepdims=True)
    std  = X_tr.std(axis=0, keepdims=True) + 1e-8
    X     = (X_tr - mean) / std
    X_test = (X_te - mean) / std
    
    y = train['label'].values.astype(np.float32)
    
    log(f'X: {X.shape}, X_test: {X_test.shape}')
    log(f'X mean={X.mean():.3f}, std={X.std():.3f} (after scale)')
    log(f'Zero-var features: {(std < 1e-6).sum()}')
    
    del train, test
    gc.collect()
    
    # --- 设备 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'\n[Device] {device}')
    if device == 'cuda':
        log(f'  GPU: {torch.cuda.get_device_name(0)}')
    
    # --- 超参 ---
    params = dict(
        k=32,       # 隐向量维度（工业常用 16~64）
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=8192,
        n_epochs=80,
        early_stopping=15,
    )
    log(f'[Params] {params}')
    
    # --- 5-Fold ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    fold_aucs = []
    models = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        log(f'\n--- Fold {fold+1}/5 ---')
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model, best_auc = train_one_fold(
            X_tr, y_tr, X_val, y_val, params, device, fold+1
        )
        
        val_preds = predict(model, X_val, device)
        oof[val_idx] = val_preds
        
        auc = roc_auc_score(y_val, val_preds)
        fold_aucs.append(auc)
        models.append(model)
        log(f'  Fold {fold+1} AUC = {auc:.4f}')
        
        # 存模型
        torch.save(model.state_dict(),
                   os.path.join(BASE, f'fm_fold{fold+1}.pt'))
    
    # --- OOF 评估 ---
    oof_auc = roc_auc_score(y, oof)
    oof_ll  = log_loss(y, oof)
    
    log('\n' + '=' * 62)
    log(f'  OOF AUC    = {oof_auc:.4f}')
    log(f'  OOF LogLoss = {oof_ll:.4f}')
    auc_str = ', '.join(f'{a:.4f}' for a in fold_aucs)
    log(f'  Fold AUCs   = [{auc_str}]')
    log(f'  Mean AUC    = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}')
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
    result.to_csv(os.path.join(BASE, 'fm_result.csv'), index=False)
    log(f'\n  fm_result.csv saved ({result.shape})')
    
    oof_df = pd.DataFrame({'oof_pred': oof, 'label': y})
    oof_df.to_csv(os.path.join(BASE, 'fm_oof.csv'), index=False)
    
    info = dict(
        model='FM (Factorization Machines)',
        oof_auc=float(oof_auc),
        n_features=X.shape[1],
        embedding_dim=params['k'],
    )
    with open(os.path.join(BASE, 'fm_model_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    log('\n' + '=' * 62)
    log(f'  DONE — OOF AUC = {oof_auc:.4f}')
    log('=' * 62)

if __name__ == '__main__':
    main()
