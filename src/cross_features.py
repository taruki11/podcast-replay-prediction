"""
交叉特征脚本 - 从 base parquet 构建剩余交叉特征
分两步走以避免内存溢出
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import os
import gc

OUTPUT_DIR = r'D:\Pycharm_workplace\即刻笔试'
DATA_DIR = r'D:\Pycharm_workplace\即刻笔试\data\Iftech\dataset_algo2025'


def build_cross_features():
    print('加载基础特征数据...')
    train = pd.read_parquet(os.path.join(OUTPUT_DIR, 'train_base.parquet'))
    test = pd.read_parquet(os.path.join(OUTPUT_DIR, 'test_base.parquet'))

    with open(os.path.join(OUTPUT_DIR, 'feature_cols_base.txt'), 'r') as f:
        base_feature_cols = [line.strip() for line in f if line.strip()]

    print(f'  train: {train.shape}, test: {test.shape}, base features: {len(base_feature_cols)}')

    # 内存优化
    for df in [train, test]:
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
        for col in df.select_dtypes(include=['int64']).columns:
            if col not in ['label']:
                df[col] = df[col].astype(np.int32)
    gc.collect()
    print(f'  train 内存: {train.memory_usage(deep=True).sum()/1024**2:.0f} MB')

    feature_cols = list(base_feature_cols)
    new_feats = []

    # ========== 1. Episode × 上下文 热度交叉 ==========
    print('  Episode × Context 交叉...')
    for ctx_col, prefix in [('tab_name', 'ep_tab'), ('scene_name', 'ep_scene'),
                             ('entrance_type', 'ep_entrance')]:
        stats = train.groupby(['episode_id', ctx_col])['label'].agg(
            repeat_rate='mean', play_count='size'
        ).reset_index()
        stats.columns = ['episode_id', ctx_col, f'{prefix}_repeat_rate', f'{prefix}_play_count']

        train = train.merge(stats, on=['episode_id', ctx_col], how='left')
        test = test.merge(stats, on=['episode_id', ctx_col], how='left')
        del stats
        gc.collect()

        for col in [f'{prefix}_repeat_rate', f'{prefix}_play_count']:
            train[col] = train[col].fillna(0.5 if 'rate' in col else 0).astype(np.float32)
            test[col] = test[col].fillna(0.5 if 'rate' in col else 0).astype(np.float32)
            new_feats.append(col)

    # ========== 2. Category × 上下文 热度交叉 ==========
    print('  Category × Context 交叉...')
    for ctx_col, prefix in [('tab_name', 'cat_tab'), ('scene_name', 'cat_scene')]:
        stats = train.groupby(['primary_cat', ctx_col])['label'].agg(
            repeat_rate='mean', play_count='size'
        ).reset_index()
        stats.columns = ['primary_cat', ctx_col, f'{prefix}_repeat_rate', f'{prefix}_play_count']

        train = train.merge(stats, on=['primary_cat', ctx_col], how='left')
        test = test.merge(stats, on=['primary_cat', ctx_col], how='left')
        del stats
        gc.collect()

        for col in [f'{prefix}_repeat_rate', f'{prefix}_play_count']:
            train[col] = train[col].fillna(0.5 if 'rate' in col else 0).astype(np.float32)
            test[col] = test[col].fillna(0.5 if 'rate' in col else 0).astype(np.float32)
            new_feats.append(col)

    # ========== 3. 用户偏好类目/主播 ==========
    print('  用户偏好类目/主播...')
    user_cat_vc = train.groupby(['uid', 'primary_cat']).size().reset_index(name='cnt')
    user_cat_vc = user_cat_vc.sort_values('cnt', ascending=False).drop_duplicates('uid')
    user_cat_vc = user_cat_vc[['uid', 'primary_cat']].rename(columns={'primary_cat': 'user_top_cat'})

    user_host_vc = train.groupby(['uid', 'primary_host']).size().reset_index(name='cnt')
    user_host_vc = user_host_vc.sort_values('cnt', ascending=False).drop_duplicates('uid')
    user_host_vc = user_host_vc[['uid', 'primary_host']].rename(columns={'primary_host': 'user_top_host'})

    train = train.merge(user_cat_vc, on='uid', how='left')
    test = test.merge(user_cat_vc, on='uid', how='left')
    del user_cat_vc
    gc.collect()

    train = train.merge(user_host_vc, on='uid', how='left')
    test = test.merge(user_host_vc, on='uid', how='left')
    del user_host_vc
    gc.collect()

    for df in [train, test]:
        df['cat_match_user_pref'] = (df['primary_cat'] == df['user_top_cat']).astype(np.float32)
        df['host_match_user_pref'] = (df['primary_host'] == df['user_top_host']).astype(np.float32)
        df['user_top_cat'] = df['user_top_cat'].fillna('unknown')
        df['user_top_host'] = df['user_top_host'].fillna('unknown')

    new_feats += ['cat_match_user_pref', 'host_match_user_pref']

    # 编码 user_top_cat, user_top_host
    for col in ['user_top_cat', 'user_top_host']:
        all_vals = pd.concat([train[col], test[col]]).unique()
        mapping = {v: i for i, v in enumerate(all_vals)}
        train[col + '_enc'] = train[col].map(mapping).fillna(-1).astype(np.int32)
        test[col + '_enc'] = test[col].map(mapping).fillna(-1).astype(np.int32)
    new_feats += ['user_top_cat_enc', 'user_top_host_enc']

    # 组合匹配分数
    for df in [train, test]:
        df['full_match_score'] = (
            df['cat_match_user_pref'] +
            df['host_match_user_pref'] +
            df['tab_match'] +
            df['scene_match'] +
            df['entrance_match']
        ).astype(np.float32)
    new_feats += ['full_match_score']

    # ========== 保存 ==========
    all_feature_cols = feature_cols + new_feats
    print(f'\n交叉特征完成！新增 {len(new_feats)} 个，总计 {len(all_feature_cols)} 个')
    print(f'  train: {train.shape}, test: {test.shape}')

    # 只保存特征列 + label
    train_save = feature_cols + ['label'] + new_feats
    test_save = feature_cols + new_feats

    train[train_save].to_parquet(os.path.join(OUTPUT_DIR, 'train_featured.parquet'), index=False)
    test[test_save].to_parquet(os.path.join(OUTPUT_DIR, 'test_featured.parquet'), index=False)

    with open(os.path.join(OUTPUT_DIR, 'feature_cols.txt'), 'w') as f:
        f.write('\n'.join(all_feature_cols))

    print(f'  train_featured.parquet: {os.path.getsize(os.path.join(OUTPUT_DIR, "train_featured.parquet"))/1024/1024:.1f} MB')
    print(f'  test_featured.parquet: {os.path.getsize(os.path.join(OUTPUT_DIR, "test_featured.parquet"))/1024/1024:.1f} MB')


if __name__ == '__main__':
    build_cross_features()
