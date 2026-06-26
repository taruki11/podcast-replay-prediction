"""
特征工程脚本
从原始数据构建 5 组特征：用户画像、播客内容、上下文、用户历史行为、播客热度
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import os
from datetime import datetime

DATA_DIR = r'D:\Pycharm_workplace\即刻笔试\data\Iftech\dataset_algo2025'
OUTPUT_DIR = r'D:\Pycharm_workplace\即刻笔试'

def load_data():
    """加载所有原始数据"""
    print('[1/6] 加载数据...')
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    user_feat = pd.read_csv(os.path.join(DATA_DIR, 'user_feature.csv'))
    ep_feat = pd.read_csv(os.path.join(DATA_DIR, 'episode_feature.csv'))
    ep_add = pd.read_csv(os.path.join(DATA_DIR, 'episode_additional.csv'))
    print(f'  train: {train.shape}, test: {test.shape}')
    print(f'  user_feat: {user_feat.shape}, ep_feat: {ep_feat.shape}, ep_add: {ep_add.shape}')
    return train, test, user_feat, ep_feat, ep_add


def build_user_features(user_feat):
    """A. 用户画像特征"""
    print('[2/6] 构建用户画像特征...')
    uf = user_feat.copy()

    # age 分段
    def age_bin(a):
        if a <= 0: return 0  # 未知
        if a < 18: return 1
        if a < 25: return 2
        if a < 35: return 3
        if a < 45: return 4
        if a < 55: return 5
        return 6
    uf['age_bin'] = uf['age'].apply(age_bin)

    # sex 编码
    uf['sex_enc'] = uf['sex'].map({'MAN': 1, 'WOMAN': 2}).fillna(0).astype(int)

    # rg_source 编码
    uf['rg_source_enc'] = uf['rg_source'].astype(str).astype('category').cat.codes

    # address 是否真实
    uf['address_real'] = (uf['address'] != 1).astype(int)

    # 用户资料完整度
    uf['profile_complete'] = ((uf['age'] > 0) & (uf['sex'].notna()) & (uf['sex'] != '')).astype(int)

    # 注册时长（天）
    uf['rg_date'] = pd.to_datetime(uf['rg_date'], errors='coerce')
    uf['exp_date'] = pd.to_datetime(uf['exp_date'], errors='coerce')
    ref_date = datetime(2025, 6, 1)  # 参考日期
    uf['reg_days'] = (ref_date - uf['rg_date']).dt.days
    uf['reg_days'] = uf['reg_days'].fillna(-1).astype(int)

    # 账户剩余时长
    uf['exp_days'] = (uf['exp_date'] - uf['rg_date']).dt.days
    uf['exp_days'] = uf['exp_days'].fillna(-1).astype(int)

    # age 异常值处理 (>100 视为缺失)
    uf.loc[uf['age'] > 100, 'age'] = 0
    uf.loc[uf['age'] > 100, 'age_bin'] = 0

    return uf[['uid', 'age_bin', 'sex_enc', 'rg_source_enc', 'address_real',
               'profile_complete', 'reg_days', 'exp_days']]


def build_episode_features(ep_feat, ep_add):
    """B. 播客内容特征"""
    print('[3/6] 构建播客内容特征...')

    # === 从 ep_feat ===
    ef = ep_feat.copy()

    # duration 秒
    ef['duration_sec'] = ef['duration'] / 1000

    # duration 分段
    def dur_bin(d):
        if d < 180: return 0   # <3min
        if d < 300: return 1   # 3-5min
        if d < 600: return 2   # 5-10min
        if d < 900: return 3   # 10-15min
        return 4               # >15min
    ef['duration_bin'] = ef['duration_sec'].apply(dur_bin)

    # 主类目
    ef['primary_cat'] = ef['category_ids'].fillna('').str.split('|').str[0]
    ef.loc[ef['primary_cat'] == '', 'primary_cat'] = np.nan

    # 类目数量
    ef['cat_count'] = ef['category_ids'].fillna('').apply(lambda x: len(x.split('|')) if x else 0)

    # 是否热门类目 (top 5)
    top_cats = {'CID0141', 'CID0066', 'CID0107', 'CID0003', 'CID0047'}
    ef['is_top_cat'] = ef['primary_cat'].isin(top_cats).astype(int)

    # host 数量
    ef['host_count'] = ef['host'].fillna('').apply(lambda x: len(x.split('|')) if x else 0)
    # 是否默认 host
    ef['is_default_host'] = ef['host'].fillna('').str.contains('W000000').astype(int)
    # 主 host
    ef['primary_host'] = ef['host'].fillna('').str.split('|').str[0]
    ef.loc[ef['primary_host'] == '', 'primary_host'] = np.nan

    # producer 数量 & 是否默认
    ef['producer_count'] = ef['producer'].fillna('').apply(lambda x: len(x.split('|')) if x else 0)
    ef['is_default_producer'] = ef['producer'].fillna('').str.contains('W111914').astype(int)

    # writer 数量 & 是否默认
    ef['writer_count'] = ef['writer'].fillna('').apply(lambda x: len(x.split('|')) if x else 0)
    ef['is_default_writer'] = ef['writer'].fillna('').str.contains('W111914').astype(int)

    # language 编码
    ef['language_enc'] = ef['language'].fillna(-1).astype(int)
    ef['lang_is_unknown'] = (ef['language_enc'] == -1).astype(int)
    ef['lang_is_zh'] = (ef['language_enc'] == 24).astype(int)

    ep_out = ef[['episode_id', 'duration_sec', 'duration_bin', 'primary_cat', 'cat_count',
                 'is_top_cat', 'host_count', 'is_default_host', 'primary_host',
                 'producer_count', 'is_default_producer', 'writer_count', 'is_default_writer',
                 'language_enc', 'lang_is_unknown', 'lang_is_zh']]

    # === 从 ep_add ===
    ea = ep_add.copy()

    # title 长度（T-token 数量）
    ea['title_len'] = ea['title'].fillna('').apply(lambda x: len(x.split('|')) if x else 0)

    # uuid 组大小
    uuid_sizes = ea.groupby('uuid')['episode_id'].count().reset_index()
    uuid_sizes.columns = ['uuid', 'uuid_group_size']
    ea = ea.merge(uuid_sizes, on='uuid', how='left')

    # uuid 是否含 nan
    ea['uuid_has_nan'] = ea['uuid'].str.contains('nan', na=False).astype(int)

    # uuid 是否 singleton
    ea['uuid_is_singleton'] = (ea['uuid_group_size'] == 1).astype(int)

    ep_add_out = ea[['episode_id', 'title_len', 'uuid_group_size', 'uuid_has_nan', 'uuid_is_singleton']]

    return ep_out, ep_add_out


def build_context_features(train, test):
    """C. 上下文特征"""
    print('[4/6] 构建上下文特征...')

    for df in [train, test]:
        # tab_name 填补缺失
        df['tab_name'] = df['tab_name'].fillna('unknown')
        # scene_name 填补缺失
        df['scene_name'] = df['scene_name'].fillna('unknown')
        # entrance_type 填补缺失
        df['entrance_type'] = df['entrance_type'].fillna('unknown')

        # 上下文完整度
        df['ctx_complete'] = (
            (df['tab_name'] != 'unknown') &
            (df['scene_name'] != 'unknown') &
            (df['entrance_type'] != 'unknown')
        ).astype(int)

    return train, test


def build_user_history_features(train):
    """D. 用户历史行为特征（基于训练集全局统计，用于测试集推理）"""
    print('[5/6] 构建用户历史行为特征...')

    # 用户播放次数
    user_play_count = train.groupby('uid').size().reset_index(name='user_play_count')

    # 用户重复播放率
    user_repeat_rate = train.groupby('uid')['label'].mean().reset_index(name='user_repeat_rate')

    # 用户偏好 tab_name（出现最多的）
    user_top_tab = train.groupby('uid')['tab_name'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown').reset_index()
    user_top_tab.columns = ['uid', 'user_top_tab']

    # 用户偏好 entrance_type
    user_top_entrance = train.groupby('uid')['entrance_type'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown').reset_index()
    user_top_entrance.columns = ['uid', 'user_top_entrance']

    # 用户偏好 scene_name
    user_top_scene = train.groupby('uid')['scene_name'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown').reset_index()
    user_top_scene.columns = ['uid', 'user_top_scene']

    return user_play_count, user_repeat_rate, user_top_tab, user_top_entrance, user_top_scene


def build_episode_popularity_features(train, ep_out):
    """E. 播客热度特征"""
    print('[6/6] 构建播客热度特征...')

    # 单集播放次数
    ep_play_count = train.groupby('episode_id').size().reset_index(name='ep_play_count')

    # 单集重复播放率
    ep_repeat_rate = train.groupby('episode_id')['label'].mean().reset_index(name='ep_repeat_rate')

    # 主播播放量 & 重复率
    ep_with_label = train.merge(ep_out[['episode_id', 'primary_host']], on='episode_id', how='left')
    host_stats = ep_with_label.groupby('primary_host').agg(
        host_play_count=('label', 'size'),
        host_repeat_rate=('label', 'mean')
    ).reset_index()
    host_stats.rename(columns={'primary_host': 'host_key'}, inplace=True)

    # 类目播放量 & 重复率
    ep_with_cat = train.merge(ep_out[['episode_id', 'primary_cat']], on='episode_id', how='left')
    cat_stats = ep_with_cat.groupby('primary_cat').agg(
        cat_play_count=('label', 'size'),
        cat_repeat_rate=('label', 'mean')
    ).reset_index()
    cat_stats.rename(columns={'primary_cat': 'cat_key'}, inplace=True)

    return ep_play_count, ep_repeat_rate, host_stats, cat_stats


def merge_all_features(train, test, uf, ep_out, ep_add_out,
                       user_play_count, user_repeat_rate, user_top_tab, user_top_entrance, user_top_scene,
                       ep_play_count, ep_repeat_rate, host_stats, cat_stats):
    """合并所有特征到 train 和 test"""
    print('合并所有特征...')

    user_feats = ['uid', 'age_bin', 'sex_enc', 'rg_source_enc', 'address_real',
                  'profile_complete', 'reg_days', 'exp_days']
    user_hist_feats = ['uid', 'user_play_count', 'user_repeat_rate',
                       'user_top_tab', 'user_top_entrance', 'user_top_scene']

    def _merge_df(df, label_col_exists):
        # 用户画像
        df = df.merge(uf[user_feats], on='uid', how='left')
        # 用户历史
        df = df.merge(user_play_count, on='uid', how='left')
        df = df.merge(user_repeat_rate, on='uid', how='left')
        df = df.merge(user_top_tab, on='uid', how='left')
        df = df.merge(user_top_entrance, on='uid', how='left')
        df = df.merge(user_top_scene, on='uid', how='left')
        # 播客内容
        df = df.merge(ep_out, on='episode_id', how='left')
        df = df.merge(ep_add_out, on='episode_id', how='left')
        # 播客热度
        df = df.merge(ep_play_count, on='episode_id', how='left')
        df = df.merge(ep_repeat_rate, on='episode_id', how='left')
        df = df.merge(host_stats, left_on='primary_host', right_on='host_key', how='left')
        df = df.merge(cat_stats, left_on='primary_cat', right_on='cat_key', how='left')

        # 填补缺失值
        df['user_play_count'] = df['user_play_count'].fillna(0)
        repeat_fill = df['label'].mean() if label_col_exists else 0.5
        df['user_repeat_rate'] = df['user_repeat_rate'].fillna(repeat_fill)
        df['ep_play_count'] = df['ep_play_count'].fillna(0)
        df['ep_repeat_rate'] = df['ep_repeat_rate'].fillna(0.5)
        df['host_play_count'] = df['host_play_count'].fillna(0)
        df['host_repeat_rate'] = df['host_repeat_rate'].fillna(0.5)
        df['cat_play_count'] = df['cat_play_count'].fillna(0)
        df['cat_repeat_rate'] = df['cat_repeat_rate'].fillna(0.5)

        # 数值特征填补
        num_cols = ['duration_sec', 'cat_count', 'host_count', 'producer_count', 'writer_count',
                    'title_len', 'uuid_group_size', 'reg_days', 'exp_days']
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(-1)

        # 类别特征填补
        cat_cols = ['primary_cat', 'primary_host', 'user_top_tab', 'user_top_entrance', 'user_top_scene']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')

        return df

    train = _merge_df(train, label_col_exists=True)
    test = _merge_df(test, label_col_exists=False)

    return train, test


def encode_categorical(train, test):
    """将类别特征编码为整数（LightGBM 需要）"""
    print('编码类别特征...')

    cat_cols = ['tab_name', 'scene_name', 'entrance_type',
                'primary_cat', 'primary_host',
                'user_top_tab', 'user_top_entrance', 'user_top_scene']

    for col in cat_cols:
        # 合并 train+test 的类别
        all_vals = pd.concat([train[col], test[col]]).unique()
        mapping = {v: i for i, v in enumerate(all_vals)}
        train[col + '_enc'] = train[col].map(mapping).fillna(-1).astype(int)
        test[col + '_enc'] = test[col].map(mapping).fillna(-1).astype(int)

    return train, test


def build_cross_features(train, test, feature_cols):
    """F. 交叉特征 - 用户×播客×上下文的多维交互"""
    print('构建交叉特征...')
    n_before = len(feature_cols)
    new_num_feats = []
    new_cat_feats = []

    # ========== 内存优化 ==========
    print('  内存优化: downcasting...')
    import gc
    # 删除不再需要的原始字符串列（已经编码过了）
    drop_cols = [c for c in train.columns if c not in feature_cols and c not in
                 ['uid', 'episode_id', 'id', 'label',
                  'tab_name', 'scene_name', 'entrance_type',
                  'primary_cat', 'primary_host',
                  'user_top_tab', 'user_top_scene', 'user_top_entrance']]
    if drop_cols:
        train.drop(columns=drop_cols, inplace=True)
        test.drop(columns=[c for c in drop_cols if c in test.columns], inplace=True)
        print(f'    删除 {len(drop_cols)} 个多余列')

    # 数值列 downcast 到 float32
    for df in [train, test]:
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
        for col in df.select_dtypes(include=['int64']).columns:
            if col not in ['uid', 'episode_id', 'id', 'label']:
                df[col] = df[col].astype(np.int32)
    gc.collect()
    mem_mb = train.memory_usage(deep=True).sum() / 1024**2
    print(f'    train 内存: {mem_mb:.0f} MB')

    # ========== 1. 用户 × 播客 数值交互 ==========
    for df in [train, test]:
        # repeat_rate 组合
        df['user_x_ep_repeat'] = df['user_repeat_rate'] * df['ep_repeat_rate']
        df['user_ep_repeat_diff'] = df['user_repeat_rate'] - df['ep_repeat_rate']
        df['user_ep_repeat_sum'] = df['user_repeat_rate'] + df['ep_repeat_rate']

        # 热度组合
        df['log_user_play'] = np.log1p(df['user_play_count'].clip(upper=1e6))
        df['log_ep_play'] = np.log1p(df['ep_play_count'].clip(upper=1e6))
        df['log_user_x_log_ep'] = df['log_user_play'] * df['log_ep_play']

        # 相对热度：episode 在用户播放中的占比
        df['ep_popularity_ratio'] = df['ep_play_count'] / (df['user_play_count'] + 1)
        df['ep_popularity_ratio'] = df['ep_popularity_ratio'].clip(upper=100)

        # 主播热度比
        df['host_pop_ratio'] = df['host_play_count'] / (df['ep_play_count'] + 1)
        df['host_pop_ratio'] = df['host_pop_ratio'].clip(upper=100)

        # 用户参与度
        df['user_engagement'] = df['user_play_count'] * df['user_repeat_rate']
        df['log_user_engagement'] = np.log1p(df['user_engagement'].clip(upper=1e8))

        # episode 质量：热度 × 重复率
        df['ep_quality_score'] = df['ep_play_count'] * df['ep_repeat_rate']
        df['log_ep_quality'] = np.log1p(df['ep_quality_score'].clip(upper=1e8))

        # duration × 用户偏好
        df['duration_x_user_repeat'] = df['duration_sec'] * df['user_repeat_rate']
        df['duration_x_ep_repeat'] = df['duration_sec'] * df['ep_repeat_rate']

    new_num_feats += [
        'user_x_ep_repeat', 'user_ep_repeat_diff', 'user_ep_repeat_sum',
        'log_user_play', 'log_ep_play', 'log_user_x_log_ep',
        'ep_popularity_ratio', 'host_pop_ratio',
        'user_engagement', 'log_user_engagement',
        'ep_quality_score', 'log_ep_quality',
        'duration_x_user_repeat', 'duration_x_ep_repeat',
    ]

    # ========== 2. 用户 × 上下文匹配 ==========
    for df in [train, test]:
        df['tab_match'] = (df['tab_name'] == df['user_top_tab']).astype(int)
        df['scene_match'] = (df['scene_name'] == df['user_top_scene']).astype(int)
        df['entrance_match'] = (df['entrance_type'] == df['user_top_entrance']).astype(int)
        df['ctx_match_count'] = df['tab_match'] + df['scene_match'] + df['entrance_match']

        # 是否为非典型上下文（全不匹配）
        df['ctx_all_mismatch'] = (df['ctx_match_count'] == 0).astype(int)

    new_num_feats += [
        'tab_match', 'scene_match', 'entrance_match',
        'ctx_match_count', 'ctx_all_mismatch',
    ]

    # ========== 3. K-Fold 目标编码（防泄漏） ==========
    print('  K-Fold 目标编码: user×category, user×host...')
    from sklearn.model_selection import StratifiedKFold

    for cat_col, prefix in [('primary_cat', 'user_cat'), ('primary_host', 'user_host')]:
        # 只取需要的列，避免 iloc 复制整个宽表
        kf_cols = ['uid', cat_col, 'label']
        kf_df = train[kf_cols].copy()

        train[f'{prefix}_repeat_rate'] = np.nan
        train[f'{prefix}_play_count'] = np.nan

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in skf.split(kf_df, kf_df['label']):
            fold_stats = kf_df.iloc[tr_idx].groupby(['uid', cat_col])['label'].agg(
                repeat_rate='mean', play_count='size'
            ).reset_index()
            fold_stats.columns = ['uid', cat_col, f'{prefix}_repeat_rate', f'{prefix}_play_count']

            val_keys = kf_df.iloc[val_idx][['uid', cat_col]].reset_index(drop=False)
            merged = val_keys.merge(fold_stats, on=['uid', cat_col], how='left')
            train.loc[merged['index'].values, f'{prefix}_repeat_rate'] = \
                merged[f'{prefix}_repeat_rate'].values
            train.loc[merged['index'].values, f'{prefix}_play_count'] = \
                merged[f'{prefix}_play_count'].values

        del kf_df

        # 填补 OOF 中的 NaN（用户或类目在 fold 中未出现）
        global_stats = train.groupby(['uid', cat_col])['label'].agg(
            repeat_rate='mean', play_count='size'
        ).reset_index()
        global_stats.columns = ['uid', cat_col, f'{prefix}_repeat_rate_global', f'{prefix}_play_count_global']

        train = train.merge(global_stats, on=['uid', cat_col], how='left')
        train[f'{prefix}_repeat_rate'] = train[f'{prefix}_repeat_rate'].fillna(
            train[f'{prefix}_repeat_rate_global'])
        train[f'{prefix}_play_count'] = train[f'{prefix}_play_count'].fillna(
            train[f'{prefix}_play_count_global'])
        train.drop(columns=[f'{prefix}_repeat_rate_global', f'{prefix}_play_count_global'], inplace=True)

        # test: 全局统计
        test = test.merge(global_stats, on=['uid', cat_col], how='left')
        test[f'{prefix}_repeat_rate'] = test[f'{prefix}_repeat_rate_global'].fillna(0.5)
        test[f'{prefix}_play_count'] = test[f'{prefix}_play_count_global'].fillna(0)
        test.drop(columns=[f'{prefix}_repeat_rate_global', f'{prefix}_play_count_global'], inplace=True)

        new_num_feats += [f'{prefix}_repeat_rate', f'{prefix}_play_count']

    # ========== 4. 用户分段交叉 ==========
    for df in [train, test]:
        df['is_new_user'] = (df['user_play_count'] <= 3).astype(np.float32)
        df['is_power_user'] = (df['user_play_count'] >= 50).astype(np.float32)
        df['new_user_x_ep_repeat'] = (df['is_new_user'] * df['ep_repeat_rate']).astype(np.float32)
        df['power_user_x_ep_repeat'] = (df['is_power_user'] * df['ep_repeat_rate']).astype(np.float32)
        df['reg_days_x_play'] = (df['reg_days'].clip(lower=0) * df['log_user_play']).astype(np.float32)
        df['old_user_low_activity'] = ((df['reg_days'] > 365) & (df['user_play_count'] <= 5)).astype(np.float32)

    new_num_feats += [
        'is_new_user', 'is_power_user',
        'new_user_x_ep_repeat', 'power_user_x_ep_repeat',
        'reg_days_x_play', 'old_user_low_activity',
    ]

    # ========== 5. Episode × 用户画像交叉 ==========
    for df in [train, test]:
        df['age_x_duration_bin'] = (df['age_bin'] * df['duration_bin']).astype(np.float32)
        df['sex_x_duration'] = (df['sex_enc'] * df['duration_sec']).astype(np.float32)
        df['sex_x_ep_repeat'] = (df['sex_enc'] * df['ep_repeat_rate']).astype(np.float32)

    new_num_feats += ['age_x_duration_bin', 'sex_x_duration', 'sex_x_ep_repeat']

    all_new_feats = new_num_feats
    feature_cols = feature_cols + all_new_feats

    print(f'  新增 {len(all_new_feats)} 个交叉特征')
    print(f'  总特征数: {len(feature_cols)}')

    return train, test, feature_cols


def get_feature_columns():
    """返回基础特征列名（41个）"""
    numeric_feats = [
        'age_bin', 'sex_enc', 'rg_source_enc', 'address_real', 'profile_complete',
        'reg_days', 'exp_days',
        'duration_sec', 'duration_bin', 'cat_count', 'is_top_cat',
        'host_count', 'is_default_host', 'producer_count', 'is_default_producer',
        'writer_count', 'is_default_writer',
        'language_enc', 'lang_is_unknown', 'lang_is_zh',
        'title_len', 'uuid_group_size', 'uuid_has_nan', 'uuid_is_singleton',
        'ctx_complete',
        'user_play_count', 'user_repeat_rate',
        'ep_play_count', 'ep_repeat_rate',
        'host_play_count', 'host_repeat_rate',
        'cat_play_count', 'cat_repeat_rate',
    ]
    encoded_cat_feats = [
        'tab_name_enc', 'scene_name_enc', 'entrance_type_enc',
        'primary_cat_enc', 'primary_host_enc',
        'user_top_tab_enc', 'user_top_entrance_enc', 'user_top_scene_enc',
    ]
    return numeric_feats + encoded_cat_feats


def run_feature_pipeline():
    """运行基础特征工程，保存为 base parquet（第二阶段由 cross_features.py 处理）"""
    train, test, user_feat, ep_feat, ep_add = load_data()

    uf = build_user_features(user_feat)
    ep_out, ep_add_out = build_episode_features(ep_feat, ep_add)
    train, test = build_context_features(train, test)
    user_play_count, user_repeat_rate, user_top_tab, user_top_entrance, user_top_scene = \
        build_user_history_features(train)
    ep_play_count, ep_repeat_rate, host_stats, cat_stats = \
        build_episode_popularity_features(train, ep_out)

    train, test = merge_all_features(
        train, test, uf, ep_out, ep_add_out,
        user_play_count, user_repeat_rate, user_top_tab, user_top_entrance, user_top_scene,
        ep_play_count, ep_repeat_rate, host_stats, cat_stats
    )

    train, test = encode_categorical(train, test)

    # 基础特征列
    base_feature_cols = get_feature_columns()
    print(f'基础特征: {len(base_feature_cols)} 个')

    # 轻量交叉特征（不需要 merge）
    train, test, feature_cols = build_cross_features(train, test, base_feature_cols)

    print(f'\n第一阶段完成！共 {len(feature_cols)} 个特征')
    print(f'训练集: {train.shape}, 测试集: {test.shape}')

    return train, test, feature_cols


if __name__ == '__main__':
    train, test, feature_cols = run_feature_pipeline()

    # 保留原始 key 列 + 特征列 + label（只保留存在的列）
    keep_train = ['uid', 'episode_id', 'label',
                  'tab_name', 'scene_name', 'entrance_type',
                  'primary_cat', 'primary_host',
                  'user_top_tab', 'user_top_scene', 'user_top_entrance'] + feature_cols
    keep_test = ['uid', 'episode_id',
                 'tab_name', 'scene_name', 'entrance_type',
                 'primary_cat', 'primary_host',
                 'user_top_tab', 'user_top_scene', 'user_top_entrance'] + feature_cols

    # 只保留存在的列 & 去重
    keep_train = list(dict.fromkeys(c for c in keep_train if c in train.columns))
    keep_test = list(dict.fromkeys(c for c in keep_test if c in test.columns))

    train[keep_train].to_parquet(os.path.join(OUTPUT_DIR, 'train_base.parquet'), index=False)
    test[keep_test].to_parquet(os.path.join(OUTPUT_DIR, 'test_base.parquet'), index=False)

    with open(os.path.join(OUTPUT_DIR, 'feature_cols_base.txt'), 'w') as f:
        f.write('\n'.join(feature_cols))

    print(f'\nBase parquet 已保存:')
    print(f'  train_base.parquet: {os.path.getsize(os.path.join(OUTPUT_DIR, "train_base.parquet"))/1024/1024:.1f} MB')
    print(f'  test_base.parquet: {os.path.getsize(os.path.join(OUTPUT_DIR, "test_base.parquet"))/1024/1024:.1f} MB')
