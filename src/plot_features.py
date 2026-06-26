"""
特征分布可视化脚本
生成关键特征的分布图，用于分析报告
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互后端
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = r'D:\Pycharm_workplace\即刻笔试'
OUTPUT_DIR = r'D:\Pycharm_workplace\即刻笔试\figs'

def plot_feature_distributions():
    """绘制关键特征的分布"""
    print('[1/3] 加载训练数据...')
    train = pd.read_parquet(os.path.join(DATA_DIR, 'train_featured.parquet'))
    
    # 只取前10万行（加速可视化）
    if len(train) > 100000:
        train_sample = train.sample(n=100000, random_state=42)
    else:
        train_sample = train
    
    print(f'  使用 {len(train_sample)} 行样本进行可视化')
    
    # 关键特征列表
    key_features = [
        ('user_repeat_rate', '用户历史重复播放率'),
        ('ep_repeat_rate', '单集历史重复播放率'),
        ('user_play_count', '用户播放次数'),
        ('ep_play_count', '单集播放次数'),
        ('duration_sec', '播客时长(秒)'),
        ('age_bin', '年龄分段'),
        ('title_len', '标题Token数'),
        ('uuid_group_size', 'UUID组大小'),
    ]
    
    print('[2/3] 绘制特征分布图...')
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('关键特征分布', fontsize=16, fontweight='bold')
    
    for idx, (feat, name) in enumerate(key_features):
        ax = axes[idx // 4, idx % 4]
        
        if feat in train_sample.columns:
            data = train_sample[feat].dropna()
            
            if feat in ['user_repeat_rate', 'ep_repeat_rate']:
                # 重复率：直方图
                ax.hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                ax.set_xlabel('概率')
                ax.set_ylabel('频数')
                ax.axvline(data.mean(), color='red', linestyle='--', 
                           label=f'均值={data.mean():.3f}')
                ax.legend()
            elif feat in ['user_play_count', 'ep_play_count', 'duration_sec']:
                # 长尾分布：log-scale直方图
                ax.hist(np.log1p(data), bins=50, alpha=0.7, color='green', edgecolor='black')
                ax.set_xlabel('Log(1+x)')
                ax.set_ylabel('频数')
                ax.set_title(f'{name}\n(对数变换后)', fontsize=10)
            else:
                # 离散特征：柱状图
                vc = data.value_counts().sort_index()
                ax.bar(vc.index, vc.values, alpha=0.7, color='orange', edgecolor='black')
                ax.set_xlabel('取值')
                ax.set_ylabel('频数')
            
            ax.set_title(name, fontsize=10)
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'fig6_feature_dist.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'  保存: {output_path}')
    plt.close()
    
    print('[3/3] 绘制特征vs标签关系图...')
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('特征 vs 标签 (重复播放率)', fontsize=16, fontweight='bold')
    
    for idx, (feat, name) in enumerate(key_features):
        ax = axes[idx // 4, idx % 4]
        
        if feat in train_sample.columns and 'label' in train_sample.columns:
            data = train_sample[[feat, 'label']].dropna()
            
            if feat in ['user_repeat_rate', 'ep_repeat_rate', 'duration_sec']:
                # 连续特征：按特征分箱，计算每箱的正样本率
                data['bin'] = pd.qcut(data[feat], q=20, duplicates='drop')
                bin_stats = data.groupby('bin')['label'].agg(['mean', 'count']).reset_index()
                bin_stats['bin_mid'] = bin_stats['bin'].apply(lambda x: x.mid if hasattr(x, 'mid') else x.left)
                
                ax.plot(bin_stats['bin_mid'], bin_stats['mean'], 'bo-', linewidth=2, markersize=5)
                ax.set_xlabel(name)
                ax.set_ylabel('重复播放率')
                ax.grid(alpha=0.3)
            else:
                # 离散特征：每个取值的正样本率
                grp = data.groupby(feat)['label'].mean()
                ax.bar(grp.index, grp.values, alpha=0.7, color='purple', edgecolor='black')
                ax.set_xlabel(name)
                ax.set_ylabel('重复播放率')
                ax.grid(alpha=0.3)
            
            ax.set_title(name, fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'fig7_feature_vs_label.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'  保存: {output_path}')
    plt.close()
    
    print('\n完成！生成了2张新图：')
    print('  - fig6_feature_dist.png: 特征分布')
    print('  - fig7_feature_vs_label.png: 特征vs标签关系')

if __name__ == '__main__':
    plot_feature_distributions()
