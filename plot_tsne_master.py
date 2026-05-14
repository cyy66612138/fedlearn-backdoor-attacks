import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import glob
import os

# 学术排版设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 攻击名称规范化映射
TITLE_MAP = {
    'badnets': 'BadNets', 'dba': 'DBA', 'blended': 'Blended',
    'sinusoidal': 'Sinusoidal', 'modelreplacement': 'Model Replacement',
    'layerwisepoisoning': 'LP Attack', 'cerp': 'CERP', 'fcba': 'FCBA',
    'darkfed': 'DarkFed', 'threedfed': '3DFed', 'feddare': 'FedDARE (Ours)'
}


def analyze_and_plot():
    file_paths = glob.glob("tsne_data/tsne_*.pt")
    if not file_paths:
        print("未检测到特征文件，请确保服务器端已成功运行至目标轮次。")
        return

    os.makedirs("tsne_results", exist_ok=True)

    # 用于记录量化数据的列表
    quant_results = []

    for file_path in file_paths:
        base_name = os.path.basename(file_path).replace('tsne_', '').replace('.pt', '').lower()
        formal_name = TITLE_MAP.get(base_name, base_name.upper())

        data = torch.load(file_path)
        X, y = data['X'], data['y']

        if len(np.unique(y)) < 2:
            print(f"[{formal_name}] 警告：数据中仅包含单一种类样本，跳过分析。")
            continue

        # -----------------------------
        # 1. 量化计算：L2 距离与轮廓系数
        # -----------------------------
        benign_center = np.mean(X[y == 0], axis=0)
        malicious_center = np.mean(X[y == 1], axis=0)
        l2_distance = np.linalg.norm(malicious_center - benign_center)
        sil_score = silhouette_score(X, y)

        quant_results.append({
            'Attack': formal_name,
            'L2_Shift': l2_distance,
            'Silhouette': sil_score
        })

        # -----------------------------
        # 2. 降维计算：PCA (去噪) + T-SNE
        # -----------------------------
        X_pca = PCA(n_components=min(50, X.shape[0])).fit_transform(X)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        X_2d = tsne.fit_transform(X_pca)

        # -----------------------------
        # 3. 独立绘图并保存
        # -----------------------------
        plt.figure(figsize=(6, 5))

        # 良性节点
        plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1],
                    c='#4C72B0', label='Benign', alpha=0.5, s=80, edgecolors='none')

        # 恶意节点 (若是本文算法，用绿色高亮)
        is_ours = 'feddare' in base_name
        color = '#C44E52' if not is_ours else '#2ca02c'
        marker = '*' if not is_ours else 'D'
        size = 200 if not is_ours else 250

        plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1],
                    c=color, label='Malicious', alpha=0.9, s=size, marker=marker, edgecolors='w', linewidths=0.5)

        plt.title(f"Feature Manifold: {formal_name}", fontsize=14, fontweight='bold', pad=12)
        plt.xticks([])
        plt.yticks([])

        for spine in plt.gca().spines.values():
            spine.set_color('#DDDDDD')
            spine.set_linewidth(1.5)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
        plt.tight_layout()

        save_name = f"tsne_results/Manifold_{base_name}.pdf"
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[已完成] {formal_name} -> {save_name} (Silhouette: {sil_score:.4f})")

    # -----------------------------
    # 4. 输出量化对比表格
    # -----------------------------
    print("\n" + "=" * 50)
    print("实验 3.1 特征空间量化度量汇总")
    print("=" * 50)
    print(f"{'攻击范式':<25} | {'轮廓系数 (Silhouette) ↓':<25} | {'L2 中心偏移量 ↓':<20}")
    print("-" * 75)
    # 按轮廓系数降序排列（越低越隐蔽）
    quant_results.sort(key=lambda x: x['Silhouette'], reverse=True)
    for res in quant_results:
        print(f"{res['Attack']:<25} | {res['Silhouette']:<25.4f} | {res['L2_Shift']:<20.4f}")
    print("=" * 50)


if __name__ == "__main__":
    analyze_and_plot()