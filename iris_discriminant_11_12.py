#!/usr/bin/env python3
"""
教材第 11.12 题：Fisher-Anderson Iris 数据的判别分析。

直接运行：
    python exercises/iris_discriminant_11_12.py

不读取外部 CSV；数据来自 sklearn.datasets.load_iris。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hajime_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/hajime_xdg_cache")

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Ellipse
    from scipy import stats
    from sklearn.datasets import load_iris
    from sklearn.metrics import confusion_matrix
    from statsmodels.multivariate.manova import MANOVA
except ImportError as exc:
    print("缺少必要依赖，无法运行本脚本。")
    print("需要安装：pandas numpy scipy statsmodels scikit-learn matplotlib")
    print(f"原始错误：{exc}")
    sys.exit(1)


FEATURES = ["x1", "x2", "x3", "x4"]
PAIR_FEATURES = ["x2", "x4"]
GROUP_LABELS = ["pi1", "pi2", "pi3"]
GROUP_NAMES = {
    "pi1": "setosa",
    "pi2": "versicolor",
    "pi3": "virginica",
}
PRIORS = {label: 1 / 3 for label in GROUP_LABELS}
COLORS = {
    "pi1": "#1f77b4",
    "pi2": "#ff7f0e",
    "pi3": "#2ca02c",
}


def r4(x):
    """Round numeric objects to 4 decimals for display."""
    return np.round(x, 4)


def print_matrix(name, value, row_labels=None, col_labels=None):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        s = pd.Series(r4(arr), index=row_labels)
        print(f"{name}:")
        print(s.to_string())
    else:
        df = pd.DataFrame(r4(arr), index=row_labels, columns=col_labels)
        print(f"{name}:")
        print(df.to_string())


def load_iris_dataframe():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=FEATURES)
    target_to_group = {0: "pi1", 1: "pi2", 2: "pi3"}
    df["group"] = [target_to_group[int(t)] for t in iris.target]
    df["species"] = df["group"].map(GROUP_NAMES)
    return df


def pooled_covariance(df, feature_cols, group_col="group", labels=None):
    labels = list(labels or sorted(df[group_col].unique()))
    p = len(feature_cols)
    pooled_num = np.zeros((p, p), dtype=float)
    covs = {}
    means = {}
    counts = {}

    for label in labels:
        xg = df.loc[df[group_col] == label, feature_cols].to_numpy(dtype=float)
        counts[label] = xg.shape[0]
        means[label] = xg.mean(axis=0)
        covs[label] = np.cov(xg, rowvar=False, bias=False)
        pooled_num += (counts[label] - 1) * covs[label]

    n_total = sum(counts.values())
    pooled = pooled_num / (n_total - len(labels))
    return pooled, covs, means, counts


def box_m_test(df, feature_cols, group_col="group", labels=None):
    labels = list(labels or sorted(df[group_col].unique()))
    pooled, covs, _, counts = pooled_covariance(df, feature_cols, group_col, labels)
    p = len(feature_cols)
    k = len(labels)
    n_total = sum(counts.values())

    sign_p, logdet_p = np.linalg.slogdet(pooled)
    if sign_p <= 0:
        raise ValueError("合并协方差矩阵非正定，无法计算 Box's M。")

    weighted_logdet = 0.0
    correction_sum = 0.0
    for label in labels:
        sign_i, logdet_i = np.linalg.slogdet(covs[label])
        if sign_i <= 0:
            raise ValueError(f"{label} 的协方差矩阵非正定，无法计算 Box's M。")
        weighted_logdet += (counts[label] - 1) * logdet_i
        correction_sum += 1 / (counts[label] - 1)

    m_stat = (n_total - k) * logdet_p - weighted_logdet
    correction = (
        (2 * p**2 + 3 * p - 1)
        / (6 * (p + 1) * (k - 1))
        * (correction_sum - 1 / (n_total - k))
    )
    chi_square = (1 - correction) * m_stat
    df_chi = p * (p + 1) * (k - 1) / 2
    p_value = stats.chi2.sf(chi_square, df_chi)

    return {
        "M": m_stat,
        "chi_square": chi_square,
        "df": df_chi,
        "p_value": p_value,
        "reject": p_value < 0.05,
        "pooled_cov": pooled,
        "group_covs": covs,
        "counts": counts,
    }


def plot_cov_ellipse(ax, mean, cov, color, n_std=2.0, label=None):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=2,
        label=label,
    )
    ax.add_patch(ellipse)
    return ellipse


def qda_scores(x, means, covs, priors, labels=None):
    x = np.asarray(x, dtype=float)
    labels = list(labels or means.keys())
    scores = {}
    for label in labels:
        mu = np.asarray(means[label], dtype=float)
        cov = np.asarray(covs[label], dtype=float)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError(f"{label} 的协方差矩阵非正定。")
        diff = x - mu
        score = (
            -0.5 * logdet
            - 0.5 * diff.T @ np.linalg.inv(cov) @ diff
            + np.log(priors[label])
        )
        scores[label] = float(score)
    return scores


def lda_scores(x, means, pooled_cov, priors, labels=None):
    x = np.asarray(x, dtype=float)
    labels = list(labels or means.keys())
    inv_cov = np.linalg.inv(pooled_cov)
    scores = {}
    for label in labels:
        mu = np.asarray(means[label], dtype=float)
        score = x.T @ inv_cov @ mu - 0.5 * mu.T @ inv_cov @ mu + np.log(priors[label])
        scores[label] = float(score)
    return scores


def predict_from_scores(scores):
    return max(scores, key=scores.get)


def plot_lda_regions_2d(
    df,
    feature_cols,
    means,
    pooled_cov,
    priors,
    labels,
    output_path,
):
    x = df[feature_cols[0]].to_numpy(dtype=float)
    y = df[feature_cols[1]].to_numpy(dtype=float)
    x_min, x_max = x.min() - 0.5, x.max() + 0.5
    y_min, y_max = y.min() - 0.3, y.max() + 0.3

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    z = []
    for point in grid:
        scores = lda_scores(point, means, pooled_cov, priors, labels)
        z.append(labels.index(predict_from_scores(scores)))
    z = np.asarray(z).reshape(xx.shape)

    cmap_light = ListedColormap(["#d8e8f7", "#ffe0bf", "#d9f0d3"])
    cmap_points = ListedColormap([COLORS[label] for label in labels])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.contourf(xx, yy, z, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap_light, alpha=0.75)
    ax.contour(xx, yy, z, levels=[0.5, 1.5], colors="black", linewidths=1.4)

    for i, label in enumerate(labels):
        part = df[df["group"] == label]
        ax.scatter(
            part[feature_cols[0]],
            part[feature_cols[1]],
            s=36,
            color=cmap_points(i),
            edgecolor="white",
            linewidth=0.5,
            label=f"{label} ({GROUP_NAMES[label]})",
        )
        mu = means[label]
        ax.text(mu[0], mu[1], f"R{i + 1}", fontsize=12, weight="bold", ha="center")

    ax.set_xlabel(feature_cols[0])
    ax.set_ylabel(feature_cols[1])
    ax.set_title("LDA classification regions on x2 and x4")
    ax.legend(frameon=True, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def compute_aper(df, feature_cols, labels):
    pooled, _, means, _ = pooled_covariance(df, feature_cols, labels=labels)
    true_labels = df["group"].to_numpy()
    pred_labels = []

    for _, row in df.iterrows():
        x = row[feature_cols].to_numpy(dtype=float)
        scores = lda_scores(x, means, pooled, PRIORS, labels)
        pred_labels.append(predict_from_scores(scores))

    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    correct = int(np.trace(cm))
    errors = int(cm.sum() - correct)
    aper = errors / cm.sum()
    return {
        "confusion_matrix": cm,
        "correct": correct,
        "errors": errors,
        "aper": aper,
        "predicted": np.asarray(pred_labels),
    }


def loocv_lda_error(df, feature_cols, labels):
    true_labels = df["group"].to_numpy()
    pred_labels = []

    for i in range(len(df)):
        train_df = df.drop(index=df.index[i])
        x = df.iloc[i][feature_cols].to_numpy(dtype=float)
        pooled, _, means, _ = pooled_covariance(train_df, feature_cols, labels=labels)
        scores = lda_scores(x, means, pooled, PRIORS, labels)
        pred_labels.append(predict_from_scores(scores))

    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    errors = int(cm.sum() - np.trace(cm))
    eaer = errors / cm.sum()
    return {
        "confusion_matrix": cm,
        "errors": errors,
        "eaer": eaer,
        "predicted": np.asarray(pred_labels),
    }


def mardia_normality_2d(x):
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    mean = x.mean(axis=0)
    cov_mle = np.cov(x, rowvar=False, bias=True)
    inv_cov = np.linalg.inv(cov_mle)
    centered = x - mean
    d = centered @ inv_cov @ centered.T
    b1p = np.mean(d**3)
    md2 = np.diag(d)
    b2p = np.mean(md2**2)

    skew_chi = n * b1p / 6
    skew_df = p * (p + 1) * (p + 2) / 6
    skew_p = stats.chi2.sf(skew_chi, skew_df)

    kurt_mean = p * (p + 2)
    kurt_var = 8 * p * (p + 2) / n
    kurt_z = (b2p - kurt_mean) / np.sqrt(kurt_var)
    kurt_p = 2 * stats.norm.sf(abs(kurt_z))

    return {
        "b1p": b1p,
        "skew_chi": skew_chi,
        "skew_df": skew_df,
        "skew_p": skew_p,
        "b2p": b2p,
        "kurt_z": kurt_z,
        "kurt_p": kurt_p,
        "normal_like": skew_p >= 0.05 and kurt_p >= 0.05,
    }


def print_scores(title, scores):
    print(title)
    for label, score in scores.items():
        print(f"  {label} ({GROUP_NAMES[label]}): {score:.4f}")


def print_confusion(title, cm, labels):
    table = pd.DataFrame(cm, index=[f"actual_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    print(title)
    print(table.to_string())


def run_a(df, summaries):
    print("========== (a) ==========")
    output_path = SCRIPT_DIR / "a_scatter_x2_x4.png"
    pooled2, covs2, means2, _ = pooled_covariance(df, PAIR_FEATURES, labels=GROUP_LABELS)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for label in GROUP_LABELS:
        part = df[df["group"] == label]
        ax.scatter(
            part["x2"],
            part["x4"],
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.5,
            s=40,
            label=f"{label} ({GROUP_NAMES[label]})",
        )
        plot_cov_ellipse(ax, means2[label], covs2[label], COLORS[label], n_std=2.0)

    ax.set_xlabel("x2 = sepal width")
    ax.set_ylabel("x4 = petal width")
    ax.set_title("Scatter plot of x2 and x4 by group")
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"散点图已保存：{output_path}")

    x4_ranges = {}
    for label in GROUP_LABELS:
        part = df[df["group"] == label]
        x4_ranges[label] = (part["x4"].min(), part["x4"].max())
    print("x4 范围：")
    for label, value in x4_ranges.items():
        print(f"  {label}: [{value[0]:.4f}, {value[1]:.4f}]")

    separation_text = (
        "三组在 (x2, x4) 平面上总体明显可分；pi1 与另外两组分离非常明显，"
        "pi2 与 pi3 在边界附近有少量重叠。"
    )
    print(f"三组是否明显可分：{separation_text}")

    normality_rows = []
    for label in GROUP_LABELS:
        xg = df.loc[df["group"] == label, PAIR_FEATURES].to_numpy(dtype=float)
        result = mardia_normality_2d(xg)
        normality_rows.append(
            {
                "group": label,
                "skew_p": result["skew_p"],
                "kurt_p": result["kurt_p"],
                "normal_like": result["normal_like"],
            }
        )
    normality_df = pd.DataFrame(normality_rows).set_index("group")
    print("Mardia 二元正态性近似检查：")
    print(normality_df.assign(skew_p=r4(normality_df["skew_p"]), kurt_p=r4(normality_df["kurt_p"])).to_string())

    normality_text = (
        "各组 Mardia 偏度和峰度检验在 0.05 水平下均未拒绝二元正态性，"
        "可认为 (x2, x4) 近似满足组内二元正态分布。"
        if normality_df["normal_like"].all()
        else "至少一组的 Mardia 检查在 0.05 水平下不支持严格二元正态；散点椭圆形态仍可作为近似判别分析依据。"
    )
    print(f"是否近似满足二元正态分布：{normality_text}")

    summaries["a"] = f"{separation_text} {normality_text}"
    return {"pooled2": pooled2, "covs2": covs2, "means2": means2}


def run_b(df, summaries):
    print("\n========== (b) ==========")
    manova = MANOVA.from_formula("x1 + x2 + x3 + x4 ~ group", data=df)
    manova_result = manova.mv_test()
    stat_table = manova_result.results["group"]["stat"]
    print("MANOVA: H0: mu1 = mu2 = mu3")
    print(stat_table.to_string())

    stat_names = [
        "Wilks' lambda",
        "Pillai's trace",
        "Hotelling-Lawley trace",
        "Roy's greatest root",
    ]
    print("四个多元检验统计量：")
    manova_p_values = []
    for name in stat_names:
        row = stat_table.loc[name]
        p_value = float(row["Pr > F"])
        manova_p_values.append(p_value)
        print(
            f"  {name}: Value={float(row['Value']):.4f}, "
            f"F={float(row['F Value']):.4f}, p-value={p_value:.6g}"
        )

    manova_reject = min(manova_p_values) < 0.05
    print(
        "MANOVA 结论："
        + ("在 alpha=0.05 下拒绝 H0，至少有一个总体均值向量不同。" if manova_reject else "在 alpha=0.05 下不拒绝 H0。")
    )

    box = box_m_test(df, FEATURES, labels=GROUP_LABELS)
    print("\nBox's M Test: H0: Sigma1 = Sigma2 = Sigma3")
    print(f"Box's M = {box['M']:.4f}")
    print(f"Chi-square = {box['chi_square']:.4f}")
    print(f"df = {box['df']:.0f}")
    print(f"p-value = {box['p_value']:.6g}")
    print(
        "协方差矩阵齐性结论："
        + ("拒绝协方差矩阵相等假设。" if box["reject"] else "不拒绝协方差矩阵相等假设。")
    )

    summaries["b"] = (
        "MANOVA 在 0.05 水平下"
        + ("拒绝均值向量相等；" if manova_reject else "不拒绝均值向量相等；")
        + "Box's M "
        + ("拒绝协方差矩阵相等。" if box["reject"] else "不拒绝协方差矩阵相等。")
    )
    return {"manova": manova_result, "box": box}


def run_c(df, summaries):
    print("\n========== (c) ==========")
    _, covs, means, _ = pooled_covariance(df, FEATURES, labels=GROUP_LABELS)
    x0 = np.array([3, 5.1, 7.5, 2], dtype=float)
    print_matrix("x0", x0, row_labels=FEATURES)
    scores = qda_scores(x0, means, covs, PRIORS, GROUP_LABELS)
    print_scores("QDA discriminant scores:", scores)
    pred = predict_from_scores(scores)
    print(f"最终分类结果：{pred} ({GROUP_NAMES[pred]})")

    summaries["c"] = f"x0 的 QDA 分类为 {pred} ({GROUP_NAMES[pred]})。"
    return {"x0": x0, "scores": scores, "pred": pred, "means": means, "covs": covs}


def run_d(df, box_reject, qda_pred, summaries):
    print("\n========== (d) ==========")
    pooled, _, means, _ = pooled_covariance(df, FEATURES, labels=GROUP_LABELS)
    x0 = np.array([3, 5.1, 7.5, 2], dtype=float)
    print_matrix("合并协方差矩阵 Sp", pooled, row_labels=FEATURES, col_labels=FEATURES)
    scores = lda_scores(x0, means, pooled, PRIORS, GROUP_LABELS)
    print_scores("LDA discriminant scores:", scores)
    pred = predict_from_scores(scores)
    print(f"分类结果：{pred} ({GROUP_NAMES[pred]})")
    print(f"与 (c) QDA 是否一致：{'一致' if pred == qda_pred else '不一致'}")
    print("如果 Box's M 检验拒绝协方差齐性：更倾向使用 QDA。")
    print("如果 Box's M 检验不拒绝协方差齐性：更倾向使用 LDA。")
    print(f"本数据 Box's M 实际结论下，更倾向：{'QDA' if box_reject else 'LDA'}。")

    summaries["d"] = (
        f"x0 的 LDA 分类为 {pred} ({GROUP_NAMES[pred]})，"
        f"与 QDA {'一致' if pred == qda_pred else '不一致'}；"
        f"按 Box's M 结果本题更倾向 {'QDA' if box_reject else 'LDA'}。"
    )
    return {"pooled": pooled, "means": means, "scores": scores, "pred": pred}


def run_e(df, summaries):
    print("\n========== (e) ==========")
    pooled2, _, means2, _ = pooled_covariance(df, PAIR_FEATURES, labels=GROUP_LABELS)
    print("三组均值（仅 x2, x4）：")
    means_df = pd.DataFrame({label: r4(means2[label]) for label in GROUP_LABELS}, index=PAIR_FEATURES).T
    print(means_df.to_string())
    print_matrix("二维合并协方差矩阵", pooled2, row_labels=PAIR_FEATURES, col_labels=PAIR_FEATURES)

    x0_pair = np.array([5.1, 2], dtype=float)
    print_matrix("x0_pair", x0_pair, row_labels=PAIR_FEATURES)
    scores = lda_scores(x0_pair, means2, pooled2, PRIORS, GROUP_LABELS)
    print_scores("二维 LDA scores:", scores)
    pred = predict_from_scores(scores)
    print(f"分类结果：{pred} ({GROUP_NAMES[pred]})")

    output_path = SCRIPT_DIR / "e_classification_regions_x2_x4.png"
    plot_lda_regions_2d(
        df,
        PAIR_FEATURES,
        means2,
        pooled2,
        PRIORS,
        GROUP_LABELS,
        output_path,
    )
    print(f"分类区域图已保存：{output_path}")

    summaries["e"] = f"仅用 (x2, x4) 的二维 LDA 将 x0_pair 分类为 {pred} ({GROUP_NAMES[pred]})。"
    return {"pooled2": pooled2, "means2": means2, "scores": scores, "pred": pred}


def run_f(df, summaries):
    print("\n========== (f) ==========")
    aper_result = compute_aper(df, FEATURES, GROUP_LABELS)
    print_confusion("APER 混淆矩阵：", aper_result["confusion_matrix"], GROUP_LABELS)
    print(f"正确分类数 = {aper_result['correct']}")
    print(f"错误分类数 = {aper_result['errors']}")
    print(f"APER = {aper_result['aper']:.4f}")

    loocv_result = loocv_lda_error(df, FEATURES, GROUP_LABELS)
    print_confusion("\nLOOCV 混淆矩阵：", loocv_result["confusion_matrix"], GROUP_LABELS)
    print(f"LOOCV 错误分类数 = {loocv_result['errors']}")
    print(f"E(AER) = {loocv_result['eaer']:.4f}")

    summaries["f"] = (
        f"四变量 LDA 的 APER={aper_result['aper']:.4f}，"
        f"LOOCV 估计 E(AER)={loocv_result['eaer']:.4f}。"
    )
    return {"aper": aper_result, "loocv": loocv_result}


def main():
    df = load_iris_dataframe()
    summaries = {}

    counts = df["group"].value_counts().reindex(GROUP_LABELS)
    print("Fisher-Anderson Iris 数据集")
    print("类别对应：setosa -> pi1, versicolor -> pi2, virginica -> pi3")
    print("变量对应：x1=sepal length, x2=sepal width, x3=petal length, x4=petal width")
    print(f"n1 = {counts['pi1']}, n2 = {counts['pi2']}, n3 = {counts['pi3']}, N = {len(df)}")

    run_a(df, summaries)
    b_result = run_b(df, summaries)
    c_result = run_c(df, summaries)
    run_d(df, b_result["box"]["reject"], c_result["pred"], summaries)
    run_e(df, summaries)
    run_f(df, summaries)

    print("\n========== SUMMARY ==========")
    for key in ["a", "b", "c", "d", "e", "f"]:
        print(f"({key}) {summaries[key]}")


if __name__ == "__main__":
    main()
