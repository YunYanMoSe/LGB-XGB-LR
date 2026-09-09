"""阶段五（程序要求）：CBOW 与 Skip-gram 训练商品序列向量（item2vec）。

语料语义：每笔订单(InvoiceNo) = 一个句子，词 = 订单内去重商品 StockCode
（购物篮视角，篮内顺序无意义，已固定随机化）。
分别训练两个 gensim Word2Vec 模型：
    CBOW       sg=0  用上下文(篮内其它商品)预测当前商品
    Skip-gram  sg=1  用当前商品预测上下文

产出：
    models/item2vec_cbow.model / item2vec_sg.model    gensim 模型
    models/item2vec_vectors_cbow.npz / _sg.npz        词表+稠密向量
    outputs/item2vec/loss_by_epoch.csv                逐 epoch 训练损失
    outputs/item2vec/hyperparams.json                 本组超参数记录
    outputs/item2vec/loss_curve.png                   训练损失曲线（重要图）
    outputs/item2vec/train_summary.txt                打印摘要（供写报告用）

运行：venv\\Scripts\\python.exe scripts\\06_train_item2vec.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.settings import COLUMNS, PATHS, SEED
from src.embedding import HYPER, build_basket_sentences, train_basket_embeddings
from src.user_similarity import load_clean_transactions

_USER, _ITEM = COLUMNS["user"], COLUMNS["item"]

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

OUT = PATHS["outputs_dir"] / "item2vec"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = PATHS["models_dir"]

METHODS = [
    ("cbow", "CBOW（连续词袋，sg=0）", 0),
    ("skip", "Skip-gram（跳字，sg=1）", 1),
]


def main() -> None:
    t0 = time.time()
    df = load_clean_transactions()
    df = df.copy()
    df[_USER] = df[_USER].astype("int64")
    df[_ITEM] = df[_ITEM].astype(str)

    # ---------- 语料 ----------
    sentences = build_basket_sentences(df, seed=SEED, min_len=2)
    n_baskets = len(sentences)
    tokens_raw = sum(len(s) for s in sentences)

    records: dict[str, dict] = {}
    losses_all: list[dict] = []
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("阶段五 · 训练序列向量（Item2Vec，购物篮=句子）")
    lines.append(f"语料：{n_baskets} 个订单(篮)  平均每篮 {tokens_raw / n_baskets:.1f} 种商品")
    lines.append("=" * 78)
    lines.append("超参数（两种方法共用，仅 sg 不同）")
    for k, v in HYPER.items():
        lines.append(f"    {k:<12} = {v}")
    lines.append("")

    for key, label, sg in METHODS:
        tk = time.time()
        model, per_epoch = train_basket_embeddings(
            sentences, sg=sg, hyper={"seed": SEED, "workers": 4}
        )
        vocab = len(model.wv)
        model.save(str(MODELS / f"item2vec_{key}.model"))
        np.savez_compressed(
            str(MODELS / f"item2vec_vectors_{key}.npz"),
            vectors=model.wv.vectors,
            keys=np.asarray(model.wv.index_to_key, dtype=str),
        )

        for i, loss in enumerate(per_epoch, 1):
            losses_all.append({"method": key, "epoch": i, "loss": round(float(loss), 4)})
        records[key] = {
            "label": label, "sg": sg,
            "vocab": vocab, "n_baskets": n_baskets, "tokens": tokens_raw,
            "train_sec": round(time.time() - tk, 1),
            "per_epoch_loss": [round(float(x), 4) for x in per_epoch],
        }

        lines.append(f"--- {label}  耗时 {records[key]['train_sec']}s ---")
        lines.append(f"    词表大小 = {vocab}")
        head = per_epoch[:3]
        tail = per_epoch[-3:]
        lines.append(f"    前 3 轮损失: {[f'{x:.1f}' for x in head]}")
        lines.append(f"    后 3 轮损失: {[f'{x:.1f}' for x in tail]}")
        lines.append(f"    第1轮={per_epoch[0]:.1f}  末轮={per_epoch[-1]:.1f}"
                     f"  相对末轮/首轮={per_epoch[-1]/per_epoch[0]:.3f}")
        lines.append("")

    # 两条损失曲线便于对照：拼接打印
    cbow = records["cbow"]["per_epoch_loss"]
    skip = records["skip"]["per_epoch_loss"]
    lines.append("epoch    CBOW损失       Skip-gram损失")
    for i in range(1, len(cbow) + 1):
        lines.append(f"  {i:<6}  {cbow[i-1]:>10.1f}   {skip[i-1]:>14.1f}")
    lines.append("=" * 78)

    (OUT / "loss_by_epoch.csv").write_text(
        "method,epoch,loss\n"
        + "".join(f"{r['method']},{r['epoch']},{r['loss']}\n" for r in losses_all),
        encoding="utf-8",
    )
    hyper_record = dict(HYPER)
    hyper_record["sg"] = {"cbow": 0, "skip": 1}
    hyper_record["seed"] = int(SEED)
    (OUT / "hyperparams.json").write_text(
        json.dumps({"hyper": hyper_record, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    text = "\n".join(lines)
    (OUT / "train_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n总耗时 {time.time()-t0:.0f}s；模型/向量见 models/，记录见 outputs/item2vec/")

    _plot_loss_curve(OUT / "loss_curve.png", cbow, skip)


def _plot_loss_curve(path: Path, cbow: list[float], skip: list[float]) -> None:
    """损失曲线图：左 CBOW、右 Skip-gram（两面板各用自己刻度，避免被量级误导）。

    采用与网页一致的自校验调色板：CBOW 蓝 #2a78d6，Skip-gram 橙 #eb6834。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fam in ("Microsoft YaHei", "SimHei", "DengXian"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=fam),
                                  fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [fam, "DejaVu Sans"]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    BLUE, ORANGE = "#2a78d6", "#eb6834"
    INK, GRID, PAGE = "#0b0b0b", "#e1e0d9", "#ffffff"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    panels = [
        (axes[0], cbow, "CBOW（sg=0）", BLUE),
        (axes[1], skip, "Skip-gram（sg=1）", ORANGE),
    ]
    for ax, ys, title, color in panels:
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=4,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.3)
        ax.set_title(title, fontsize=13, color=INK, loc="left", pad=10)
        ax.set_xlabel("epoch（训练轮数）", fontsize=10.5, color=INK)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK, labelsize=10)
        ax.set_xticks(xs)
        ax.annotate(f"末轮 {ys[-1]:.0f}", xy=(len(ys), ys[-1]),
                    xytext=(len(ys) * 0.55, ys[-1] * 0.72), fontsize=10, color=color)
    fig.suptitle("CBOW 与 Skip-gram 的训练损失曲线（每轮损失 = 该轮所有样本负对数似然累加，越小越好）",
                 fontsize=13.5, color=INK, x=0.01, ha="left", y=1.02)
    fig.text(0.5, -0.02,
             "两方法损失量级不同（目标不同），故分面板按各自刻度展示；两者均随 epoch 收敛下降。",
             ha="center", fontsize=9.5, color="#52514e")
    fig.set_facecolor(PAGE)
    for ax, _, _, _ in panels:
        ax.set_facecolor(PAGE)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)
    print(f"损失曲线图 -> {path}")


if __name__ == "__main__":
    main()
