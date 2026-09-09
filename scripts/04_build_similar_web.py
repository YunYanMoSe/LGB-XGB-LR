"""阶段四入口：相似用户 + 相似商品 → 静态网页数据。

预先离线计算好两套 Top-K 相似关系，序列化为纯 JS 数据文件：
    web/data/user_sim.js    window.USER_SIM  全部用户的 Top-K 相似用户
    web/data/item_sim.js    window.ITEM_SIM  全部商品的 Top-K 相似商品
页面本身是手写的静态 HTML（web/similar_users.html、web/similar_items.html），
通过 <script src> 直接加载本地数据文件，因此双击即可离线打开。

运行：venv\\Scripts\\python.exe scripts\\04_build_similar_web.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.settings import COLUMNS
from src.item_similarity import build_item_catalog, top_k_similar_rows
from src.user_similarity import load_clean_transactions, load_purchase_matrix

_USER, _ITEM = COLUMNS["user"], COLUMNS["item"]
_INVOICE = COLUMNS["invoice"]

TOP_K = 10
DATA_DIR = PROJECT_ROOT / "web" / "data"


def _js_bundle(name: str, payload: dict) -> str:
    return f"/* 由 scripts/04_build_similar_web.py 自动生成，请勿手改 */\nwindow.{name} = " + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ) + ";\n"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean_transactions()
    df = df.copy()
    df[_USER] = df[_USER].astype("int64")
    df[_ITEM] = df[_ITEM].astype(str)

    # ---------- 用户侧 ----------
    M, user_ids, item_ids = load_purchase_matrix(df)  # 用户×商品, 购买数量加权
    nnz = int(M.nnz)
    n_users, n_items = M.shape
    user_top = top_k_similar_rows(M, user_ids, top_k=TOP_K, batch=256)
    deg = M.getnnz(axis=1)
    n_orders = df.groupby(_USER)[_INVOICE].nunique()

    def user_stat(uid: int) -> list[int]:
        r = int(np.where(user_ids == uid)[0][0])
        return [int(n_orders.get(int(uid), 0)), int(deg[r])]

    # ---- 用户侧 总结分析数字（供页面"总结与分析"区展示）----
    _row_of = {int(u): i for i, u in enumerate(user_ids)}
    _t1: dict[int, float] = {}
    _ov: list[int] = []
    for r, u in enumerate(user_ids):
        lst = user_top.get(str(u))
        if not lst:
            continue
        _t1[r] = lst[0][1]
        nb_row = _row_of.get(int(lst[0][0]))
        if nb_row is None:
            continue
        a = set(M[r].indices)
        b = set(M[nb_row].indices)
        _ov.append(len(a & b))
    _edges = [("1–2 种", 1, 2), ("3–10 种", 3, 10), ("11–50 种", 11, 50),
              ("51–200 种", 51, 200), (">200 种", 201, 10**9)]
    _buckets = []
    for lab, lo, hi in _edges:
        _idx = [r for r in range(len(deg)) if lo <= deg[r] <= hi and r in _t1]
        _buckets.append([lab, len(_idx), round(float(np.median([_t1[r] for r in _idx])), 3)])
    user_analysis = {
        "medTop1": round(float(np.median(list(_t1.values()))), 3),
        "overlapMed": int(np.median(_ov)) if _ov else 0,
        "overlapMean": round(float(np.mean(_ov)), 1) if _ov else 0.0,
        "degMed": int(np.median(deg)),
        "buckets": _buckets,
    }

    # 抽样示例用户（页面里的"试试"按钮）：给出 id 与一句话标注
    heavy_id = int(user_ids[np.argmax(deg)])
    single_id = int(user_ids[deg == 1][0])
    n_heavy = int(deg[np.argmax(deg)])
    def r17850():
        return int(np.where(user_ids == 17850)[0][0])
    med = deg[r17850()]
    samples = [
        [heavy_id, f"重度用户 · {n_heavy} 种商品"],
        [17850, f"中等用户 · {int(med)} 种商品"],
        [single_id, "只买过 1 种商品"],
    ]
    user_meta = {
        "label": "相似用户推荐",
        "method": "基于用户的协同过滤 · 余弦相似度（购买数量加权）",
        "topK": TOP_K,
        "nUsers": n_users,
        "nItems": n_items,
        "nnz": nnz,
        "density": round(nnz / (n_users * n_items) * 100, 3),
        "nUsersWithNbr": int(sum(1 for v in user_top.values() if v)),
        "samples": samples,
        "analysis": user_analysis,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    # stat 存放每个用户 [订单数, 商品种类]（邻居行展示"该用户买了多少种"需要它）
    user_data = {
        "meta": user_meta,
        "stat": {str(u): user_stat(u) for u in user_ids},
        "rec": {str(u): [[int(b), s] for b, s in v] for u, v in user_top.items()},
    }

    # ---------- 商品侧 ----------
    cat = build_item_catalog(df)
    MI = M.T.tocsr()  # 商品×用户
    item_top = top_k_similar_rows(MI, item_ids, top_k=TOP_K, batch=128)

    # ---- 商品侧 总结分析数字 ----
    _ib = [("1 人", 1, 1), ("2–5 人", 2, 5), ("6–30 人", 6, 30),
           ("31–200 人", 31, 200), (">200 人", 201, 10**9)]
    _it1 = {c: v[0][1] for c, v in item_top.items() if v}
    _ibuckets = []
    for lab, lo, hi in _ib:
        _vals = [s for c, s in _it1.items() if lo <= cat[c]["buyers"] <= hi]
        _ibuckets.append([lab, len(_vals), round(float(np.median(_vals)), 3) if _vals else 0.0])
    _buyers_all = np.array([cat[str(c)]["buyers"] for c in item_ids])
    item_analysis = {
        "medTop1": round(float(np.median(list(_it1.values()))), 3),
        "oneBuyer": int((_buyers_all == 1).sum()),
        "medBuyers": int(np.median(_buyers_all)),
        "buckets": _ibuckets,
    }

    item_samples = [c for c in ["85123A", "85099B", "22138", "22423"] if c in cat]
    item_meta = {
        "label": "相似商品推荐",
        "method": "基于物品的协同过滤 · 余弦相似度（同一批用户的购买数量画像）",
        "topK": TOP_K,
        "nItems": n_items,
        "nUsers": n_users,
        "nnz": nnz,
        "density": round(nnz / (n_users * n_items) * 100, 3),
        "nItemsWithNbr": int(sum(1 for v in item_top.values() if v)),
        "samples": item_samples,
        "analysis": item_analysis,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    item_data = {
        "meta": item_meta,
        "cat": {str(c): v for c, v in cat.items()},
        "rec": {str(c): [[str(b), s] for b, s in v] for c, v in item_top.items()},
    }

    (DATA_DIR / "user_sim.js").write_text(
        _js_bundle("USER_SIM", user_data), encoding="utf-8"
    )
    (DATA_DIR / "item_sim.js").write_text(
        _js_bundle("ITEM_SIM", item_data), encoding="utf-8"
    )

    print("已写出:")
    for f in ("user_sim.js", "item_sim.js"):
        p = DATA_DIR / f
        print(f"  {p.relative_to(PROJECT_ROOT)}  {p.stat().st_size/1024:.0f} KB")
    print(f"用户数={n_users} 商品数={n_items} 非零={nnz} 密度={user_meta['density']}%")
    print(f"有相似用户的用户数={user_meta['nUsersWithNbr']} / {n_users}")
    print(f"有相似商品商品数={item_meta['nItemsWithNbr']} / {n_items}")


if __name__ == "__main__":
    main()
