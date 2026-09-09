"""阶段六 · Flask 演示/联调服务（指导书验收物）。

一条完整链路：召回(热门+相似扩展+用户向量 三路融合 50) → 精排(LR / LightGBM)
→ 输出 Top-N；并提供 /similar（商品相似）与 /baseline（基线对照）。

运行：venv\\Scripts\\python.exe web\\app.py            # 默认 http://127.0.0.1:5001
     （端口可用环境变量 PORT 覆盖；scripts/13 截图用 PORT=5051 起独立进程）

在线口径说明：网页/演示 API 用**全量**清洗交易与**全量**商品向量索引建上下文
（商品热度、用户画像、语义向量都是“已知历史/静态知识”，无任何用户评估标签，
不涉及离线评测的防泄漏问题）。离线评测（scripts/11、12）另用训练期-only 向量，
本服务不用于产出论文级指标。

端点：
    GET /health                         健康检查
    GET /                                演示主页（列出演示用户/商品与入口）
    GET /similar?item_id=85123A&topn=5&method=item2vec|itemcf
    GET /recommend?user_id=12597&topn=10&ranker=lr|lgb
    GET /baseline?user_id=12597&topn=10&format=json|html(默认)
   任何端点加 &format=json 返回 JSON（便于 curl/脚本联调）。

演示 ID：models/demo_ids.json（由 scripts/12 从数据选出）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request
from sklearn.pipeline import Pipeline

from config.settings import PATHS, SEED
from src import recommender as rec
from src.item_similarity import build_item_catalog

app = Flask(__name__)

# ---------------------------------------------------------------- 启动加载
DEMO = json.loads((PATHS["models_dir"] / "demo_ids.json").read_text(encoding="utf-8")) \
    if (PATHS["models_dir"] / "demo_ids.json").exists() else {"users": [], "products": []}
META = json.loads((PATHS["models_dir"] / "rank_meta.json").read_text(encoding="utf-8")) \
    if (PATHS["models_dir"] / "rank_meta.json").exists() else {}

# 在线全量上下文（无切分，用全部历史）
DF = rec.load_clean()
CTX = None
KEYS = None
UNIT = None          # (vocab,) 行归一化向量索引
ZH = rec.load_products_zh()
CATALOG = build_item_catalog(DF)

LR_PIPE: Pipeline | None = None
LGB_MODEL = None

ROUTE_ZH = {"popular": "热门", "similar": "相似扩展", "user_vec": "用户向量"}
_ITEM = "StockCode"
_READY = False


def _load():
    global CTX, KEYS, UNIT, LR_PIPE, LGB_MODEL, _READY
    vi = PATHS["vector_index_dir"]
    KEYS = [str(k) for k in np.load(vi / "keys.npy", allow_pickle=False)]
    UNIT = np.load(vi / "vectors_unit.npy", allow_pickle=False).astype(np.float32)
    CTX = rec.build_ctx(DF, KEYS, UNIT)  # 全量 ctx（含单笔订单用户，recent=最后一单）
    lr = PATHS["models_dir"] / "lr_model.pkl"
    if lr.exists():
        LR_PIPE = joblib.load(str(lr))
    lgb = PATHS["models_dir"] / "lgb_model.pkl"
    if lgb.exists():
        LGB_MODEL = joblib.load(str(lgb))
    _READY = True


def zh_of(code: str) -> dict:
    """合并 catalog(英文/价格/买家) 与 products(中文) 的展示字段。"""
    c = CATALOG.get(code, {})
    z = ZH.get(code, {})
    return {
        "code": code,
        "desc": c.get("desc", ""),
        "desc_zh": z.get("desc_zh", "") or c.get("desc", ""),
        "category": z.get("category", ""),
        "price": c.get("price"),
        "buyers": c.get("buyers", 0),
    }


def fmt(cands, proba: bool = True) -> list[dict]:
    out = []
    for rank, item in enumerate(cands, start=1):
        code, sc = item if isinstance(item, tuple) else (item, None)
        row = zh_of(code)
        row["rank"] = rank
        if proba and sc is not None:
            row["score"] = round(float(sc), 4)
        out.append(row)
    return out


# ---------------------------------------------------------------- 通用渲染
def page(title: str, subtitle: str, kpis: list[tuple[str, str]],
         tables: list[dict], foot: str = "") -> str:
    """轻量 HTML（无外部依赖，打印/截图都清晰）。"""
    kpi_html = "".join(
        f'<div class="kpi"><div class="kpi-v">{v}</div><div class="kpi-k">{k}</div></div>'
        for k, v in kpis)
    tbl_html = ""
    for t in tables:
        cols = t["cols"]
        head = "".join(f"<th>{c}</th>" for c in cols)
        rows = ""
        for row in t["rows"]:
            cells = "".join(f"<td>{c}</td>" for c in row)
            rows += f"<tr>{cells}</tr>"
        tbl_html += f'<h3>{t.get("caption", "")}</h3><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{title}</title><style>
body{{font-family:'Segoe UI','Microsoft YaHei',sans-serif;margin:0;background:#f4f5f7;color:#222}}
.wrap{{max-width:1000px;margin:0 auto;padding:24px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{color:#666;font-weight:400;font-size:13px;margin:0 0 18px}}
.kpis{{display:flex;gap:12px;margin:0 0 20px}} .kpi{{flex:1;background:#fff;border-radius:10px;padding:12px 14px;
box-shadow:0 1px 3px rgba(0,0,0,.08)}} .kpi-v{{font-size:20px;font-weight:600;color:#1a4f8b}}
.kpi-k{{font-size:12px;color:#777;margin-top:2px}}
table{{border-collapse:collapse;width:100%;background:#fff;border-radius:8px;overflow:hidden;
box-shadow:0 1px 3px rgba(0,0,0,.08);font-size:13px}} h3{{font-size:14px;margin:20px 0 8px}}
th{{background:#1a4f8b;color:#fff;text-align:left;padding:8px 10px;font-weight:500}}
td{{padding:7px 10px;border-top:1px solid #eef0f3;vertical-align:top}}
tr:hover td{{background:#f6f9ff}} .muted{{color:#999;font-size:12px}}
.tag{{display:inline-block;background:#eef4fb;color:#1a4f8b;border-radius:20px;padding:1px 8px;font-size:11px}}
.hit{{color:#c00;font-weight:600}} code{{background:#eef0f3;padding:1px 5px;border-radius:4px;font-size:12px}}
.foot{{margin:18px 0;font-size:12px;color:#888;line-height:1.8}}
.links a{{margin-right:14px}} a{{color:#1a4f8b}}
</style></head><body><div class="wrap">
<h1>{title}</h1><h2>{subtitle}</h2><div class="kpis">{kpi_html}</div>{tbl_html}
<div class="foot">{foot}</div></div></body></html>"""


def bad(reason: str):
    return jsonify({"error": reason}), 404


# ---------------------------------------------------------------- 路由
@app.get("/health")
def health():
    return jsonify({"status": "ok", "ready": _READY})


@app.get("/")
def home():
    demo_users = DEMO.get("users", [])
    demo_products = DEMO.get("products", [])
    user_rows = []
    for u in demo_users:
        info = user_summary(u)
        user_rows.append([str(u), f"{info.get('n_invoices', '?')} 张订单 / "
                                   f"{info.get('n_items', '?')} 种商品",
                          info.get("last_basket_zh", "")])
    prod_rows = []
    for code in demo_products:
        z = zh_of(code)
        prod_rows.append([code, z["desc"], z["desc_zh"], z["category"]])
    tables = [
        {"caption": "演示用户（有真实购买历史，可直接点“精排推荐”）", "cols": ["用户 ID", "历史", "最近一单摘要"],
         "rows": user_rows},
        {"caption": "演示商品（用于 /similar 相似推荐）", "cols": ["商品码", "英文描述", "中文描述", "类目"],
         "rows": prod_rows},
        {"caption": "接口入口", "cols": ["端点", "示例"],
         "rows": [
             ["<code>/similar</code>", "<a href='/similar?item_id=85123A&amp;topn=5'>/similar?item_id=85123A&amp;topn=5</a>"],
             ["<code>/recommend</code>", "<a href='/recommend?user_id=12597&amp;ranker=lr'>/recommend?user_id=12597&amp;ranker=lr</a>"],
             ["<code>/baseline</code>", "<a href='/baseline?user_id=12597'>/baseline?user_id=12597</a>"],
         ]},
    ]
    html = page("推荐系统 · 召回→精排 在线演示", "Online Retail | 三路召回融合 → LR/LightGBM 精排",
                [("链路", "召回50 → 精排Top-N"), ("用户数", f"{DF[_rec_user()].nunique():,}"),
                 ("商品数", f"{len(CATALOG):,}"), ("词表", str(len(KEYS) if KEYS else 0))],
                tables)
    return html.replace("&amp;", "&")


def _rec_user():
    from config.settings import COLUMNS
    return COLUMNS["user"]


def user_summary(u) -> dict:
    """该用户全量历史的概览（主页展示用）。"""
    du = DF[DF[_rec_user()] == int(u)]
    if du.empty:
        return {}
    last_inv = du["InvoiceDate"].max()
    last = du[du["InvoiceDate"] == last_inv]
    codes = [str(c) for c in last["StockCode"].unique()]
    return {
        "n_invoices": int(du["InvoiceNo"].nunique()),
        "n_items": int(du["StockCode"].nunique()),
        "last_basket_zh": "、".join(zh_of(c)["desc_zh"][:14] for c in codes[:5]),
    }


@app.get("/similar")
def similar():
    want = request.args.get("format", "html")
    item_id = str(request.args.get("item_id", "")).strip().upper()
    topn = int(request.args.get("topn", 5))
    method = request.args.get("method", "item2vec").strip().lower()
    if not item_id:
        return bad("缺少 item_id 参数，例如 /similar?item_id=85123A&topn=5")
    if method == "item2vec":
        if item_id not in CTX.embed_row:
            return bad(f"商品 {item_id} 不在嵌入词表（出现次数 < 5 的冷门品没有向量）")
        ri = CTX.embed_row[item_id]
        sims = UNIT @ UNIT[ri]
        order = np.argsort(-sims)
        cands = []
        for j in order:
            if str(KEYS[j]) == item_id:
                continue
            cands.append((str(KEYS[j]), float(sims[j])))
            if len(cands) >= topn:
                break
        rows = fmt(cands)
        caption = f"商品 {item_id} 的相似商品（Item2Vec Skip-gram 全量向量 · Top-{topn}）"
    else:  # itemcf：0/1 商品画像余弦（与离线特征同口径）
        col = CTX.col_of.get(item_id)
        if col is None:
            return bad(f"商品 {item_id} 不在训练/全量商品中")
        sims = CTX.item_unit @ CTX.item_unit[col]
        order = np.argsort(-sims)
        cands = []
        for j in order:
            if j == col:
                continue
            cands.append((str(CTX.code_of[j]), float(sims[j])))
            if len(cands) >= topn:
                break
        rows = fmt(cands)
        caption = f"商品 {item_id} 的相似商品（ItemCF 买家画像余弦 · Top-{topn}）"
    head = zh_of(item_id)
    kpis = [("查询商品", item_id), ("中文", head["desc_zh"]),
            ("类目", head["category"]), ("买家数", str(head["buyers"] or 0))]
    if want == "json":
        return jsonify({"item_id": item_id, "method": method,
                        "query": head, "similar": rows})
    tbl = [{"caption": caption,
            "cols": ["#", "商品码", "商品(中/英)", "类目", "单价£", "相似度"],
            "rows": [[r["rank"], r["code"], f"{r['desc_zh']}<br><span class='muted'>{r['desc']}</span>",
                      r["category"], r["price"], round(r["score"], 4)] for r in rows]}]
    return page("相似商品", f"/similar?item_id={item_id}&topn={topn}&method={method}",
                kpis, tbl,
                "说明：Item2Vec 邻居来自“同一订单成套购买”的全量语义向量；"
                "ItemCF 邻居来自同一批买家的 0/1 购买画像（与离线指标同口径）。")


@app.get("/recommend")
def recommend():
    want = request.args.get("format", "html")
    uid = int(request.args.get("user_id", 0) or 0)
    topn = int(request.args.get("topn", 10))
    ranker = request.args.get("ranker", "lr").strip().lower()
    if uid not in CTX.user_row:
        return bad(f"用户 {uid} 不存在（全量共 {len(CTX.users):,} 位用户）")
    model = {"lr": LR_PIPE, "lgb": LGB_MODEL}.get(ranker)
    if model is None:
        return bad("ranker 只能是 lr 或 lgb")
    cands = rec.recall_fusion(CTX, uid, final_k=rec._FINAL_K)
    if not cands:
        return bad("该用户没有可召回候选")
    X = rec.features_for_rows([(uid, c["code"], c["routes"]) for c in cands], CTX)
    p = model.predict_proba(X)[:, 1]
    order = np.argsort(-p)
    rows = []
    for rank, j in enumerate(order[:topn], start=1):
        c = cands[j]
        info = zh_of(c["code"])
        info["rank"] = rank
        info["proba"] = round(float(p[j]), 4)
        info["routes"] = " / ".join(ROUTE_ZH[r] for r in c["routes"])
        rows.append(info)
    info = user_summary(uid)
    name = f"LR (LogisticRegression)" if ranker == "lr" else "LightGBM"
    if want == "json":
        return jsonify({"user_id": uid, "ranker": ranker, "topn": topn,
                        "user": info, "recommendations": rows})
    kpis = [("用户", str(uid)), ("历史", f"{info.get('n_invoices','?')} 张订单"),
            ("精排模型", name), ("召回候选", f"{len(cands)} 个")]
    tbl = [{"caption": f"召回(3路融合50) → {name} 精排 Top-{topn}",
            "cols": ["#", "商品码", "商品(中/英)", "类目", "单价£", "命中路由", "模型分P(y=1)"],
            "rows": [[r["rank"], r["code"], f"{r['desc_zh']}<br><span class='muted'>{r['desc']}</span>",
                      r["category"], r["price"], r["routes"], round(r["proba"], 4)]
                     for r in rows]}]
    return page("精排推荐", f"/recommend?user_id={uid}&topn={topn}&ranker={ranker}",
                kpis, tbl,
                "流程：三路召回（热门10/相似扩展20/用户向量20）按配额融合去重为候选池，"
                "提取 10 维特征，模型打分取 Top-N。分值是 P(候选∈用户下一单)。")


@app.get("/baseline")
def baseline():
    want = request.args.get("format", "html")
    uid = int(request.args.get("user_id", 0) or 0)
    topn = int(request.args.get("topn", 10))
    if uid not in CTX.user_row:
        return bad(f"用户 {uid} 不存在")
    pop = [c for c, _ in rec.recall_popular_only(CTX, uid, topn)]
    cf = [c for c, _ in rec.recall_itemcf_only(CTX, uid, topn)]
    fuse = [c["code"] for c in rec.recall_fusion(CTX, uid)[:topn]]
    # 说明：测试篮(该用户全量最后一单)是“事实答案”，仅演示/联调用，标注命中
    last_basket = CTX.recent_items.get(uid, set())
    columns = [("全局热门", pop), ("ItemCF 画像", cf), ("融合召回(未精排)", fuse)]
    out = []
    for col_name, codes in columns:
        col = []
        for rank, code in enumerate(codes, 1):
            z = zh_of(code)
            hit = code in last_basket
            z["rank"] = rank
            z["hit"] = hit
            col.append(z)
        out.append({"name": col_name, "items": col})
    if want == "json":
        return jsonify({"user_id": uid, "user": user_summary(uid),
                        "last_basket": sorted(last_basket), "columns": out})
    info = user_summary(uid)
    tables = []
    for col in out:
        tables.append({
            "caption": f"{col['name']} Top-{topn}（●=命中该用户最近一单）",
            "cols": ["#", "商品码", "商品(中/英)", "类目", "命中"],
            "rows": [[r["rank"], r["code"],
                      f"{r['desc_zh']}<br><span class='muted'>{r['desc']}</span>",
                      r["category"], ("<span class='hit'>●</span>" if r["hit"] else "")]
                     for r in col["items"]]})
    kpis = [("用户", str(uid)), ("历史", f"{info.get('n_invoices','?')} 张订单"),
            ("最近一单商品数", str(len(last_basket))), ("说明", "基线对照")]
    return page("基线对照（热门 vs ItemCF vs 融合召回）",
                f"/baseline?user_id={uid}&topn={topn}",
                kpis, tables,
                "三条基线都没有经过精排：热门=买家数最多的 Top；"
                "ItemCF=与用户已购画像最像的 Top；融合召回=三路按配额混合的 Top。"
                "与本页对比可见 LR/LGB 精排带来的命中提升（离线指标见报告）。")


def main() -> None:
    port = int(os.environ.get("PORT", "5001"))
    _load()
    print(f"[ready] {len(CTX.users):,} 用户 / {CTX.item_codes.shape[0]} 商品 / 词表 {len(KEYS)}")
    print(f"[ready] lr={'✓' if LR_PIPE is not None else '✗'} lgb={'✓' if LGB_MODEL is not None else '✗'}"
          f"  → http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
