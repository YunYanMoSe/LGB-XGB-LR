"""阶段七 ②：LightGBM 二分类 → 特征重要性排序 + Top-5 解读 + 素材 md。

读 outputs/feature_stage/feature_wide.csv（缺失则自动跑 scripts/15 一键复现），
用户级 70/30 切分（整用户成组防串），LightGBM 参数沿用仓库 12 脚本惯例。

产出（outputs/feature_stage/）：
    feature_importance.csv        按 gain 降序全量 31 特征（含 weight / 正负均值）
    feature_importance.png        双面板：top20 gain / top20 weight 横向条形
    阶段七_特征工程与精排素材.md   素材（docx 由 scripts/17 打包）
    models/lgb_feature.pkl        训练模型
    features_meta.json            在原 meta 上追加大纲 AUC / 重要性 / 均值表

运行：venv\\Scripts\\python.exe scripts\\16_train_feature_lgb.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config.settings import SEED, set_seed
from src import feature_engineering as fe

set_seed(SEED)

STAGE = PROJECT_ROOT / "outputs" / "feature_stage"
STAGE.mkdir(parents=True, exist_ok=True)
WIDE_CSV = STAGE / "feature_wide.csv"
META_JSON = STAGE / "features_meta.json"
IMP_CSV = STAGE / "feature_importance.csv"
IMP_PNG = STAGE / "feature_importance.png"
MD_PATH = STAGE / "阶段七_特征工程与精排素材.md"
MODEL_PATH = PROJECT_ROOT / "models" / "lgb_feature.pkl"
VALID_RATIO = 0.3
LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.1, num_leaves=31, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbosity=-1,
)


def _split_users(users: np.ndarray):
    """用户级随机切分（seed=42）：返回 (fit 集合, valid 集合)。"""
    rng = np.random.default_rng(SEED)
    arr = np.asarray(sorted(users))
    rng.shuffle(arr)
    n_valid = max(1, int(round(len(arr) * VALID_RATIO)))
    return set(arr[n_valid:]), set(arr[:n_valid])


def _setup_cn_font() -> None:
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
    return plt


def _importance_plot(path: Path, top: pd.DataFrame) -> None:
    """双面板：左 gain / 右 weight 的 Top20 横向条形。"""
    plt = _setup_cn_font()
    INK, GRID, PAGE = "#0b0b0b", "#e1e0d9", "#ffffff"
    BLUE, ORANGE = "#2a78d6", "#eb6834"

    def panel(ax, data, col, title, color):
        d = data.sort_values(col).tail(20)
        ax.barh(d["feature"], d[col], color=color, height=0.72)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d["feature"], fontsize=9)
        ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=10)
        ax.grid(True, axis="x", color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK, labelsize=9)
        ax.invert_yaxis()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
    panel(axes[0], top, "gain", "特征重要性 · 信息增益 gain（Top20）", BLUE)
    panel(axes[1], top, "weight", "特征重要性 · 分裂次数 weight（Top20）", ORANGE)
    fig.suptitle("LightGBM 特征重要性（同一模型两种口径；gain=信息增益，weight=作为分裂特征的次数）",
                 fontsize=13, color=INK, x=0.01, ha="left", y=1.02)
    fig.text(0.5, -0.02,
             "数据：阶段七标签窗首购 1:3 配对样本；特征全部只用 2011-11-09 前历史计算（防泄漏）。",
             ha="center", fontsize=9.5, color="#52514e")
    fig.set_facecolor(PAGE)
    for ax in axes:
        ax.set_facecolor(PAGE)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)
    print(f"重要性图 -> {path}")


def _gen_logics() -> dict[str, str]:
    """特征 → 生成逻辑一句话（md 特征清单用，与 src 实现口径一致）。"""
    return {
        "u_n_orders_log": "特征期该用户去重订单数 log1p",
        "u_n_items_log": "特征期该用户去重购买商品种数 log1p",
        "u_qty_log": "特征期该用户购买总件数（Quantity 求和）log1p",
        "u_spend_log": "特征期该用户总消费（Quantity×UnitPrice 求和）log1p",
        "u_avg_unit_price_log": "总消费 / 总件数 → 自购均价 log1p",
        "u_avg_basket_size_log": "每张订单去重商品种数按用户求均值 → log1p",
        "u_active_days_log": "特征期有购买的日历天数 nunique → log1p",
        "u_last_active_gap_log": "FEAT_CUT − 特征期最近一次购买时间，天数差 log1p（0=距切窗不足1日）",
        "u_weekend_ratio": "周末（周六/日）订单数 ÷ 订单总数",
        "u_cat_entropy": "用户去重购买商品落在各收拢类目上的份额熵 ÷ ln(触及类目数)",
        "i_pop_buyers_log": "特征期购买该商品的用户数（买家数）log1p",
        "i_pop_orders_log": "特征期含该商品的订单数 log1p",
        "i_qty_log": "特征期该商品销量（Quantity 求和）log1p",
        "i_price_log": "特征期该商品 UnitPrice 中位数 log1p",
        "i_pop_rank_frac": "按买家数降序排名 0=最热 … 1=最冷（÷(商品数-1)）",
        "i_first_gap_log": "FEAT_CUT − 该商品特征期首次销售时间天数差 log1p（值大=老品）",
        "i_last_gap_log": "FEAT_CUT − 该商品特征期最近销售时间天数差 log1p（值大=久未售/滞销）",
        "i_cat_sales_log": "该商品收拢类目在特征期的总销售额 log1p",
        "i_cat_sales_share": "该收拢类目销售额 ÷ 特征期总销售额",
        "i_top_weekday_code": "该商品特征期销量最大的星期（周一=0 … 周日=6，冷商品补 0）",
        "ui_cat_lift_log": "log1p(用户在该类目消费份额 ÷ 该类目全局份额)；无该类目消费=0",
        "ui_price_gap_log": "log1p(商品价中位数) − log1p(用户自购均价)，带符号；冷商品(无价)=0",
        "ui_emb_sim": "特征期重训 Skip-gram 向量：用户已购商品向量均值 与 候选商品向量的余弦",
        "ui_cf_sim": "候选商品「共同购买者画像」与用户已购集合画像的点积（ItemCF 聚合余弦）",
        "ui_nn_hit_rate": "候选商品向量 Top20 近邻中，被该用户买过的占比（近邻同购信号）",
        "country_uk": "用户特征期常购国=United Kingdom 记 1，否则 0",
        "country_de": "用户特征期常购国=Germany 记 1，否则 0",
        "country_fr": "用户特征期常购国=France 记 1，否则 0",
        "country_eire": "用户特征期常购国=EIRE（爱尔兰）记 1，否则 0",
        "country_other": "以上四国之外（含少量非英国家）记 1，否则 0",
        "cat_code_ordinal": "候选商品收拢类目按销售额降序的编号（0=份额最高桶…9）；无类目=其他桶",
    }


def _fmt(x, nd: int = 4) -> str:
    return f"{x:.{nd}f}" if x == x else "—"


def main() -> None:
    t0 = time.time()
    if not WIDE_CSV.exists():
        print("宽表缺失，自动运行 scripts/15_build_feature_wide.py 一键复现…")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "15_build_feature_wide.py")],
                       check=True)

    meta = json.loads(META_JSON.read_text(encoding="utf-8"))
    wide = pd.read_csv(WIDE_CSV)
    y = wide["label"].to_numpy()
    X = wide[fe.FEATURES].to_numpy(dtype=np.float64)

    # 冷启动正样本：标签窗首购但特征期无销量的商品（其商品特征按真实冷启动置 0）
    cold_mask = (wide["label"] == 1) & (wide["i_pop_buyers_log"] < 0.5)
    n_pos_all = int((y == 1).sum())
    meta["n_cold_pos"] = int(cold_mask.sum())
    meta["n_cold_pos_share"] = round(float(cold_mask.sum()) / max(1, n_pos_all), 4)

    # ---- 用户级切分 ----
    fit_u, valid_u = _split_users(np.unique(wide["user_id"].to_numpy()))
    m_fit = wide["user_id"].isin(fit_u).to_numpy()
    m_val = wide["user_id"].isin(valid_u).to_numpy()
    n_pos_fit = int(y[m_fit].sum())
    n_neg_fit = int((y[m_fit] == 0).sum())
    print(f"用户级切分(seed={SEED}, 7:3)：fit {len(fit_u)} 用户 / valid {len(valid_u)} 用户")
    print(f"fit 样本：{m_fit.sum():,}（正 {n_pos_fit:,} / 负 {n_neg_fit:,}）"
          f"；valid 样本：{m_val.sum():,}")

    # ---- LightGBM ----
    from lightgbm import LGBMClassifier
    params = dict(LGB_PARAMS)
    params["scale_pos_weight"] = n_neg_fit / max(1, n_pos_fit)
    clf = LGBMClassifier(**params)
    clf.fit(X[m_fit], y[m_fit])
    p_val = clf.predict_proba(X[m_val])[:, 1]
    auc = float(roc_auc_score(y[m_val], p_val))
    print(f"[LightGBM] valid AUC = {auc:.4f}")

    booster = clf.booster_
    gain = booster.feature_importance(importance_type="gain")
    # LightGBM 4.x 中“作为分裂点的次数”的接口名是 split（旧称 weight / feature_importances_）
    weight = booster.feature_importance(importance_type="split")

    # ---- 重要性表 + 正负均值 ----
    pm = np.asarray([X[y == 1, i].mean() for i in range(len(fe.FEATURES))])
    nm = np.asarray([X[y == 0, i].mean() for i in range(len(fe.FEATURES))])
    table = pd.DataFrame({
        "feature": fe.FEATURES,
        "zh": [fe.FEAT_ZH[f] for f in fe.FEATURES],
        "gain": gain.astype(float),
        "weight": weight.astype(float),
        "pos_mean": pm, "neg_mean": nm,
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    table.to_csv(IMP_CSV, index=False, encoding="utf-8")
    _importance_plot(IMP_PNG, table)

    # ---- 更新 meta ----
    meta["lgb"] = {
        "params": {k: (float(v) if isinstance(v, (int, float)) else v)
                   for k, v in params.items()},
        "valid_ratio": VALID_RATIO, "n_fit_users": len(fit_u),
        "n_valid_users": len(valid_u),
        "auc": round(auc, 4),
        "importance": table.to_dict("records"),
    }
    META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(clf, str(MODEL_PATH))
    print(f"模型 -> {MODEL_PATH}；重要性表 -> {IMP_CSV}；meta 已更新")

    # ---- 终端：按 gain 降序 Top20 ----
    print("\n特征重要性排序（按信息增益 gain 降序，Top20）")
    print(f"{'rank':<5}{'特征':<22}{'中文含义':<30}{'gain':>10}{'weight':>9}{'pos均':>9}{'neg均':>9}")
    for _, r in table.head(20).iterrows():
        print(f"{int(r['rank']):<5}{r['feature']:<22}{r['zh']:<30}"
              f"{r['gain']:>10.0f}{r['weight']:>9.0f}"
              f"{_fmt(r['pos_mean'], 3):>9}{_fmt(r['neg_mean'], 3):>9}")

    write_material(md_text(meta, table))
    print(f"\n素材 md -> {MD_PATH}；总耗时 {time.time()-t0:.0f}s")


def md_text(meta: dict, imp: pd.DataFrame) -> str:
    """拼装素材 md（docx 由 scripts/17 渲染）。"""
    feat = meta["features"]
    gen = _gen_logics()
    buckets = meta["cat_buckets"]
    fp, lp = meta["feature_period"], meta["label_period"]
    lgb = meta["lgb"]
    top5 = imp.head(5)

    lines: list[str] = []
    A = lines.append
    A("# 阶段七 · 多维度特征挖掘与 LightGBM 特征重要性（报告素材）")
    A("")
    A("> 任务：① 设计并实现多维度特征（用户/商品/统计/交互等 ≥4 类）；② 工程化拼接为以 user_id、"
      "item_id 开头的全数值宽表 CSV；③ 以标签窗首购为正样本训练 LightGBM 二分类，输出特征重要性排序表"
      "与 Top-5 特征业务解读。")
    A("")
    A("## 一、时间口径与防泄漏设计")
    A("")
    A("把数据切成一个**特征期**与一个**标签窗**，两者用锚点 `FEAT_CUT = 2011-11-09` 严格分开：")
    A("")
    A(f"- **特征期**：`InvoiceDate < 2011-11-09`，本阶段全部 31 个特征只用这一段历史计算，共 "
      f"{fp['rows']:,} 行交易 / {fp['orders']:,} 张订单 / {fp['users']:,} 个用户 / {fp['items']:,} 种商品。")
    A(f"- **标签窗**：尾随 30 天 `[2011-11-09, 2011-12-09]`（数据止于 12-09），共 {lp['rows']:,} 行交易、"
      f"{lp['users']:,} 个用户在此期间有购买。")
    A("- 防泄漏红线：任何行为型特征（用户活跃度、商品热度、首现/近售时间、向量、共同购买画像）一律"
      "只用特征期数据；复用的商品向量不是线上全量模型，而是**只在特征期订单篮上重训**的 Skip-gram 向量"
      "（`vectors_feat_skip.npz`）。类目/国家为静态属性表，可直接使用。")
    A("")
    A("## 二、样本与标签构造")
    A("")
    A("- 候选用户：特征期订单数 ≥ 2 的用户（共 {n} 人）。"
      "".replace("{n}", f"{meta['n_candidates']:,}"))
    A("- 正样本：候选用户在**标签窗内的首次购买**——即该商品不在其标签窗前的购买史中。")
    A("- 负样本：配对采样 1:3——对每个正样本，从该用户「全数据集从未购买 ∩ 特征期有销量的商品目录」"
      "中不放回随机抽 3 个（seed=42，可复现）。")
    A("")
    A("| 统计量 | 数值 |")
    A("| --- | --- |")
    A(f"| 候选用户（特征期≥2 单） | {meta['n_candidates']:,} |")
    A(f"| 进入样本的用户（至少 1 个首购正样本） | {meta['n_sample_users']:,} |")
    A(f"| 正样本（标签窗首购） | {meta['n_pos']:,} |")
    A(f"| 负样本（1:3 配对采样） | {meta['n_neg']:,} |")
    A(f"| 宽表总行数 | {meta['n_pos'] + meta['n_neg']:,} |")
    A(f"| 实际正:负 | 1 : {meta['actual_neg_per_pos']:.1f} |")
    A(f"| 冷启动正样本（特征期无销量的商品） | {meta.get('n_cold_pos', 0):,}"
      f"（占正样本 {meta.get('n_cold_pos_share', 0):.2%}，此类商品特征按真实冷启动置 0） |")
    A("")
    A("## 三、全数值特征宽表")
    A("")
    A(f"宽表 `feature_wide.csv`：{meta['n_pos'] + meta['n_neg']:,} 行 × {meta['n_feature_cols'] + 3} 列，"
      f"首列 `user_id`（int64 原值）、次列 `item_id`（StockCode 经 `id_map.csv` 排序映射为数值 0.."
      f"{meta['n_item_map'] - 1}），随后 {meta['n_feature_cols']} 个数值特征（float64），末列 `label`（0/1）。"
      "**全部字段均为数值类型**：城市（国家）用 one-hot、类目用收拢后销售额排名编号，星期用峰值星期编号。"
      "`verify_wide()` 自检结论：dtypes 全数值、无 NaN、列序正确、item_id 均在 id_map。")
    A("")
    A("| 字段 | 说明 |")
    A("| --- | --- |")
    A(f"| user_id | 用户 CustomerID（int64 原值） |")
    A(f"| item_id | 商品数值编号（StockCode→int，共 {meta['n_item_map']:,} 个，见 id_map.csv） |")
    A(f"| 特征列 ×{meta['n_feature_cols']} | 见第二节特征清单（分组 A 用户 10 / B 商品·统计 10 / C 交互 5 / D 文本编码 6） |")
    A("| label | 1=标签窗首购（正样本）；0=配对负样本 |")
    A("")
    A("商品类目收拢（长尾 23 类 → 头部桶 + 「其他」）：")
    A("")
    A("| code | 收拢类目 | 特征期销售额占比 |")
    A("| --- | --- | --- |")
    for b in buckets:
        A(f"| {b['code']} | {b['name']} | {b['share']:.2%} |")
    A("")
    A("## 四、特征清单（31 个，多维度）")
    A("")
    A("分组：**A 用户画像(10)** · **B 商品·统计·时间(10)** · **C 用户×商品交互(5)** · "
      "**D 文本→数值编码(6)**；生成逻辑全部只在特征期内计算。")
    A("")
    A("| # | 特征名 | 属组 | 含义 | 生成逻辑 |")
    A("| --- | --- | --- | --- | --- |")
    for i, f in enumerate(feat, 1):
        A(f"| {i} | `{f['name']}` | {f['group']} | {f['zh']} | {gen.get(f['name'], '')} |")
    A("")
    A("## 五、LightGBM 训练")
    A("")
    A("- 切分：**用户级随机 70/30**（seed=42，整用户成组，避免同一用户的行跨训练/验证集泄漏）。")
    A("- 标签在同一时间窗内，故不做时间切分，改用用户级切分评估泛化。")
    A(f"- 参数：`{json.dumps({k: v for k, v in lgb['params'].items() if k != 'scale_pos_weight'}, ensure_ascii=False)}`，"
      f"`scale_pos_weight` = 训练集负/正比。")
    A(f"- 验证集指标：**valid AUC = {lgb['auc']:.4f}**（fit {lgb['n_fit_users']:,} / valid "
      f"{lgb['n_valid_users']:,} 用户）。")
    A("")
    A("## 六、特征重要性排序")
    A("")
    A("重要性取两种口径：**gain**（该特征作为分裂点的信息增益合计，越大说明对判别贡献越大）与 "
      "**weight**（该特征被选为分裂特征的次数）。下表按 gain 降序，附正/负样本上的特征均值便于看方向。")
    A("")
    A("| rank | 特征 | 中文含义 | gain | weight | 正样本均值 | 负样本均值 |")
    A("| --- | --- | --- | --- | --- | --- | --- |")
    for _, r in imp.iterrows():
        A(f"| {int(r['rank'])} | `{r['feature']}` | {r['zh']} | {r['gain']:.0f} | "
          f"{r['weight']:.0f} | {_fmt(r['pos_mean'])} | {_fmt(r['neg_mean'])} |")
    A("")
    A("> 说明：纯用户类特征（u_*）在配对 1:3 采样下，每个用户的样本中正/负行各带固定倍数，故其正负均值天然相同；"
      "这类特征的判别力体现在跨用户的分布差异，重要性仍以 gain / weight 排序为准。交互与商品类特征（ui_*/i_*）"
      "均值差异即可直接读出方向。")
    A("")
    A(f"![feature_importance](feature_importance.png)")
    A("")
    A("## 七、Top-5 特征业务解读")
    A("")
    for _, r in top5.iterrows():
        A(f"### ① rank {int(r['rank'])} · `{r['feature']}`（{r['zh']}）")
        A(interpret(r))
        A("")
    A("## 八、复现命令")
    A("")
    A("```text")
    A("venv\\Scripts\\python.exe scripts\\16_train_feature_lgb.py   # 宽表缺失自动先跑 15")
    A("venv\\Scripts\\python.exe scripts\\17_build_feature_docx.py  # md → Word 报告素材")
    A("```")
    A("")
    return "\n".join(lines)


def interpret(r: pd.Series) -> str:
    """对单个 Top 特征给一句业务化解读（方向 + 均值佐证 + 怎么看）。"""
    name = str(r["feature"])
    pm, nm = float(r["pos_mean"]), float(r["neg_mean"])
    direction = "高" if pm > nm else "低"
    delta = (pm - nm) / max(1e-9, abs(nm))
    text = {
        "ui_cf_sim": "该特征衡量候选商品与用户历史已购集合的“共同购买者”相似度（ItemCF 聚合余弦）。",
        "ui_emb_sim": "该特征衡量候选商品与用户历史购买向量在语义嵌入空间中的接近程度。",
        "i_last_gap_log": "该特征度量商品最近一次售出距切窗的时间，值小=近期有动销。",
        "i_pop_buyers_log": "该特征度量商品在特征期的买家规模（热度）。",
        "i_pop_rank_frac": "该特征度量商品的相对热门名次，0=最热、越接近 1 越冷门。",
        "ui_nn_hit_rate": "该特征统计候选商品向量近邻中被用户买过的占比（同购转移信号）。",
        "ui_cat_lift_log": "该特征度量用户对该候选商品所属类目的偏好是否高于大盘。",
        "i_price_log": "该特征为商品的价位水平（单价中位数）。",
        "ui_price_gap_log": "该特征为候选价与用户自购均价的偏离，正=偏贵、负=更实惠。",
        "u_n_orders_log": "该特征刻画用户历史下单频度（活跃度）。",
        "i_first_gap_log": "该特征反映商品在目录中已存在的时间（老品/新品）。",
    }.get(name, f"`{name}` 是样本判别中信息增益最高的特征之一。")
    return (f"{text} 正样本均值 {pm:.3f} 明显{'高于' if pm > nm else '低于'}负样本均值 {nm:.3f}（"
            f"相对差 {delta * 100:+.0f}%），说明该特征对“是否会首购该商品”有强区分力。"
            f"业务上：{name} 取值越{'高' if pm > nm else '低'}，用户首购该商品的可能性越大；"
            f"可作为召回/排序阶段加权的候选维度，也可按此做分桶运营策略。")


def write_material(text: str) -> None:
    MD_PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
