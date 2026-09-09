# BASKET / HYPERGRAPH RESIDUAL —— 组员路线实验报告

- 路线：任务书「四、组员任务」Basket/Hypergraph Residual
- 日期：2026-09-09 ｜ 环境：本地 venv（Python 3.11.9；pandas/numpy/sklearn/lightgbm；BLAS 线程锁 1）
- 随机种子：SEED=42（全脚本固定）；脚本均在 `scripts/`，输入只读、绝不覆盖旧产物
- anchor：`data/candidate_aligned_oof_v15.csv` 的 `blend_v15`（146,530 行，2010-08/09/10）
- 评分：`composite=0.4·GAUC+0.4·NDCG@10+0.2·Recall@10`，组=`(user_id, snapshot_month)`，宏平均，单类组跳过（与 scripts/59、scripts/74 同函数）

> **一句话结论**：把真实 Invoice 购物篮形成的规范化共现 / 线性超图 / 受约束学习残差叠加到 v15 上，在 09 选型、10 冻结确认口径下**没有任何一条达到 +5e-4 的确认增益**；
> 线性超图与规范化共现排序几乎相同（行级 Pearson 0.9987、Top10 重叠 0.963），即**停止条件 4.4#1 命中**；
> Explore 正例的净救回为 0，Repeat 无净误踢但零净增益。**结论：该路线停止，不晋级、不做非线性编码器（Exp-H3 按门控跳过），冠军保留 anchor（alpha=0）**。

---

## 0. 交付物清单（outputs/experiment_basket_hypergraph/）

| 交付物 | 内容 | 状态 |
|---|---|---|
| `candidate_aligned_oof_basket.csv` | 146,530 行统一 8 列；`final_score=anchor_score`（alpha_final=0），`residual_score`=主残差信号组内百分位 | ✅ |
| `monthly_metrics.csv` | 逐月 anchor/final 指标与 delta（final=anchor） | ✅ |
| `alpha_sweep.csv` | 主残差列 alpha×月份 composite | ✅ |
| `normalization_ablation.csv` | 5 种归一化消融 + 相关 | ✅ |
| `cooccurrence_vs_hypergraph.csv` | H2 vs H1 相关 / Top10 重叠 / 指标 | ✅ |
| `repeat_explore_analysis.csv` | repeat/explore 分桶命中率 | ✅ |
| `experiment_config.json` | 配置、冻结项、采用判定 | ✅ |
| `BASKET_HYPERGRAPH_REPORT.md` | 本报告 | ✅ |
| （附）`h4_configs.csv` / `h4_saved_kicked_confirm.csv` / `residual_col_selection.csv` / `explore_rescue_diagnosis.csv` / `hist_basket_bucket_metrics.csv` / `dominance_diagnosis.csv` / `alpha_sweep_main.csv` / `monthly_metrics_*` / `h0_anchor_repro.txt` | 中间证据 | ✅ |

---

## 1. Exp-H0：复现 anchor（必须 PASS 才继续）

输入与评分器与任务书统一：组内宏平均、同 composite 定义。`blend_v15` 整体 `composite=0.57753`（GAUC 0.8675 / NDCG@10 0.2630 / Recall@10 0.6268），逐月：

| ym | composite | GAUC | NDCG@10 | Recall@10 | rows | users | pos |
|---|---|---|---|---|---|---|---|
| 2010-08 | 0.59729 | 0.88544 | 0.27365 | 0.66826 | 45,399 | 625 | 1,627 |
| 2010-09 | 0.57844 | 0.86126 | 0.26737 | 0.63493 | 48,097 | 653 | 2,313 |
| 2010-10 | 0.56130 | 0.85930 | 0.25050 | 0.58693 | 53,034 | 683 | 2,728 |

与既有基准 Δcomposite=+0.00000 → **PASS**（scripts/80；证据 `h0_anchor_repro.txt`）。

---

## 2. 数据与口径（时间安全）

- 篮 = `(CustomerID, InvoiceNo)`，篮内商品去重，服务码（POST/DOT/…）排除；批发大篮（中位 15、p90=45、p99=97、max 250）⇒ 默认 `1/sqrt(|B|-1)` 大篮归一。
- 特征仅用交易时间 `< snapshot 月 1 日` 的篮（`build_sim_matrix` 每月的 S 矩阵各自按当月 cutoff 切片）。11 月候选只用于扩 vocab（使未见候选有 embedding 列），不含任何未来标签/分数。
- 残差信号列（H1 主 `sim_last3_mean`，族含 `sim_last1/max/mean` 与自排除 `simnb_*`）；residual = 该列在 `(user,snapshot_month)` 组内等秩百分位（量纲统一，任务书第六节规则）；融合 `final = anchor + alpha·residual`，alpha=0 恒等于 anchor。

---

## 3. Exp-H1：规范化篮共现基线（任务书 4.2）

主方案 `cos_bigbasket`：带权篮频次 W（1/sqrt(|B|-1)）→ C=W Wᵀ → cosine。消融 `cos_raw`(0/1)、`jaccard`、`cond`(条件共现)、`cos_bigbasket_t90/t30`(时间衰减)。alpha 在 **2010-09 上 composite argmax** 选型，alpha_sweep 见 `alpha_sweep.csv`：

| alpha | comp_08 | comp_09 | comp_10 | comp_all |
|---|---|---|---|---|
| 0.0 | 0.59729 | **0.57844** | 0.56130 | 0.57753 |
| 0.01 | 0.59614 | 0.57726 | 0.56188 | 0.57701 |
| 0.1 | 0.59085 | 0.57215 | 0.55396 | 0.57082 |

正 alpha 全部单调下降，**09 上 argmax alpha*=0.0（即不融合）**。

归一化消融（`normalization_ablation.csv`，全部在 09 上 argmax=0、d=0）：

| scheme | 与主方案 sim 相关 | 说明 |
|---|---|---|
| cos_bigbasket（主） | 1.0000 | 1/sqrt(\|B\|-1)+cos |
| cos_raw | 0.9938 | 无大篮归一仍几乎同排序 ⇒ 大篮 O(m²) 不支配（见 §7） |
| jaccard | 0.9940 | 对称 Jaccard |
| cond | 0.9163 | 条件共现 P(j\|i)（最低，仍无增益） |
| cos_bigbasket_t90 | 0.9900 | 90 天指数衰减 |
| cos_bigbasket_t30 | 0.9645 | 30 天强衰减 |

残差列族（09 选型 / 10 冻结确认，`residual_col_selection.csv`）——**选型月最好也只 +0.0002，确认月全部 ≤0**：

| 残差列 | alpha*_09 | d09 | d10(冻结) |
|---|---|---|---|
| sim_last1 | 0.002 | +0.00007 | −0.00030 |
| sim_last3_max / mean | 0.000 | 0.00000 | 0.00000 |
| simnb_last1 | 0.002 | +0.00019 | **−0.00053** |
| simnb_last3_max | 0.002 | +0.00020 | −0.00017 |
| simnb_last3_mean | 0.002 | +0.00016 | −0.00026 |

α* 的 +1e-4 级"增益"在 10 月确认时全部转负 → 09 选型噪声，非真实信号。

---

## 4. Exp-H2：线性 Hypergraph 对照（任务书 4.2）

item→basket incidence `H`，权重 `1/sqrt(|B|)`（篮度归一），`S=cos(HHᵀ)`，即规范化 item→basket→item 线性传播（`scheme=hg_linear`）。与 H1 用同一行特征聚合、同 alpha 预算对照（`cooccurrence_vs_hypergraph.csv`）：

| 对照量 | 值 |
|---|---|
| 行级 Pearson（残差信号） | **0.99866** |
| 组内 Spearman 均值 | 0.99551 |
| residual Top10 商品重叠率 | **0.96313** |
| H1 / H2 d09（alpha*=.002） | +0.00016 / +0.00015 |
| H1 / H2 d10（冻结确认） | −0.00026 / −0.00028 |

→ 线性超图传播**只是另一种共现归一化**：排序与规范化共现基本相同，指标无差异。**停止条件 4.4#1 直接命中**。

---

## 5. Exp-H4：受约束 residual（任务书 4.2 的核心对照）

将可用的编码器（H1 规范化共现族、H2 线性超图）各自特征接入**同一简单 residual ranker**（LogisticRegression 与浅层 LGB：depth≤2、≤80 树、l2/子采样约束），预测 label=1 的组内百分位作 residual，走**同一 alpha 预算**。严格 ladder：fit 08 → 在 09 选 alpha*/配置 → 冻结 → fit 08+09 → 在 **10 确认**（`h4_configs.csv`）：

| config | alpha*_09 | d09 | comp10_anchor | comp10_final | d10(确认) |
|---|---|---|---|---|---|
| H1 特征 + LR | 0.000 | 0.00000 | 0.56130 | 0.56130 | 0.00000 |
| H1 特征 + LGB | 0.002 | 0.00000 | 0.56130 | 0.56134 | **+0.00004** |
| H2 特征 + LR | 0.000 | 0.00000 | 0.56130 | 0.56130 | 0.00000 |
| H2 特征 + LGB | 0.002 | +0.00002 | 0.56130 | 0.56129 | −0.00001 |

确认月 10 救回/误踢（`h4_saved_kicked_confirm.csv`）：

| config | repeat 正例 救回/误踢(净) | explore 正例 救回/误踢(净) |
|---|---|---|
| H1+LGB | 7 / 6（**+1**） | 0 / 0（0） |
| H2+LGB | 7 / 7（0） | 0 / 0（0） |
| LR（两种特征） | 0 / 0 | 0 / 0 |

即：即便让一个浅层学习器按 label 拟合残差，确认月最好也只有 **net +1 个 repeat** 正例被救、explore **零救回**——在学习器层面复现了"残差无可用增量"。H1 与 H2 特征接入同 ranker 表现相同 → 按任务书 **"Hypergraph 不能稳定超过规范化共现 ⇒ 保留简单共现"**。

---

## 6. Exp-H3：轻量非线性篮编码器 —— 按门控跳过（不浪费算力）

任务书 Exp-H3 明确**只在 Exp-H1/H2 完成后**评估，且晋级门槛要求相对规范化共现产生"可见的额外净增量"、两者差异需超出评估波动。本路线实际观测：

1. H1 简单共现：09 argmax=0（无增益），confirm=0；
2. H2 线性超图 = H1（corr 0.999、Top10 重叠 0.96，停止条件 #1）；
3. H4 证明**即使是非线性浅层学习器**（LGB depth 2，本质覆盖 H3 的浅层非线性容量）在相同特征与预算下也零增益（confirm ≤ +4e-5）。

三项全部为空 ⇒ 没有任何证据表明"换更强的非线性集合编码器（H3：32 维 item embedding + 1–2 层 + 篮级非线性聚合）"能产生超图尚未提供的增量；瓶颈在信号本身而非模型容量（见 §8 机理）。据此**跳过 Exp-H3**，符合停止条件与"保留简单共现"指引。

---

## 7. 分析责任（任务书 4.3）逐项

- **归一化消融**：见 §3 表。5 种方案 09 上全部 argmax=0。
- **线性超图 vs pairwise 相关/Top10 重叠**：Pearson 0.9987、Spearman 0.9955、Top10 重叠 0.963（§4）。
- **共现 vs 非线性对照**：H3 按门控跳过；H4 的浅层 LGB 已是"受约束非线性 residual"，H1/H2 特征表现相同（§5）。
- **Repeat/Explore 分桶指标**（`repeat_explore_analysis.csv`）：

| ym | bucket | 候选行 | 正例 | anchor top10 命中率 | top30 后正例占比 |
|---|---|---|---|---|---|
| 08 | explore / repeat | 39,845 / 5,554 | 454 / 1,173 | 12.8% / 63.4% | 50% / 12% |
| 09 | explore / repeat | 41,920 / 6,177 | 779 / 1,534 | 13.0% / 56.9% | 59% / 18% |
| 10 | explore / repeat | 45,734 / 7,300 | 1,044 / 1,684 | 17.2% / 48.4% | 54% / 24% |

  → Explore 才是 anchor 的主要短板（过半正例在 top30 之外），但残差**救不回**（见下）。
- **Explore 净救回 / Repeat 误踢**：确认月 10 上，加性诊断 alpha=0.002 在全部桶中 `explore_saved=0`（`hist_basket_bucket_metrics.csv`）；H4 学习器同样 explore 零救回、repeat 净 ≤ +1。即"提升不来自把 Explore 拉进 top10"。
- **按历史篮数 / 最近篮大小分桶**（confirm 10，alpha=0.002 仅诊断方向）：

| 桶 | 组数 | pos(rep/exp) | d_alpha002 | repeat 救/踢 | explore 救/踢 |
|---|---|---|---|---|---|
| 历史篮 1-3 | 303 | 188/336 | −0.00090 | 0/0 | 0/1 |
| 历史篮 4-8 | 216 | 350/314 | −0.00066 | 0/1 | 0/0 |
| 历史篮 9-15 | 97 | 191/131 | +0.00015 | 0/0 | 0/0 |
| 历史篮 16+ | 67 | 955/263 | +0.00091 | 7/7 | 0/0 |
| 最近篮 ≤1-2 件 | 48/18 | 84/36,11/7 | +0.0012 / 0 | 1/1,0/0 | 0/0 |
| 最近篮 11-30 件 | 260 | 442/356 | −0.00116 | 1/3 | 0/1 |
| 最近篮 31+ 件 | 184 | 865/455 | −0.00003 | 4/3 | 0/0 |

  历史篮 16+ 的 +0.0009 完全来自 **repeat 相互顶替（救 7 = 踢 7）**，explore 依旧 0 救回；样本小的桶指标方差大不作结论。没有任何桶出现 explore 净救回。
- **最大篮/高频商品支配**（`dominance_diagnosis.csv`）：2010-10 cutoff 前 13,514 个篮，size 中位 15 / p90 45 / p99 97 / max 250；按大篮权重 `w∝1/sqrt(B-1)` 计，top-1% 篮仅占 2.7% 总权重；`cos_raw` vs `cos_bigbasket` 排序相关 0.9938 ⇒ **O(m²) 大篮与高频商品不支配相似度**，且 cos 归一已把二者压平。
- **alpha 稳定性**：任何正 alpha 逐月/整体单调下滑（§3 表）；最优 alpha 恒落在边界 0（或 09 噪声 0.002 且确认转负）；相邻 alpha 无平台 ⇒ 不可依赖单点 alpha。

---

## 8. 为什么是空的：机理诊断（explore_rescue_diagnosis.csv）

confirm 月 10 上，候选对用户最近篮的自排除共购相似度 `simnb_last3_mean`：

| 分桶 | rows | frac(sim>0) | mean sim |
|---|---|---|---|
| explore 负例 | 44,690 | 0.989 | 0.0506 |
| explore 正例 | 1,044 | 0.989 | 0.0688 |
| repeat 负例 | 5,616 | 0.994 | 0.1515 |
| repeat 正例 | 1,684 | 0.979 | 0.1487 |
| pos\|explore\|top10 / 11_30 / below30 | 180/298/566 | ~0.99 | 0.0657 / 0.0715 / 0.0685 |
| pos\|repeat\|top10 / 11_30 / below30 | 815/462/407 | ~0.96-1.0 | 0.1693 / 0.1450 / 0.1117 |

要点：共现图密（几乎所有候选都有非零相似度），但
- **Repeat** 正例的 co-occurrence 与 anchor 排位单调对齐（0.169→0.112）——即"用户最近篮里买过的商品"这类信息 v15 **已经吃进**，残差没有新梯度；
- **Explore** 正例虽然 mean sim 略高于 explore 负例（0.069 vs 0.051），但在 anchor 的 top10/11_30/below30 三带内几乎恒定（0.066/0.072/0.069）——对这些被 anchor 排低的 explore 正例，篮共现**不随排位提供区分度**，residual 百分位只能把组内大量 0.05–0.07 的近等值候选做噪声性洗牌。

因此残差要么顺着 anchor 已会排对的 repeat 走（顶替，零净益），要么在 explore 的近等值区做无梯度重排（无净救回）——这解释了 H1/H2/H4 全空，也说明该瓶颈**与模型容量无关**，非线性编码器（H3）无法突破。

---

## 9. 任务书「九、最终报告必须回答的问题」

**1. 新信号与 v15 已有特征有什么实质区别？**
篮共现理论上提供"用户最近一次/三次购物篮 ↔ 候选"的二阶局域亲合，与 v15 的活动/流行度特征机制不同。但实测：repeat 候选的共现强度与 anchor 排位单调一致（§8），v15 已充分吸收；explore 候选的共现信号弱且在锚排位各带近乎恒定。**机制上不同、增量上无可用差异。**

**2. 增益来自哪个月、哪类用户、Repeat/Explore 哪类？**
无实质增益。选型月 09 的最好单配置（simnb，+0.0002）与学习器（+0.00004）在确认月 10 全部归零/转负。诊断性重排唯一正桶（历史篮 16+，+0.0009@alpha.002）来自 repeat 相互顶替；**explore 全路线零净救回**。

**3. 救回多少、误踢多少？**
确认月 10：H4 学习器最好 net **+1**（repeat 救 7 / 踢 6，explore 0/0）；H2 特征 net 0（7/7）；LR 0/0；加性残差 explore_saved 全域 0。Repeat 无净误踢但无净增益。

**4. 增益在相邻 alpha、月份、用户组中稳定吗？**
不稳定。正 alpha 逐月单调降（§3）；最优 alpha 落在边界 0 或为 09 噪声，且任何非零配置在 10 都退化（d10 ≤ 0，除 H4 h1|lgb +0.00004）。SEED 固定、网格唯一，不存在"多 seed 幸存"问题。

**5. 简单模型是否已取得同等效果？**
是，且是决定性的。规范化共现（H1）与线性超图（H2）行级相关 0.9987、Top10 重叠 0.963、指标差 <3e-4；同一 ranker 下 H1/H2 特征表现相同。按任务书指引，**保留简单共现（此处因无增益即为不采用）**。

**6. 是否有时间穿越 / 标签泄漏 / 候选偏差 / 口径问题？**
自查未见：(a) 每月 S 矩阵与用户特征只用 `< snapshot cutoff` 的交易；(b) 11 月候选仅扩 vocab 索引；(c) 08/09 只做 debug/选型，10 仅确认，且确认月用 08+09 重拟合模型（时间安全）；(d) 评分器与 scripts/74 同函数、组=(user,snapshot_month) 宏平均、单类组跳过；(e) residual 为组内百分位、alpha=0 恒等 anchor（未引入截断/并列篡改）。候选行来自 leader 统一对齐文件，无自行负采样。

**7. 该路线应晋级、保留为简单特征，还是停止？**
**停止。** H1/H2 等价（停止条件 #1）、确认月退化/归零（#4）、explore 零救回（#3）、无多 seed 幸存（#5）多重命中停止条件；且 H4 证实容量不是瓶颈。**不采用任何残差融合，冠军 = anchor（alpha_final=0）**。OOF 交付文件中保留 `residual_score` 以便未来若出现新的时间安全信号可再次融合比对。

---

## 附录 A：冻结与复现

- alpha 候选网格：`[0, .002, .005, .01, .02, .03, .05, .07, .1, .15, .2, .3, .5, .75, 1.0]`（部分脚本截断到 .5，均在 `experiment_config.json` 记录）
- 脚本：`80_exp_h0_repro_anchor.py`、`81_exp_h1_cooccurrence.py`、`82_exp_h2_hypergraph.py`、`84_exp_h4_residual.py`、`85_consolidate_deliverables.py`、`86_h3_gate_diagnosis.py`、`87_hist_basket_buckets.py`
- 源码：`src/basket_experiment/{io_data,features,run_features,eval,metrics}.py`
- 复核锚：脚本 80 复现 composite=0.57753 Δ=0（PASS），与 `scripts/74` 基准一致；residual 列族选型/确认表见 §3。
