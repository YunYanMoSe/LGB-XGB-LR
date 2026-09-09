# 推荐系统项目（Online Retail）

基于公开的英国在线零售（Online Retail）交易数据构建的**商品推荐系统**：从原始订单表出发，经过数据清洗与探索分析，训练**商品向量（Item2Vec）**，实现**协同过滤与多路召回**，再用 **LR / LightGBM** 对召回候选精排，最终以**离线静态网页 + Flask 在线接口**两种方式演示推荐效果。

所有脚本幂等、各自打印结果并存档；评估采用一致的留出口径并固定随机种子，结果可复现。

> **📦 公开仓库说明（2026-09 结档）**：本仓库**只含代码与文档，不含任何数据/模型文件**。
> `data/`、`models/`、`train_data/` 及全部 `*.csv / *.pkl / *.npz` 已由 `.gitignore` 排除——
> 课程数据不随公开仓库分发，克隆后运行脚本需自备数据（旧演示线为公开 UCI Online Retail
> 数据集；`scripts/18` 起的「候选集打分」线为课程内部赛题数据）。目录树与文件表中的 `data/`
> 引用在克隆环境里不可用，属预期。
>
> 仓库含两条线：**01–17 旧答辩演示线**（本文档主体）+ **18–88 候选集打分竞赛线**
> （2026-09 新增，见文末「候选集打分线」）。

## 主要能力

| 能力 | 说明 | 对应代码 |
| ---- | ---- | ---- |
| 数据清洗与 EDA | 三级清洗漏斗 + 留存报告 + 可视化 HTML 报告 | `scripts/01`、`scripts/02` |
| 协同过滤 | 用户相似 / 商品相似（余弦，UserCF / ItemCF） | `scripts/03`–`05` |
| 商品向量 | 以「每笔订单 = 句子、订单内商品 = 词」训练 CBOW / Skip-gram | `scripts/06` |
| 相似推荐网页 | 相似用户 / 相似商品静态页（ItemCF、CBOW、Skip-gram 三口径） | `scripts/04`、`scripts/07` |
| 用户购买序列与向量索引 | 长表序列 + 在线相似接口用的模型与索引 | `scripts/10` |
| 多路召回 | 热门 / 相似扩展 / 用户向量 配额融合 → 候选集 `recall_set.csv` | `scripts/11` |
| 召回精排 | 候选集上打 10 维特征 → LR / LightGBM 排序 | `scripts/12` |
| 特征工程扩展实验 | 31 维特征宽表 + LightGBM 二分类（独立标签口径） | `scripts/15`、`scripts/16` |
| 在线演示 | Flask API + 离线静态页 | `web/` |

## 目录结构

```
recommender_project/
├── config/
│   └── settings.py            # 全局配置：路径 / 编码 / SEED=42 / 超参统一出口
├── data/
│   ├── raw/
│   │   ├── transactions.csv   # 原始交易记录（541,909 行）
│   │   └── products.csv       # 商品主数据（含类目）
│   └── processed/
│       ├── clean_transactions.csv   # 清洗后交易数据
│       ├── user_sequences.csv       # 用户视角按时间排序的购买序列（长表）
│       └── recall_set.csv           # 多路召回候选集（2,843 测试用户 × 50 候选）
├── scripts/                   # 01–17 全部入口脚本（按编号即建议执行顺序）
├── src/                       # 可复用核心模块（清洗/相似度/向量/召回精排/特征工程）
├── web/                       # 前端：静态相似页 + Flask 在线 API
├── models/                    # 训练产物：Item2Vec 模型/向量索引、LR/LGB 精排模型、元信息
├── outputs/                   # 运行结果存档：报告/指标 json/图/训练记录（详见下文）
└── venv/                      # Python 3.11 虚拟环境（依赖已装好）
```

## 文件与模块说明

### 数据（data/）

| 文件 | 说明 |
| ---- | ---- |
| `data/raw/transactions.csv` | 原始订单明细（含 `CustomerID` 缺失、取消单、负数量等脏数据） |
| `data/raw/products.csv` | 商品主数据，含 `Product_Category` 类目（供特征工程用） |
| `data/processed/clean_transactions.csv` | 清洗结果：0 缺失 / 0 取消 / 0 非正数量 / 0 非正单价，留存率 73.42% |
| `data/processed/user_sequences.csv` | 每行 = 用户在某订单买的一种商品（含单内行序），便于按订单聚合与排序建模 |
| `data/processed/recall_set.csv` | 召回层产物，供精排/在线接口使用 |
| `data/processed/recall_set_5route.csv` | 阶段四 5 路召回融合产物（独立实验，不进入 11/12/在线链路） |

### 核心模块（src/）

| 模块 | 职责 |
| ---- | ---- |
| `data_cleaning.py` | 数据加载 / 原始探查 / 三级清洗 / 验证 |
| `user_similarity.py` | 用户相似度：用户 × 商品稀疏矩阵 + 余弦 Top-K |
| `item_similarity.py` | 商品相似度：矩阵转置口径 + 商品目录 + 批量 Top-K |
| `embedding.py` | Item2Vec：订单篮 → 句子语料、训练回调、商品向量 Top-K 相似 |
| `recommender.py` | 召回/精排共用逻辑：数据切分、特征、在线排序 `score_rank` |
| `feature_engineering.py` | 特征工程扩展实验：切窗、31 维特征、样本构造、验证（scripts/15–16 用） |
| `md_to_docx.py` | Markdown → Word 渲染器（scripts/09、14、17 的底层封装） |

### 脚本（scripts/，按编号顺序执行）

| 脚本 | 作用 | 主要产物 |
| ---- | ---- | ---- |
| `01_clean_data.py` | 加载 → 原始探查 → 三级清洗 → 留存报告 → Top10 验证 | `clean_transactions.csv`、`cleaning_report.txt`、`raw_data_profile.txt`、`top10_verification.csv` |
| `02_generate_eda_report.py` | 关键指标 + 多图 EDA 可视化 | `outputs/eda_report.html` |
| `03_user_similar.py` | **命令行**相似用户推荐（可交互 / 指定用户 / `--top` / `--binary`） | 终端输出 |
| `04_build_similar_web.py` | 离线算好全部相似关系 → 静态页数据 | `web/data/user_sim.js`、`item_sim.js` |
| `04_build_recall.py`（另一套阶段四实验，编号与上行重复） | 5 路召回融合：热门/相似扩展/用户向量/ItemCF/最近会话，按配额 10/10/15/10/5 融合、去重截断至 50，含冷启动分支与配额对照；需先跑 01 与 10 | `recall_set_5route.csv`、`models/user_vectors.pkl`、`outputs/stage04_{coverage,quota_comparison}.txt` |
| `05_evaluate_user_vs_item.py` | UserCF vs ItemCF 离线评测（每用户最后一单 = 测试篮） | 终端对比表 |
| `06_train_item2vec.py` | 训练 CBOW 与 Skip-gram，逐轮损失落盘 | `models/item2vec_{cbow,skip}.model`、`outputs/item2vec/*` |
| `07_build_embed_web.py` | 嵌入口径相似 Top-K → 相似页切换栏数据 | `web/data/embed_sim.js` |
| `08_evaluate_item2vec.py` | 嵌入 vs ItemCF 同协议离线对比 | `outputs/item2vec/eval_vs_itemcf.txt` |
| `09_build_item2vec_docx.py` | 素材 Markdown → Word（报告用，可跳过） | `outputs/...docx` |
| `10_user_sequences_and_index.py` | 用户序列 + 规范模型副本 + 向量索引 | `user_sequences.csv`、`models/item2vec.model`、`models/vector_index/` |
| `11_recall_fusion.py` | 三路召回配额融合（热门:相似:向量 = 10:20:20） | `recall_set.csv`、`outputs/chain/*`（指标/实验存档） |
| `12_rank_lr_lgb.py` | 候选集特征 + LR / LightGBM 精排 + 汇总 | `models/lr_model.pkl`、`lgb_model.pkl`、`rank_meta.json`、`outputs/chain/metrics_summary.json` |
| `13_shot_flask.py` | 自动起 Flask + 无头浏览器截图（需本机浏览器） | `outputs/chain/flask_*.png` |
| `14_build_chain_docx.py` | 素材 Markdown → Word（报告用，可跳过） | `outputs/...docx` |
| `15_build_feature_wide.py` | 特征工程①：31 维全数值特征宽表 + 特征期向量缓存 | `outputs/feature_stage/feature_wide.csv`、`id_map.csv`、`category_map.csv` |
| `16_train_feature_lgb.py` | 特征工程②：LightGBM 训练 + 特征重要性（宽表缺失会自动重跑 15） | `feature_importance.csv/png`、`features_meta.json`、`models/lgb_feature.pkl` |
| `17_build_feature_docx.py` | 素材 Markdown → Word（报告用，可跳过） | `outputs/...docx` |

> 09 / 14 / 17 三个脚本只负责把素材 Markdown 渲染成 Word 文档，属于报告整理用途；不需要 Word 产物时可整体跳过（不影响其他能力）。

### 模型（models/）

| 文件 | 来源 | 用途 |
| ---- | ---- | ---- |
| `item2vec_{cbow,skip}.model`、`item2vec_vectors_*.npz` | `scripts/06` | CBOW / Skip-gram 训练产物 |
| `item2vec.model` + `vector_index/` | `scripts/10` | 规范命名副本 + 行归一化向量索引，供在线 `/similar` |
| `user_vectors.pkl` | `scripts/04_build_recall.py` | 每用户历史商品向量的单位平均向量（阶段四 5 路召回第 3 路用，4,338 用户 × 128 维） |
| `lr_model.pkl` / `lgb_model.pkl` | `scripts/12` | 精排模型（在线 `/recommend`） |
| `lgb_feature.pkl` | `scripts/16` | 特征工程扩展实验的 LightGBM |
| `rank_meta.json` | `scripts/12` | 特征顺序 / AUC / 系数 / 增益 / 案例等元信息（在线接口也读它） |
| `demo_ids.json` | `scripts/12` | 演示用用户 / 商品 ID |

### 网页（web/）

| 文件 | 说明 |
| ---- | ---- |
| `index.html` | 总览入口页 |
| `similar_users.html` | 输入用户 ID → Top-10 相似用户（可链式下钻） |
| `similar_items.html` | 输入商品编号 → Top-10 相似商品（ItemCF / CBOW / Skip-gram 三口径切换） |
| `app.py` | Flask 在线 API（`/health` `/similar` `/recommend` `/baseline`），默认端口 5001 |
| `assets/app.css` | 页面样式（支持明暗主题） |
| `data/{user_sim,item_sim,embed_sim}.js` | 离线预计算的相似关系数据（静态页读取） |

### 运行结果（outputs/）

| 目录/文件 | 内容 |
| ---- | ---- |
| `outputs/eda_report.html` | EDA 可视化报告（浏览器打开） |
| `outputs/cleaning_report.txt` / `raw_data_profile.txt` / `top10_verification.csv` | 清洗与原始数据体检存档 |
| `outputs/chain/` | 召回与精排的数字存档：`recall_phase4.json`（召回层指标）、`metrics_summary.json`（四方法汇总）、`experiment_a.json`（热门 vs ItemCF）、Flask 演示截图 |
| `outputs/item2vec/` | 训练记录：`hyperparams.json`、`train_summary.txt`、逐轮损失、`loss_curve.png`、`eval_vs_itemcf.txt`、网页截图 |
| `outputs/feature_stage/` | 特征工程扩展实验存档：`features_meta.json`（样本/切分/参数/AUC）、`feature_importance.csv/png`、`id_map.csv`、`category_map.csv` |

> outputs 下还散落着按项目阶段整理的过程文档（Word/素材等），仅作记录用途，删除不影响任何脚本运行；本文档不依赖它们。

## 运行方式

### 环境与依赖

仓库自带的 `venv/`（Python 3.11）已装好全部依赖，直接使用即可：

```bash
venv\Scripts\python.exe scripts\01_clean_data.py
```

全新环境安装依赖（仓库根目录不提供 requirements.txt）：

```bash
python -m venv venv
venv\Scripts\python.exe -m pip install -U pip
venv\Scripts\python.exe -m pip install pandas numpy scipy scikit-learn lightgbm gensim \
    flask joblib python-docx matplotlib requests
```

- 训练 / 向量相关需要 `gensim`；精排与在线接口需要 `lightgbm`、`joblib`、`flask`。
- `scripts/13`（自动截图）额外需要本机可被驱动的无头浏览器，可选。
- 编码：`transactions.csv` 为 Latin-1、`products.csv` 为带 BOM 的 UTF-8，已在 `config/settings.py` 配置，无需手工处理。

### 从零完整跑一遍（推荐顺序）

脚本按编号从 01 → 17 即大致依赖顺序；**主线是 `01 → 12`**，13/14 可选，15–17 是独立标签口径的特征工程扩展实验：

```bash
# —— 主线：清洗 → EDA → 相似 → 向量 → 召回 → 精排 ——
venv\Scripts\python.exe scripts\01_clean_data.py
venv\Scripts\python.exe scripts\02_generate_eda_report.py
venv\Scripts\python.exe scripts\04_build_similar_web.py            # 相似商品/用户网页数据
venv\Scripts\python.exe scripts\06_train_item2vec.py               # CBOW & Skip-gram（逐轮损失落盘）
venv\Scripts\python.exe scripts\07_build_embed_web.py              # 嵌入口径网页数据
venv\Scripts\python.exe scripts\08_evaluate_item2vec.py            # 嵌入 vs ItemCF 离线对比
venv\Scripts\python.exe scripts\10_user_sequences_and_index.py     # 用户序列 + 模型副本 + 向量索引
venv\Scripts\python.exe scripts\11_recall_fusion.py                # 多路召回 → recall_set.csv
venv\Scripts\python.exe scripts\12_rank_lr_lgb.py                  # LR/LGB 精排 → 模型 + 指标

# —— 可选 ——
venv\Scripts\python.exe scripts\03_user_similar.py 17850           # 命令行相似用户查询
venv\Scripts\python.exe scripts\05_evaluate_user_vs_item.py        # UserCF vs ItemCF 评测（只打印）
venv\Scripts\python.exe scripts\13_shot_flask.py                   # 起服务 + 截图（需浏览器）
venv\Scripts\python.exe scripts\04_build_recall.py                 # 5 路召回融合（需先有 01、10 的产物；独立实验）

# —— 特征工程扩展实验（独立口径）——
venv\Scripts\python.exe scripts\15_build_feature_wide.py           # 特征宽表（约 2 分钟，产物较大）
venv\Scripts\python.exe scripts\16_train_feature_lgb.py            # LightGBM + 特征重要性
```

### 启动 Web / 在线 API

先完成主线中的 `01`、`10`、`11`、`12`（生成清洗数据、模型、向量索引与召回集），再启动：

```bash
venv\Scripts\python.exe web\app.py        # 打开 http://127.0.0.1:5001/
```

端口可用环境变量覆盖：`PORT=8080`。

| 接口 | 参数 | 说明 |
| ---- | ---- | ---- |
| `/health` | — | 健康检查 |
| `/` | — | 演示主页 |
| `/similar?item_id=85123A&topn=5` | `item_id`、`topn`（默认 5）、`method`（`item2vec` 默认 / `itemcf`） | 商品相似 Top-N |
| `/recommend?user_id=12597&topn=10` | `user_id`、`topn`（默认 10）、`ranker`（`lr` 默认 / `lgb`） | 召回 50 → LR/LGB 精排全链路 |
| `/baseline?user_id=12597&topn=10` | `user_id`、`topn`（默认 10） | 热门 / ItemCF / 融合 三路对照 |

任一接口追加 `&format=json` 直接返回 JSON，便于联调：

```bash
curl "http://127.0.0.1:5001/recommend?user_id=12597&topn=10&ranker=lgb&format=json"
```

### 离线静态网页（无需服务）

`web/` 下的页面在 `web/data/*.js` 已生成的前提下可**双击离线打开**：

- `web/index.html`（入口）→ `similar_users.html` / `similar_items.html`
- 相似商品页右上角可在 **ItemCF / CBOW 嵌入 / Skip-gram 嵌入** 三口径间切换

## 评估口径与结果速览

统一留出协议：**每个用户「最后一笔订单」= 测试篮**，更早历史 = 训练；特征与向量只用训练期计算（无测试篮泄漏）。特征工程扩展实验另用「特征期 < 2011-11-09、尾随 30 天为标签窗」的时间切分口径，两者勿跨表比较。

| 环节 | 指标 | 数值 |
| ---- | ---- | ---- |
| 数据清洗 | 原始 541,909 行 → 清洗后 397,884 行 | 留存 73.42% |
| 多路召回（2,843 测试用户） | 召回集商品覆盖率 | 77.9% |
| | 50 候选内至少命中下一单 1 件 | 40.2% |
| | recall@50 | 0.044（测试篮平均约 20 种商品） |
| 精排（valid 853 用户） | LightGBM Hit@10 / AUC | **0.263** / 0.757 |
| | LR Hit@10 / AUC | 0.252 / 0.766 |
| | 对照：全局热门 Hit@10 / 融合直排 | 0.203 / 0.147 |
| 特征工程扩展实验 | LightGBM valid AUC（31 维特征，配对 1:3） | **0.8544** |

要点：融合召回的价值在于**候选池的覆盖与多路供给**，直接按融合分取 Top10 弱于热门榜；经 LR / LightGBM 精排重排后 Hit@10 较热门基线提升约 **+24% ~ +30%**，Precision@5 提升约 70%。更多中间对比（UserCF/ItemCF/嵌入、热门基线、逐轮损失）见 `outputs/item2vec/eval_vs_itemcf.txt`、`outputs/chain/experiment_a.json` 与 `outputs/chain/metrics_summary.json`。

## 复现说明

- `config/settings.py` 固定 `SEED=42`，全流程可复现。
- 所有脚本幂等、结果落盘 `outputs/`，重复运行不会积累中间状态。
- `scripts/15` 的宽表与特征期向量缓存较大（约 39 MB）；如已删除，`scripts/16` 检测到宽表缺失会自动重跑 15 重建。

---

## 候选集打分线（cand-align · 2026-09 新增）

课程第二阶段加入的一条**独立交付线**（`scripts/18` 起，与上面的答辩演示线并行）：
老师给定固定候选集 `(user_id, item_id)`（58,205 对），逐行打分输出 `user_id,item_id,score`，
按 `综合分 = 0.40·GAUC + 0.40·NDCG@10 + 0.20·Recall@10`（逐用户组、等权 macro）评分。

| 里程碑 | 数值 | 说明 |
| --- | --- | --- |
| 纯 own-model 冠军 **v6 = cand-align-v6-fuse** | **0.7782** | LGB×0.1 + XGB×0.3 + LR(近2月×spw)×0.6 全局百分位秩融合；44 特征、组等权训练、cut=2010-11-01 |
| 跨架构融合 v13-v15blend65 | **0.7863**（板最佳） | 0.65×rank(组长共享 Nov 提交)+0.35×rank(v6)；需组长文件方可复现 |
| 结构赌注 / 移植 / 归一化轴等已出清 | — | 详见下方文档，均留有诊断证据 |

关键代码与文档（均在仓库内）：

- 代码：`src/candidate*.py`（特征/数据/评测）、`scripts/47→48→49`（v6 组等权 OOF→选权→出提交）、
  `scripts/73/78/79`（跨架构秩融合）、`scripts/59`（老师复合口径精确实现）、`scripts/74`（OOF 复合重算）。
- 文档：`outputs/candidate/提交版本记录.md`（提交史权威）、`outputs/candidate/structural_bets_log.md`
  （结构赌注复盘）、`outputs/candidate/归档_v6对比组长v9_全探针结论.md`、
  `outputs/candidate/v6生成链全档案.md` + `v6的生成过程/`（v6 生成链自包含归档）、
  `outputs/candidate/与组长对比_cand-align-v13-v15blend65_0.7863.md`（写给组长的融合说明）。
- 数据对齐、评分口径、硬约束的完整交接说明见 `data/云端跑模型交接清单.md`（data 不入仓，本地可看）。

> 硬约束：SEED=42、BLAS 锁 1 线程、全部产物新文件、训练永不丢最近月；回放 OOF 复合与
> 真实未来月（Nov）已证反向，故 OOF 不当 Nov 选权门禁——完整纪律见交接清单与各 log。
