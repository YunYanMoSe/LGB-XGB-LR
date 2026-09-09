# cand-align-v6-fuse 生成链全档案

> 纯 own-model 冠军(v6 = cand-align-v6-fuse = **榜 0.7782**)的完整生成档案。
> 涵盖:任务口径 → 数据原料 → 回放训练表 → 组等权突破 → 融合选权 → 出提交 → 全部相关文件 → 复现命令。
> 本档案对应的整理副本在 `v6的生成过程/`(产物 / 选权证据 / 生成脚本 三类分放)。

---

## 0. 一句话配方

```
score = 0.1×rank(LGB)/N + 0.3×rank(XGB)/N + 0.6×rank(LR)/N    (全局百分位秩融合)
        LGB/XGB: 组等权、2010-06..10 全月训练
        LR    : 组等权 ×spw、只训最近 2 月(2010-09+10)
44 特征, 打分 cut = 2010-11-01, 覆盖 58,205 候选行
```

本地 OOF(Jul..Oct) gAUC_m **.8721** → 真榜 **0.7782**。

---

## 1. 任务口径(为什么这么选)

- **交付**:对固定候选集 `train_data/sample_candidates.csv`(58,205 行)逐行打分,输出 `user_id,item_id,score`,全行覆盖。
- **老师评分**:`综合分 = 0.40×GAUC + 0.40×NDCG@10 + 0.20×Recall@10`,逐 (user,month) 组、等权 macro。
- **唯一可信本地选型口径** = **gAUC_m**(逐 (user,month) 组 AUC 宏平均);池化 AUC 选权在真榜不迁移(v7-pooled 0.7774 < v6 已证)。
- v6 是**纯 own-model** 交付(不含任何组长共享文件信号);之后所有 v13 跨架构融合(blend65=0.7863)都是拿**这份 v6 提交当输入源 B**,未再重训。

---

## 2. 数据原料

| 文件 | 内容 |
|---|---|
| `data/processed/leader_clean.csv` | 清洗后交易 327,827 行,唯一数据源 |
| `train_data/sample_candidates.csv` | 58,205 候选对(user_id int / item_id str) |
| `outputs/candidate/replay/wide_v2_2010-{06..10}.csv` | **44 特征回放宽表**(每月一份,行=当月候选+当月 label) |

44 特征 = `cand.FEATURES_CAND`(33 base)+ `src/candidate_v2.V2_FEATS`(11 v2),顺序见 `candidate_meta_cand-align-v6-fuse.json` 的 `feature_order`,训练/打分同序。

---

## 3. 生成链(三段脚本)

### 第 1 段 · 回放训练表(已生成,一般不用重跑)
`scripts/29`(基础 33 列 wide)→ `scripts/32_build_replay_v2.py`(追加 11 v2 特征)。
产物:wide_v2 表 **225,735 行 / 4.37% 真实基率**,训练数据源。

### 第 2 段 · 组等权突破 OOF(`scripts/47`)
- 把训练行权重从 `1/√组大小` 改成 **`1/组大小`**(组=(user,month),每组对损失等贡献,对齐 gAUC_m 口径)。
- 重训 eq 三列(只写新列,其余从已存 oof 读):
  - `lgb_macro_eq`: LGB 全月组等权
  - `xgb_macro_eq`: XGB 全月组等权
  - `lr_eq_l2_sp`: LR 最近2月 组等权 × scale_pos(spw)
- 产物: `oof_eqg_preds.csv`(187,730 行 Jul..Oct,带 label)+ `eqgroup_report.txt`。
- 效果:XGB .8586→.8665、LR .8684→.8687;eq 三分量融合 **.8721** > v5 .8710。

### 第 3 段 · 融合选权(`scripts/48`)
- 在 oof_eqg_preds 上对候选三分量组合做 **step .025 穷举**,按 gAUC_m 排序。
- 冠军 **eq三分量 (0.1, 0.3, 0.6) = gAUC_m .8721**,且细网格邻域 .8716–.8721 平台稳健(非单点运气)。
- 产物: `fusion_v6_report.txt` + `fusion_v6_weights.json`(定稿权重)。

### 第 4 段 · 出提交(`scripts/49_build_submission_v6.py`)——独立重训,非复用 OOF 模型
1. LGB(组等权,06..10 全月)→ 概率分;
2. XGB(组等权,全月)→ 概率分;
3. LR(组等权×spw,**只训 2010-09+10 最近 2 月**)→ 概率分;
4. 打分:`cut=2010-11-01`,用全量 leader_clean,对 58,205 候选对算 44 特征;
5. 融合:`score = 0.1×rank(s_lgb)/N + 0.3×rank(s_xgb)/N + 0.6×rank(s_lr)/N`;
6. 写出提交 + 3 个 pkl + meta。

---

## 4. 关键设计点(为什么 v6 是纯 own-model 顶)

1. **eq 组等权**(v5→v6 突破):训练目标对齐老师逐组等权口径;
2. **LR 吃近 2 月、树吃全月**:相反窗口偏好互偿(v4→v6 反复实证的结构;删最近月 ⇒ 榜 −.008 灾难);
3. **全局百分位秩融合**:三模型量纲不同,rank 作单调保序的量纲对齐,per-(user,month) 组内序不受影响;
4. **44 特征、只用 `< cut` 历史、无泄漏**;
5. **确定性**:SEED=42 + BLAS 锁 1 线程 + 排序建词表 ⇒ 可跨进程逐位复现([[determinism-gotcha]])。

---

## 5. 相关文件总表

### 产物(最终交付)
| 文件 | 作用 |
|---|---|
| `outputs/candidate/sample_submission_cand-align-v6-fuse.csv` | 最终提交(58,205×3;榜 0.7782;md5 `bd95c79d420603869d257239e068fc54`) |
| `models/candidate_cand-align-v6-fuse_lgb.pkl` | LGB 分量模型(组等权全月) |
| `models/candidate_cand-align-v6-fuse_xgb.pkl` | XGB 分量模型(组等权全月) |
| `models/candidate_cand-align-v6-fuse_lr.pkl` | LR 分量模型(组等权×spw 近2月) |
| `models/candidate_meta_cand-align-v6-fuse.json` | 配方:权重 {.1,.3,.6}、44 特征序、cut=2010-11-01 |

### 选权证据(OOF 阶段产物)
| 文件 | 作用 |
|---|---|
| `outputs/candidate/replay/oof_eqg_preds.csv` | v6 前向 OOF(187,730 行 Jul..Oct,带 label,含 eq 三分量列) |
| `outputs/candidate/replay/eqgroup_report.txt` | 组等权 OOF 对照(三分量 .8721 最优) |
| `outputs/candidate/replay/fusion_v6_report.txt` | step .025 细网格确认冠军稳健 |
| `outputs/candidate/replay/fusion_v6_weights.json` | 定稿权重 {lgb_macro_eq:.1, xgb_macro_eq:.3, lr_eq_l2_sp:.6} |

### 生成脚本
| 脚本 | 阶段 |
|---|---|
| `scripts/47_eqgroup_weight_oof.py` | 组等权重训 eq 三列 → oof_eqg_preds |
| `scripts/48_finalize_v6_oof.py` | 细网格挑冠军权 → fusion_v6_weights.json |
| `scripts/49_build_submission_v6.py` | 重训三模型 + cut=11-01 打分 + 融合 → 提交/pkl/meta |

---

## 6. 复现命令

```bash
# 从 OOF 选权到提交(wide_v2 表已存在,不用重跑 29/32)
venv\Scripts\python.exe scripts\47_eqgroup_weight_oof.py   # 出 oof_eqg_preds(约数分钟)
venv\Scripts\python.exe scripts\48_finalize_v6_oof.py      # 出 fusion_v6_weights.json
venv\Scripts\python.exe scripts\49_build_submission_v6.py  # 重训 + 打分 + 融合 → 提交(约 90s+)
```

复现一致性:SEED=42 + BLAS 1 线程(脚本顶部 `os.environ.setdefault(...,"1")`);重跑产物与盘上文件应逐位一致。

---

## 7. 档案边界

- 本档案只覆盖 **v6 自己的生成**(纯 own-model)。跨架构融合线(v13-v15blend65=0.7863)见 `outputs/candidate/与组长对比_cand-align-v13-v15blend65_0.7863.md` 与 `outputs/candidate/提交版本记录.md`。
- 硬约束:原件永不覆盖、全部新文件;scripts/47/48/49 是唯一改动口径的脚本,勿再改。
