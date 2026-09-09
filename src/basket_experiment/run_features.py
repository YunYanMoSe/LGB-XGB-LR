"""按相似度方案在全部月份上算篮特征，返回合并表（键 ym,user_id,item_id + 特征）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import io_data as io
from . import features as F
from .eval import ANCHOR_COL

RESID_COLS = ["sim_last1", "sim_last3_max", "sim_last3_mean",
              "simnb_last1", "simnb_last3_max", "simnb_last3_mean", "nb_item_baskets"]


def features_for_scheme(store: F.BasketStore, n_items: int, oof: pd.DataFrame,
                        mapping: dict, scheme: str,
                        months=io.YM_ALL) -> pd.DataFrame:
    """给定 scheme，为 months 各月算特征；行 = 各月 anchor OOF 行 + RESID_COLS。"""
    parts = []
    for ym in months:
        sub = oof[oof["ym"] == ym]
        cut = io.cutoff_of(ym)
        S = F.build_sim_matrix(store, n_items, cut, scheme=scheme)
        baskets = F.all_baskets_per_user(store, sub["user_id"].to_numpy(), cut)
        feat = F.month_features(S, sub, mapping, baskets)
        base = sub[["ym", "user_id", "item_id", "label", "prior_bought", ANCHOR_COL]] \
            .reset_index(drop=True)
        parts.append(pd.concat([base, feat.reset_index(drop=True)], axis=1))
    return pd.concat(parts, ignore_index=True)
