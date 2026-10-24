import pandas as pd
from sklearn.metrics import roc_auc_score


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    for col in (target, id_col, time_col):
        if col not in df.columns:
            raise KeyError(col)

    def missing_mask(series):
        mask = series.isna()
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            mask = mask | series.map(
                lambda x: isinstance(x, str) and not x.strip()
            )
        return mask

    missing_masks = {col: missing_mask(df[col]) for col in df.columns}

    missing = {
        col: float(missing_masks[col].mean()) if len(df) else 0.0
        for col in df.columns
    }

    ids = df.loc[~missing_masks[id_col], id_col]
    duplicate_ids = sorted(
        ids[ids.duplicated(keep=False)].unique().tolist(),
        key=str,
    )

    duplicate_rows = int(df.duplicated(keep="first").sum())

    constant_columns = sorted(
        col for col in df.columns
        if col != target
        and df.loc[~missing_masks[col], col].nunique() <= 1
    )

    unlabelled_rows = int(missing_masks[target].sum())
    labelled = ~missing_masks[target]
    y = df.loc[labelled, target]
    positive_rate = float((y == 1).mean()) if len(y) else 0.0

    timestamps = pd.to_datetime(df[time_col], errors="coerce")
    bad_timestamps = int(
        ((~missing_masks[time_col]) & timestamps.isna()).sum()
    )

    valid_times = timestamps[~timestamps.isna()]
    time_range = (
        [valid_times.min().isoformat(), valid_times.max().isoformat()]
        if len(valid_times)
        else [None, None]
    )

    leakage_suspects = []

    if len(y) >= 20 and y.nunique() == 2:
        for col in df.columns:
            if col in (target, id_col, time_col):
                continue

            feature = df.loc[labelled, col]
            feature_missing = missing_masks[col].loc[labelled]

            values = feature.astype(object).where(
                ~feature_missing, "__MISSING__"
            )
            if values.nunique(dropna=False) <= 1:
                continue

            # Test whether the missingness pattern predicts the label.
            indicator = feature_missing.astype(int)
            if indicator.nunique() > 1:
                try:
                    auc = roc_auc_score(y, indicator)
                    if auc >= 0.98 or auc <= 0.02:
                        leakage_suspects.append(col)
                        continue
                except ValueError:
                    pass

            # Test numeric and boolean values using non-missing rows only.
            if (
                pd.api.types.is_numeric_dtype(df[col])
                or pd.api.types.is_bool_dtype(df[col])
            ):
                usable = ~feature_missing
                if int(usable.sum()) >= 20:
                    feature_y = y.loc[usable[usable].index]
                    feature_values = feature.loc[usable]

                    if feature_y.nunique() == 2:
                        try:
                            auc = roc_auc_score(feature_y, feature_values)
                            if auc >= 0.98 or auc <= 0.02:
                                leakage_suspects.append(col)
                        except (ValueError, TypeError):
                            pass

    return {
        "n_rows": int(len(df)),
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": sorted(set(leakage_suspects)),
    }
