"""Unsupervised split — no target, no stratify."""
from sklearn.model_selection import train_test_split

DEFAULT_RATIOS = (0.85, 0.0, 0.15)


def split_unsupervised(df, split_cfg):
    ratios = list(split_cfg.get("train_val_test") or DEFAULT_RATIOS)
    if len(ratios) != 3:
        raise ValueError("train_val_test must have exactly 3 ratios, got {}".format(len(ratios)))
    total = sum(ratios)
    if total <= 0 or abs(total - 1.0) > 1e-6:
        raise ValueError("train_val_test ratios must sum to 1.0, got {}".format(total))
    rs = split_cfg.get("random_state", 42)
    train_ratio, val_ratio, test_ratio = ratios

    # No y, just X
    if train_ratio >= 1.0:
        return {"X_train": df, "X_val": df.iloc[0:0], "X_test": df.iloc[0:0]}

    X_train, X_tmp = train_test_split(df, test_size=1 - train_ratio, random_state=rs)
    if val_ratio <= 0:
        X_val = X_tmp.iloc[0:0]
        X_test = X_tmp
    else:
        # Split remaining into val/test
        X_val, X_test = train_test_split(
            X_tmp, test_size=test_ratio / (val_ratio + test_ratio), random_state=rs
        )
    return {"X_train": X_train, "X_val": X_val, "X_test": X_test}
