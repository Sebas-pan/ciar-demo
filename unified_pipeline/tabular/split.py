from sklearn.model_selection import train_test_split

DEFAULT_RATIOS = (0.7, 0.15, 0.15)


def split_by_task(df, target, task, split_cfg):
    ratios = list(split_cfg.get("train_val_test") or DEFAULT_RATIOS)
    if len(ratios) != 3:
        raise ValueError(
            "train_val_test must have exactly 3 ratios, got {}".format(len(ratios)))
    total = sum(ratios)
    if total <= 0 or abs(total - 1.0) > 1e-6:
        raise ValueError(
            "train_val_test ratios must sum to 1.0, got {}".format(total))
    rs = split_cfg.get("random_state", 42)
    stratify = bool(split_cfg.get("stratify", True))
    train_ratio, val_ratio, test_ratio = ratios

    y = df[target]
    X = df.drop(columns=[target])

    stratify_y = y if (task == "classification" and stratify) else None
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=1 - train_ratio,
        random_state=rs,
        stratify=stratify_y,
    )
    stratify_tmp = y_tmp if (task == "classification" and stratify) else None
    if val_ratio <= 0:
        X_val, X_test, y_val, y_test = X_tmp[0:0], X_tmp, y_tmp[0:0], y_tmp
    else:
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=rs,
            stratify=stratify_tmp,
        )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }