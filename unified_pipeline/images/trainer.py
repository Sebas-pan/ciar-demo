import os


def resolve_device(cfg):
    """Devuelve str/número adecuado para ultralytics a partir de cfg['device']."""
    import torch

    device = cfg["model"].get("device", "auto")
    if isinstance(device, str):
        device = device.strip().lower()
    if device in ("auto", "gpu", "best"):
        return "0" if torch.cuda.is_available() else "cpu"
    return device


def run_training(cfg, model, data_spec, run_dir):
    """Entrena modelo YOLO11 y retorna la ruta al mejor checkpoint."""
    train_cfg = cfg["model"].get("train", {})
    epochs = int(train_cfg.get("epochs", 30))
    imgsz = int(train_cfg.get("imgsz", 640))
    batch = train_cfg.get("batch", 16)
    device = resolve_device(cfg)
    workers = int(train_cfg.get("workers", 2))
    optimizer = train_cfg.get("optimizer", "auto")
    lr = train_cfg.get("lr")
    patience = int(train_cfg.get("patience", 100))
    augment = train_cfg.get("augment", True)
    seed = train_cfg.get("seed", 42)

    train_args = dict(
        data=data_spec,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        optimizer=optimizer,
        patience=patience,
        seed=seed,
        project=run_dir,
        name="train",
        exist_ok=True,
        verbose=True,
    )
    if augment is False:
        train_args.pop("augment", None)
        train_args["hsv_h"] = 0.0
        train_args["hsv_s"] = 0.0
        train_args["hsv_v"] = 0.0
        train_args["degrees"] = 0
        train_args["translate"] = 0
        train_args["scale"] = 0
        train_args["fliplr"] = 0.0
        train_args["mosaic"] = 0.0
    if lr is not None:
        train_args["lr0"] = float(lr)

    model.train(**train_args)

    best_path = os.path.join(run_dir, "train", "weights", "best.pt")
    if not os.path.exists(best_path):
        # fallback: buscar en cualquier subcarpeta train*
        import glob

        cands = sorted(glob.glob(os.path.join(run_dir, "train*", "weights", "best.pt")))
        if cands:
            best_path = cands[0]
        else:
            raise FileNotFoundError("Could not find best.pt after training")

    return best_path