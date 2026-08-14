import os

import cv2
import numpy as np

from ..images.predict import predict_image, summarize_result
from ..images.persist import write_json


def run_lime_image(cfg, model, image_paths, out_dir,
                   num_samples=1000, num_features=10, min_conf=0.25):
    """LIME Image sobre el modelo de detección.

    El clasificador para LIME es una agregación de las confianzas por clase de
    las detecciones. Guarda la imagen con superpíxeles resaltados + pesos JSON.
    """
    from lime.lime_image import LimeImageExplainer

    os.makedirs(out_dir, exist_ok=True)
    explainer = LimeImageExplainer(random_state=42, verbose=False)

    records = []
    artifacts = []
    names = model.names

    for img_path in image_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        def classifier_fn(images):
            # images: (N, H, W, 3) float/uint8
            out = np.zeros((len(images), len(names)), dtype=np.float32)
            for i, im in enumerate(images):
                im = np.clip(im, 0, 255).astype(np.uint8)
                if im.shape[2] == 4:
                    im = im[:, :, :3]
                ibgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                tmp = os.path.join(out_dir, "_lime_samp.jpg")
                cv2.imwrite(tmp, ibgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                dets = summarize_result(
                    predict_image(cfg, model, tmp, conf=0.01, iou=0.5))["detections"]
                for d in dets:
                    out[i, d["class"]] += d["confidence"]
            total = out.sum(axis=1, keepdims=True)
            total[total == 0] = 1.0
            return out / total

        explanation = explainer.explain_instance(
            rgb, classifier_fn, top_labels=1,
            hide_color=0, num_samples=num_samples)

        top_label = explanation.top_labels[0]
        images_and_masks = explanation.get_image_and_mask(
            top_label, positive_only=True, num_features=num_features, hide_rest=False)

        masked_img, mask = images_and_masks  # masked_img RGB, mask marca píxeles relevantes
        stem = os.path.splitext(os.path.basename(str(img_path)))[0]
        out_bgr = cv2.cvtColor(np.asarray(masked_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        png_path = os.path.join(out_dir, f"lime_image_{stem}.png")
        cv2.imwrite(png_path, out_bgr)

        wts = explanation.local_exp[top_label]
        artifacts.append(png_path)
        records.append({
            "image": str(img_path),
            "top_label": int(top_label),
            "top_label_name": names[int(top_label)] if int(top_label) < len(names) else str(top_label),
            "png": os.path.basename(png_path),
            "top_pixels_weights": [
                {"superpixel": int(s), "weight": round(float(w), 5)}
                for s, w in sorted(wts, key=lambda t: -abs(t[1]))[:num_features]
            ],
        })

    if not records:
        raise ValueError("LIME no generó explicaciones (verifica imágenes/conf).")

    write_json(os.path.join(out_dir, "lime_image.json"),
               {"num_samples": num_samples, "instances": records})
    return {"saved": True, "artifacts": artifacts, "instances": records}