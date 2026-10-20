import os

from ..io import write_json
from .narrator import narrate_image
from .llm_narrator import narrate_image_detection, narrate_image_classification


def build_image_report(run_dir, metrics, names, variant, device,
                       train_minutes, predictions, xai, xai_dir, eda, model_meta,
                       narrative=True, narrate_image_func=None):
    """Genera report.md de la rama imagen."""
    model_type = model_meta.get("model_type", "yolo")
    lines = ["# Reporte — Detección {}".format(model_type.upper()), ""]

    lines.append("## Entrenamiento")
    lines.append("- Modelo: `{}` (variante `{}`)".format(model_meta.get("model_type"), variant))
    lines.append("- Dispositivo: `{}` · tiempo: {} min".format(device, train_minutes))
    lines.append("- Clases: {}".format(", ".join(names)))
    lines.append("- EDA: `class_distribution.png`, `sample_grid.jpg` y "
                 "`eda_image.json` (distribución por split, stats de tamaño)")
    lines.append("")

    lines.append("## Métricas (test)")
    lines.append("- mAP50: **{:.4f}**".format(metrics.get("mAP50", float("nan"))))
    lines.append("- mAP50-95: **{:.4f}**".format(metrics.get("mAP50-95", float("nan"))))
    lines.append("- Precision (mean): {:.4f}".format(metrics.get("precision", float("nan"))))
    lines.append("- Recall (mean): {:.4f}".format(metrics.get("recall", float("nan"))))
    for cls, vals in (metrics.get("per_class") or {}).items():
        vals = vals or {}
        lines.append("    - {} · AP50={} · AP50-95={}".format(
            cls,
            "{:.4f}".format(vals["AP50"]) if vals.get("AP50") is not None else "n/a",
            "{:.4f}".format(vals["AP50-95"]) if vals.get("AP50-95") is not None else "n/a",
        ))
    lines.append("")

    def _is_classification_model(model_meta, metrics, prediction):
        task = str((model_meta or {}).get("task", "")).lower()
        if task == "classification":
            return True
        # elif
        if task == "detection":
            return False
        if isinstance(metrics, dict):
            if "accuracy" in metrics: return True
            if "mAP50" in metrics: return True
        if predictions: 
            rec = predicttions[0] or {}
            if "pred_name" in rec or "pred_class" in rec: return True
            if "detections" in rec: return False
        return False

        
    

    if narrative:
        # narrate = narrate_image_func or narrate_image
        if _is_classification_model(model_meta, metrics, predictions):
            narrate = narrate_image_classification
        else:
            narrate = narrate_image_detection
        text = narrate(metrics, names, predictions, xai)
        if text:
            lines.append(text)
            lines.append("")

    lines.append("## Predicciones de ejemplo (`predictions/`)")
    for rec in predictions:
        lines.append("- `{}` → {} detecciones".format(
            os.path.basename(rec.get("image") or ""), len(rec["detections"])))
        for d in rec["detections"]:
            lines.append("    - {} ({:.2f}) bbox {}".format(
                d["class_name"], d["confidence"], d["bbox_xyxy"]))
    lines.append("")

    lines.append("## Explicabilidad (`xai/`)")
    lines.append("")
    for method, entry in [("gradcam", xai.get("gradcam")),
                          ("occlusion", xai.get("occlusion")),
                          ("lime_image", xai.get("lime_image"))]:
        lines.append("### {}".format(method.upper()))
        if entry is None:
            lines.append("No ejecutado.")
        elif "error" in entry:
            lines.append("**ERROR**: {}".format(entry["error"]))
        else:
            lines.append("Artefactos:")
            for inst in entry.get("instances", []):
                target = inst.get("class_name") or inst.get("top_label_name")
                if target is None:
                    lines.append("- `{}`".format(inst.get("png", "?")))
                else:
                    lines.append("- `{}` (clase objetivo: {})".format(
                        inst.get("png", "?"), target))
        lines.append("")

    lines.append("## Cómo interpretar los mapas")
    lines.append("""- **Grad-CAM**: mapa de calor sobre la última conv de la rama de clases.
    Las zonas rojas muestran qué regiones **soportan** la clase detectada.
- **Occlusion Sensitivity**: celda en gris → caída de confianza de la caja.
    Celdas rojas = la confianza Depende de esa zona.
- **LIME Image**: superpíxeles que más apoyan la clase objetivo
    (positive_only=True: solo se resaltan contribuciones positivas).
    Se agrega la confianza por clase como
    "probabilidad" para el clasificador LIME.
""")
    lines.append("Artículo de referencia: métricas en `metrics.json`, "
                 "predicciones en `predictions/predictions.json`, "
                 "detalle XAI en `xai/*.json` y mapas `.npy`.")

    write_json(os.path.join(run_dir, "model", "metadata.json"), model_meta)
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return os.path.join(run_dir, "report.md")