import os

from ..io import write_json
from .narrator import narrate_image


def build_image_report(run_dir, metrics, names, variant, device,
                       train_minutes, predictions, xai, xai_dir, eda, model_meta,
                       narrative=True):
    """Genera report.md de la rama imagen."""
    lines = ["# Reporte — Detección YOLO11", ""]

    lines.append("## Entrenamiento")
    lines.append("- Modelo: `{}` (variante `{}`)".format(model_meta.get("model_type"), variant))
    lines.append("- Dispositivo: `{}` · tiempo: {} min".format(device, train_minutes))
    lines.append("- Clases: {}".format(", ".join(names)))
    lines.append("- EDA: `class_distribution.png`, `sample_grid.jpg` y "
                 "`eda_image.json` (distribución por split, stats de tamaño)")
    lines.append("")

    lines.append("## Métricas (test)")
    lines.append("- mAP50: **{:.4f}**".format(metrics["mAP50"]))
    lines.append("- mAP50-95: **{:.4f}**".format(metrics["mAP50-95"]))
    lines.append("- Precision (mean): {:.4f}".format(metrics["precision"]))
    lines.append("- Recall (mean): {:.4f}".format(metrics["recall"]))
    for cls, vals in metrics["per_class"].items():
        lines.append("    - {} · AP50={} · AP50-95={}".format(
            cls,
            "{:.4f}".format(vals["AP50"]) if vals.get("AP50") else "n/a",
            "{:.4f}".format(vals["AP50-95"]) if vals.get("AP50-95") else "n/a",
        ))
    lines.append("")

    if narrative:
        text = narrate_image(metrics, names, predictions, xai)
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
                lines.append("- `{}` (clase objetivo: {})".format(
                    inst["png"],
                    inst.get("class_name") or inst.get("top_label_name")))
        lines.append("")

    lines.append("## Cómo interpretar los mapas")
    lines.append("""- **Grad-CAM**: mapa de calor sobre la última conv de la rama de clases.
    Las zonas rojas muestran qué regiones **soportan** la clase detectada.
- **Occlusion Sensitivity**: celda en gris → caída de confianza de la caja.
    Celdas rojas = la confianza Depende de esa zona.
- **LIME Image**: superpíxeles que, al ocultarse, aumentaron la clase objetivo
    (verde) o la disminuyeron (rojo). Se agrega la confianza por clase como
    "probabilidad" para el clasificador LIME.
""")
    lines.append("Artículo de referencia: métricas en `metrics.json`, "
                 "predicciones en `predictions/predictions.json`, "
                 "detalle XAI en `xai/*.json` y mapas `.npy`.")

    write_json(os.path.join(run_dir, "model", "metadata.json"), model_meta)
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return os.path.join(run_dir, "report.md")