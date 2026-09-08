"""Generadores de narrativa en lenguaje natural para los reportes XAI.

Plantillas deterministas (sin LLM) que traducen los datos de explicabilidad
ya calculados a texto legible: quién es el modelo, cuáles son las
características que más influyen y por qué predijo lo que predijo.
"""


def narrate_model(metrics, task, model_type, target=None):
    lines = []
    target_txt = "sobre `{}`".format(target) if target else "sobre los datos"
    lines.append(
        "Este modelo es un **{}** entrenado para **{}** {}.".format(
            model_type, task, target_txt))

    if task == "classification":
        acc = metrics.get("accuracy")
        f1 = metrics.get("f1")
        if acc is not None:
            lines.append(
                "En el conjunto de test alcanzó una **exactitud (accuracy) de "
                "{:.1f}%**".format(float(acc) * 100))
        if f1 is not None:
            lines.append(
                "y un **F1 global de {:.3f}** (la media ponderada de precisión "
                "y recall entre las clases)".format(float(f1)))
        if acc is not None or f1 is not None:
            lines[-1] += "."
    else:
        r2 = metrics.get("r2")
        rmse = metrics.get("rmse")
        mae = metrics.get("mae")
        if r2 is not None:
            lines.append(
                "En el conjunto de test alcanzó un **R² de {:.3f}**, lo que "
                "significa que el modelo explica el {:.1f}% de la variabilidad "
                "del objetivo.".format(float(r2), float(r2) * 100))
        if rmse is not None:
            lines.append(
                "El error típico de sus predicciones (RMSE) es de "
                "**{:.2f}**.".format(float(rmse)))
        if mae is not None and rmse is None:
            lines.append(
                "El error medio absoluto (MAE) es de **{:.2f}**.".format(
                    float(mae)))

    return " ".join(lines)


def narrate_top_features(feature_importance=None, shap_global=None, top_n=15):
    """Lista numerada de las top_n características más influyentes."""
    source = None
    if shap_global:
        source = [{"feature": r.get("feature"), "value": r.get("mean_abs_shap")}
                  for r in shap_global if r.get("feature")]
        metric_name = "contribución media (mean |SHAP|)"
    elif feature_importance:
        source = [{"feature": r.get("feature"), "value": r.get("importance")}
                  for r in feature_importance if r.get("feature")]
        metric_name = "importancia global"

    if not source:
        return ""

    source = sorted(source, key=lambda r: -(r["value"] or 0.0))[:top_n]

    lines = [
        "## Estas son las {} características más importantes".format(len(source)),
        "",
        "Ordenadas de mayor a menor influencia, medida como {}:".format(metric_name),
        "",
    ]
    for i, rec in enumerate(source, start=1):
        emphasis = ""
        if i == 1:
            emphasis = " — la **característica dominante**; por sí sola marca "
            emphasis += "la mayor parte de la decisión del modelo."
        elif i <= 3:
            emphasis = " — entre las tres de mayor peso en las predicciones."
        lines.append(
            "{}. **{}** ({} = {:.3f}){}".format(
                i, rec["feature"], metric_name, rec["value"], emphasis))

    top1 = source[0]["feature"]
    lines.append("")
    lines.append(
        "En resumen: si solo hubiera que mirar una variable, sería "
        "**{}**. El resto del ranking ayuda a entender qué otras señales "
        "complementan esa decisión.".format(top1))
    return "\n".join(lines)


def _fmt_value(v):
    a = abs(v)
    if a >= 1000:
        return "{:,.0f}".format(v)
    if a >= 10:
        return "{:,.1f}".format(v)
    if a >= 1:
        return "{:,.2f}".format(v)
    return "{:.4f}".format(v)


def narrate_instance(rec, task):
    """Párrafo por instancia explicando por qué el modelo predijo eso."""
    true_val = rec.get("true")
    pred_val = rec.get("prediction")
    lines = []

    if task == "regression" and true_val is not None and pred_val is not None:
        diff = float(pred_val) - float(true_val)
        if float(true_val) != 0:
            pct = abs(diff) / abs(float(true_val)) * 100
        else:
            pct = 0.0
        direc = "por encima" if diff > 0 else "por debajo"
        lines.append(
            "El valor real era **{:.2f}** y el modelo predijo **{:.2f}** "
            "(un error de {:.2f}, ≈{:.1f}% {} del valor real).".format(
                float(true_val), float(pred_val), abs(diff), pct, direc))
    elif task == "classification":
        lines.append(
            "El modelo predijo la clase **{}**.".format(pred_val))
        probs = rec.get("probabilities")
        if probs:
            best_cls = pred_val
            best_prob = probs.get(str(best_cls))
            if best_prob is not None:
                lines.append(
                    "Asignó una probabilidad del **{:.1f}%** a esa clase.".format(
                        float(best_prob) * 100))
    else:
        lines.append(
            "El modelo predijo un valor de **{}** (real: {}).".format(
                pred_val, true_val))

    up = []
    down = []
    for feat, val in (rec.get("top_shap") or []):
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        (up if v >= 0 else down).append((feat, v))
    if not up and not down:
        for feat, val in (rec.get("top_features_lime") or []):
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            (up if v >= 0 else down).append((feat, v))

    if up or down:
        parts = []
        if up:
            top = ", ".join(
                "**{}** (+{})".format(f, _fmt_value(v)) for f, v in up[:3])
            parts.append("los factores que más la empujaron hacia arriba "
                         "fueron {}".format(top))
        if down:
            top = ", ".join(
                "**{}** ({})".format(f, _fmt_value(v)) for f, v in down[:3])
            parts.append("los que más la empujaron hacia abajo fueron "
                         "{}".format(top))
        lines.append("¿Por qué? " + "; ".join(parts) + ".")

    return "\n".join(lines)


def narrate_clustering(metrics, model_type="kmeans", n_features=None):
    """Narrativa determinista para clustering — describe k, calidad y tamaños."""
    n_c = metrics.get("n_clusters")
    sil = metrics.get("silhouette")
    inertia = metrics.get("inertia")
    db = metrics.get("davies_bouldin")
    ch = metrics.get("calinski_harabasz")
    lines = []
    if n_c:
        feat_txt = f" sobre {n_features} variables" if n_features else ""
        lines.append(
            f"Este modelo es un **{model_type}** no supervisado que descubrió **{n_c} grupos**{feat_txt}."
        )
    if sil is not None:
        try:
            s = float(sil)
            if s > 0.5:
                qual = "muy bien separados — los grupos son compactos y distintos"
            elif s > 0.25:
                qual = "moderadamente separados — hay cierta superposición pero la estructura es válida"
            elif s > 0.0:
                qual = "débilmente separados — los grupos se solapan y la segmentación es exploratoria"
            else:
                qual = "sin separación clara — los datos no forman grupos naturales con este k"
            lines.append(f"La silueta es **{s:.3f}** ({qual}; rango -1 a 1, mayor es mejor).")
        except Exception:
            pass
    if inertia is not None:
        lines.append(f"La inercia (suma de distancias cuadráticas al centroide) es **{float(inertia):,.1f}** — menor indica grupos más compactos.")
    if db is not None:
        lines.append(f"Davies-Bouldin = **{float(db):.3f}** (menor es mejor, mide solapamiento).")
    if ch is not None:
        lines.append(f"Calinski-Harabasz = **{float(ch):.1f}** (mayor es mejor, mide dispersión entre vs dentro).")
    cs = metrics.get("cluster_sizes") or {}
    if cs:
        total = sum(cs.values()) or 1
        parts = ", ".join(f"**C{c}**: {cnt} ({cnt/total:.1%})" for c, cnt in sorted(cs.items()))
        lines.append(f"Tamaños: {parts}.")
    if not lines:
        return f"Modelo de clustering **{model_type}** entrenado."
    return " ".join(lines)


def narrate_cluster_profiles(centroids_json_path=None, centroids_data=None, top_k=3):
    """Describe los prototipos (centroides) — qué distingue cada cluster."""
    import json as _json
    import os as _os
    data = centroids_data
    if data is None and centroids_json_path and _os.path.exists(centroids_json_path):
        try:
            with open(centroids_json_path, "r", encoding="utf-8") as fh:
                data = _json.load(fh)
        except Exception:
            data = None
    if not data:
        return ""
    # data is list of dicts with cluster + feature_scaled
    try:
        lines = ["## Perfiles de cada cluster (prototipos)", ""]
        for rec in data:
            c = rec.get("cluster", "?")
            cnt = rec.get("count", "?")
            # collect feature_scaled values
            feats = [(k.replace("_scaled", ""), v) for k, v in rec.items() if k.endswith("_scaled")]
            feats_sorted = sorted(feats, key=lambda t: -abs(t[1]))[:top_k]
            top = ", ".join(f"**{f}** ({v:+.2f}σ)" for f, v in feats_sorted)
            lines.append(f"- **Cluster {c}** (n={cnt}): se distingue por {top}.")
        lines.append("")
        lines.append("Valores en σ (desviaciones sobre la media escalada): positivo = por encima del promedio, negativo = por debajo.")
        return "\n".join(lines)
    except Exception:
        return ""


def narrate_image(metrics, names, predictions, xai):
    """Narrativa global y por imagen para la rama de detección."""
    lines = []
    mAP50 = metrics.get("mAP50")
    mAP5095 = metrics.get("mAP50-95")
    if mAP50 is not None:
        lines.append(
            "El modelo alcanzó un **mAP50 de {:.3f}** y un **mAP50-95 de "
            "{:.3f}** en el conjunto de test, lo que indica que detecta los "
            "objetos con {} precisión.".format(
                float(mAP50),
                float(mAP5095) if mAP5095 is not None else 0.0,
                "buena" if float(mAP50) >= 0.5 else "moderada"))

    per_class = metrics.get("per_class") or {}
    if per_class:
        ranked = sorted(
            ((n, v.get("AP50") or 0.0) for n, v in per_class.items()),
            key=lambda t: -t[1])
        if ranked:
            best, best_v = ranked[0]
            worst, worst_v = ranked[-1]
            lines.append(
                "La clase mejor detectada es **{}** (AP50 = {:.3f})".format(
                    best, best_v))
            if len(ranked) > 1:
                lines.append(
                    "y la más difícil es **{}** (AP50 = {:.3f}).".format(
                        worst, worst_v))
            else:
                lines.append(".")

    if not lines:
        return ""

    header = ["## ¿Qué está haciendo este modelo?", ""]
    header.extend(lines)
    header.append("")

    for rec in predictions or []:
        dets = rec.get("detections") or []
        img = rec.get("image") or "desconocida"
        if not dets:
            header.append(
                "En `{}` el modelo **no encontró ninguna detección** por "
                "encima del umbral de confianza.".format(img))
            continue
        counts = {}
        confs = []
        for d in dets:
            cn = d.get("class_name") or str(d.get("class"))
            counts[cn] = counts.get(cn, 0) + 1
            confs.append(d.get("confidence") or 0.0)
        desc = ", ".join("{}× {}".format(v, k) for k, v in counts.items())
        mean_conf = sum(confs) / len(confs)
        header.append(
            "En `{}` el modelo detectó **{}** con una confianza media de "
            "**{:.0%}**.".format(img, desc, mean_conf))

        gradcam = (xai.get("gradcam") or {}).get("instances") or []
        for inst in gradcam:
            if inst.get("image") == rec.get("image") or inst.get("png"):
                header.append(
                    "El mapa de calor **Grad-CAM** señala las regiones de la "
                    "imagen que más respaldan la clase detectada "
                    "({}).".format(inst.get("class_name") or "objetivo"))
                break

    return "\n".join(header)