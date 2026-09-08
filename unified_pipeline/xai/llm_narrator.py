"""Genera narrativa en lenguaje natural usando LLM (OpenRouter/dots-3-note-preview).
Si falla la llamada a la API o no hay clave, devuelve el texto de las plantillas
deterministas de narrator.py para mantener el reporte siempre usable.

El cliente de OpenAI se crea bajo demanda (lazy) para evitar fallos de import
cuando no hay configurada la clave API.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def _get_client():
    """Devuelve un cliente OpenAI recién creado usando la clave del entorno."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


LLM_MODEL = "dots-studio/dots-3-note-preview:free"


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------
def _build_prompt_model(metrics, task, model_type, target=None):
    """Return a concise prompt for the LLM to narrate the model."""
    target_txt = f"sobre `{target}`" if target else "sobre los datos"

    # Resumir la métrica principal en una frase corta
    if task == "classification":
        acc = metrics.get("accuracy")
        f1 = metrics.get("f1")
        parts = []
        if acc is not None:
            parts.append(f"alcanzó una exactitud del {acc*100:.1f}%")
        if f1 is not None:
            parts.append(f"un F1 de {f1:.3f}")
        metric_str = " y ".join(parts) if parts else "buen rendimiento"
    else:
        r2 = metrics.get("r2")
        rmse = metrics.get("rmse")
        mae = metrics.get("mae")
        parts = []
        if r2 is not None:
            parts.append(f"R² de {r2:.3f}")
        if rmse is not None:
            parts.append(f"RMSE de {rmse:.2f}")
        if mae is not None:
            parts.append(f"MAE de {mae:.2f}")
        metric_str = " y ".join(parts) if parts else "resultados"

    # Top features (opcional, hasta 3)
    feat_str = ""

    return (
        f"Eres un experto en Machine Learning. Escribe un párrafo en español, "
        f"máximo 3 oraciones, explicando un modelo {model_type} para {task} "
        f"{target_txt} con {metric_str}.\n"
        f"Escribe en tono cercano, como si explicara a un usuario no técnico. "
        f"No incluyas fórmulas ni nombres de variables extraños. "
        f"Solo devuelve el texto, sin censura ni comillas alrededor."
    )


# ---------------------------------------------------------------------------
# Public API — firmas idénticas a narrator.py para swap transparente
# ---------------------------------------------------------------------------
def narrate_model(metrics, task, model_type, target=None):
    """Intenta generar texto con el LLM; si falla, usa plantillas deterministas."""

    # 1) Intentar LLM si hay clave
    client = _get_client()
    if client is not None:
        try:
            prompt = _build_prompt_model(metrics, task, model_type, target)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception:
            pass

    # 2) Fallback: plantillas deterministas de narrator.py
    from . import narrator as det
    return det.narrate_model(metrics, task, model_type, target)


def narrate_clustering(metrics, model_type="kmeans", n_features=None):
    """Intenta LLM para clustering; fallback a narrator determinista."""
    client = _get_client()
    if client is not None:
        try:
            n_c = metrics.get("n_clusters", "?")
            sil = metrics.get("silhouette")
            sil_txt = f"silueta {sil:.3f}" if sil is not None else "silueta no disponible"
            cs = metrics.get("cluster_sizes") or {}
            sizes_txt = ", ".join(f"C{c}:{cnt}" for c, cnt in sorted(cs.items())) if cs else "tamaños variables"
            feat_txt = f"sobre {n_features} variables" if n_features else ""
            prompt = (
                f"Eres un experto en clustering. Escribe un párrafo en español, máximo 4 oraciones, "
                f"explicando un modelo {model_type} no supervisado {feat_txt} que encontró {n_c} grupos "
                f"con {sil_txt} y distribución {sizes_txt}. "
                f"Tono cercano para usuario no técnico, sin fórmulas, solo el texto."
            )
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception:
            pass
    from . import narrator as det
    return det.narrate_clustering(metrics, model_type, n_features)


def narrate_cluster_profiles(centroids_json_path=None, centroids_data=None, top_k=3):
    """LLM para perfiles de clusters; fallback determinista."""
    client = _get_client()
    if client is not None and centroids_data:
        try:
            # Build short feature highlights for prompt
            highlights = []
            for rec in centroids_data[:4]:
                c = rec.get("cluster", "?")
                feats = [(k.replace("_scaled", ""), v) for k, v in rec.items() if k.endswith("_scaled")]
                top = sorted(feats, key=lambda t: -abs(t[1]))[:top_k]
                top_txt = ", ".join(f"{f} ({v:+.2f})" for f, v in top)
                highlights.append(f"Cluster {c}: {top_txt}")
            prompt = (
                f"Eres un experto en segmentación. En 3-4 oraciones en español, "
                f"describe qué distingue cada grupo de clientes a partir de estos prototipos (valores en σ): "
                f"{' | '.join(highlights)}. "
                f"Usa lenguaje de negocio (ej. 'compradores intensivos', 'usuarios de efectivo'), tono cercano, solo texto."
            )
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception:
            pass
    from . import narrator as det
    return det.narrate_cluster_profiles(centroids_json_path, centroids_data, top_k)


def narrate_top_features(
    feature_importance=None, shap_global=None, top_n=15
):
    """Versión LLM + fallback para las características más importantes."""

    client = _get_client()
    if client is None:
        from . import narrator as det
        return det.narrate_top_features(feature_importance, shap_global, top_n)

    try:
        # Construir lista corto de features (máx 3)
        source = None
        if shap_global:
            source = [
                {"feature": r.get("feature"), "value": r.get("mean_abs_shap")}
                for r in shap_global
                if r.get("feature")
            ]
            metric_name = "contribución media (mean |SHAP|)"
        elif feature_importance:
            source = [
                {"feature": r.get("feature"), "value": r.get("importance")}
                for r in feature_importance
                if r.get("feature")
            ]
            metric_name = "importancia global"

        if source:
            # ordenar y tomar top_n (o 3 para el prompt)
            n = top_n if top_n else 3
            source = sorted(source, key=lambda r: -(r["value"] or 0.0))[:n]
            feat_lines = []
            for i, rec in enumerate(source, start=1):
                feat_lines.append(
                    f"{i}. **{rec['feature']}** ({metric_name} = {rec['value']:.3f})"
                )
            features_text = "; ".join(feat_lines)
            prompt = (
                f"Eres un experto en Machine Learning. Escribe 2 oraciones en español "
                f"para un usuario no técnico, explicando cuáles son las características "
                f"más importantes del modelo y por qué ayudan a la predicción. "
                f"Características destacadas: {features_text}. "
                f"Sé conciso, usa tono cercano y devuelve solo el párrafo."
            )
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
    except Exception:
        pass

    # Fallback determinista
    from . import narrator as det
    return det.narrate_top_features(feature_importance, shap_global, top_n)