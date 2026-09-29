# Instancia explicativa

## ¿Qué está haciendo este modelo?

Hemos entrenado un árbol de decisiones que clasifica la variable objetivo con una precisión del 90.7%, lo que significa que acierta en casi 9 de cada 10 casos. Su puntuación F1 de 0.906 indica que equilibra bien la precisión y la cobertura, ofreciendo resultados fiables. Esto hace que el modelo sea una herramienta útil para predecir nuevos datos de forma sencilla.

Las características más importantes son el radio máximo, los puntos cóncavos y la concavidad máxima, ya que muestran el tamaño y la forma irregular de las células. Estos valores ayudan al modelo a predecir si un tumor es benigno o maligno.

## ¿Por qué el modelo predijo lo que predijo?

### Muestra (index 102)
El modelo predijo la clase **1**.
Asignó una probabilidad del **100.0%** a esa clase.
¿Por qué? los factores que más la empujaron hacia arriba fueron **worst radius** (+0.2780), **worst concavity** (+0.0622), **mean fractal dimension** (+0.0298); los que más la empujaron hacia abajo fueron **worst texture** (-0.0370), **compactness error** (-0.0203), **texture error** (-0.0033).

### Muestra (index 17)
El modelo predijo la clase **0**.
Asignó una probabilidad del **100.0%** a esa clase.
¿Por qué? los que más la empujaron hacia abajo fueron **worst radius** (-0.2650), **worst concave points** (-0.1860), **area error** (-0.0631).

### Muestra (index 370)
El modelo predijo la clase **0**.
Asignó una probabilidad del **100.0%** a esa clase.
¿Por qué? los factores que más la empujaron hacia arriba fueron **compactness error** (+0.0041); los que más la empujaron hacia abajo fueron **worst radius** (-0.3168), **worst concave points** (-0.2378), **worst texture** (-0.0356).

## Detalle técnico — Muestra (index 102)
- True: 1 | Pred: 1
- Probabilidades: {'0': 0.0, '1': 1.0}
- Top features LIME:
    - -0.66 < worst radius <= -0.28 -> 0.2325
    - worst texture > 0.58 -> -0.1843
    - -0.77 < worst concavity <= -0.23 -> 0.1219
    - area error <= -0.60 -> 0.0868
    - mean fractal dimension <= -0.72 -> 0.0865
    - compactness error <= -0.69 -> -0.0797
    - -0.75 < worst concave points <= -0.26 -> 0.0703
    - -0.69 < mean area <= -0.31 -> -0.0278
    - fractal dimension error <= -0.57 -> 0.0254
    - mean smoothness <= -0.74 -> 0.0221
- Top SHAP:
    - worst radius -> 0.2780
    - worst concavity -> 0.0622
    - worst texture -> -0.0370
    - mean fractal dimension -> 0.0298
    - area error -> 0.0295
    - worst concave points -> 0.0293
    - compactness error -> -0.0203
    - texture error -> -0.0033

## Detalle técnico — Muestra (index 17)
- True: 0 | Pred: 0
- Probabilidades: {'0': 1.0, '1': 0.0}
- Top features LIME:
    - worst radius > 0.52 -> -0.3791
    - worst texture > 0.58 -> -0.2246
    - worst concavity > 0.48 -> -0.1777
    - worst concave points > 0.78 -> -0.1448
    - area error > 0.20 -> -0.1415
    - worst smoothness > 0.62 -> -0.1227
    - mean fractal dimension > 0.47 -> -0.0691
    - mean area > 0.37 -> 0.0371
    - -0.20 < concavity error <= 0.28 -> -0.0352
    - -0.30 < compactness error <= 0.36 -> 0.0247
- Top SHAP:
    - worst radius -> -0.2650
    - worst concave points -> -0.1860
    - area error -> -0.0631
    - worst smoothness -> -0.0531
    - worst texture -> -0.0356
    - texture error -> -0.0151
    - worst concavity -> -0.0064
    - mean fractal dimension -> -0.0060

## Detalle técnico — Muestra (index 370)
- True: 0 | Pred: 0
- Probabilidades: {'0': 1.0, '1': 0.0}
- Top features LIME:
    - worst radius > 0.52 -> -0.3759
    - worst texture > 0.58 -> -0.1922
    - worst concavity > 0.48 -> -0.1733
    - area error > 0.20 -> -0.1590
    - worst concave points > 0.78 -> -0.1403
    - -0.20 < mean fractal dimension <= 0.47 -> -0.1030
    - compactness error > 0.36 -> 0.0403
    - symmetry error > 0.33 -> -0.0297
    - -0.68 < texture error <= -0.27 -> -0.0279
    - mean symmetry > 0.53 -> 0.0260
- Top SHAP:
    - worst radius -> -0.3168
    - worst concave points -> -0.2378
    - worst texture -> -0.0356
    - area error -> -0.0182
    - texture error -> -0.0151
    - worst concavity -> -0.0064
    - mean fractal dimension -> -0.0060
    - compactness error -> 0.0041
