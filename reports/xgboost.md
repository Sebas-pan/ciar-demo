# Instancia explicativa

## ¿Qué está haciendo este modelo?

Entrenamos un modelo XGBoost para clasificar el target y obtuvimos una exactitud del 91.9% y un F1 de 0.918. Estos números indican que el modelo acierta la mayoría de las veces y maneja bien el desbalance entre las clases. Por lo tanto, es una herramienta muy útil para predecir resultados.

El modelo se apoya principalmente en características como los puntos cóncavos peores, el área peor y el radio peor, que capturan la forma y el tamaño del tumor. Estas variables son fundamentales porque proporcionan información clara sobre la malignidad, lo que mejora la precisión de la predicción.

## ¿Por qué el modelo predijo lo que predijo?

### Muestra (index 102)
El modelo predijo la clase **1**.
Asignó una probabilidad del **99.4%** a esa clase.
¿Por qué? los factores que más la empujaron hacia arriba fueron **worst concave points** (+1.10), **worst area** (+0.9881), **area error** (+0.7685); los que más la empujaron hacia abajo fueron **worst texture** (-0.8129).

### Muestra (index 17)
El modelo predijo la clase **0**.
Asignó una probabilidad del **99.9%** a esa clase.
¿Por qué? los que más la empujaron hacia abajo fueron **worst concave points** (-1.57), **worst radius** (-0.9543), **worst area** (-0.8799).

### Muestra (index 370)
El modelo predijo la clase **0**.
Asignó una probabilidad del **99.9%** a esa clase.
¿Por qué? los que más la empujaron hacia abajo fueron **worst concave points** (-1.64), **worst radius** (-1.01), **worst area** (-0.9527).

## Detalle técnico — Muestra (index 102)
- True: 1 | Pred: 1
- Probabilidades: {'0': 0.0064, '1': 0.9936}
- Top features LIME:
    - -0.75 < worst concave points <= -0.26 -> 0.1578
    - worst texture > 0.58 -> -0.1565
    - -0.64 < worst area <= -0.35 -> 0.1172
    - area error <= -0.60 -> 0.1161
    - -0.66 < worst radius <= -0.28 -> 0.0996
    - -0.69 < worst perimeter <= -0.28 -> 0.0840
    - worst smoothness <= -0.75 -> 0.0804
    - -0.77 < worst concavity <= -0.23 -> 0.0789
    - mean smoothness <= -0.74 -> 0.0775
    - compactness error <= -0.69 -> -0.0682
- Top SHAP:
    - worst concave points -> 1.1020
    - worst area -> 0.9881
    - worst texture -> -0.8129
    - area error -> 0.7685
    - worst radius -> 0.7584
    - worst concavity -> 0.7305
    - worst perimeter -> 0.7195
    - mean smoothness -> 0.5750

## Detalle técnico — Muestra (index 17)
- True: 0 | Pred: 0
- Probabilidades: {'0': 0.9994, '1': 0.0006}
- Top features LIME:
    - worst concave points > 0.78 -> -0.3396
    - worst radius > 0.52 -> -0.2065
    - worst area > 0.38 -> -0.1882
    - worst texture > 0.58 -> -0.1816
    - area error > 0.20 -> -0.1639
    - worst perimeter > 0.56 -> -0.1238
    - worst concavity > 0.48 -> -0.0922
    - worst smoothness > 0.62 -> -0.0746
    - mean concave points > 0.67 -> -0.0599
    - worst symmetry > 0.43 -> -0.0514
- Top SHAP:
    - worst concave points -> -1.5735
    - worst radius -> -0.9543
    - worst area -> -0.8799
    - worst perimeter -> -0.7027
    - worst texture -> -0.6851
    - area error -> -0.6256
    - worst concavity -> -0.5170
    - worst smoothness -> -0.4897

## Detalle técnico — Muestra (index 370)
- True: 0 | Pred: 0
- Probabilidades: {'0': 0.9994, '1': 0.0006}
- Top features LIME:
    - worst concave points > 0.78 -> -0.3623
    - worst radius > 0.52 -> -0.1982
    - worst area > 0.38 -> -0.1822
    - worst texture > 0.58 -> -0.1603
    - area error > 0.20 -> -0.1593
    - worst perimeter > 0.56 -> -0.1221
    - worst concavity > 0.48 -> -0.1076
    - mean concave points > 0.67 -> -0.0699
    - mean texture > 0.55 -> -0.0508
    - worst symmetry > 0.43 -> -0.0456
- Top SHAP:
    - worst concave points -> -1.6405
    - worst radius -> -1.0063
    - worst area -> -0.9527
    - worst perimeter -> -0.8090
    - worst texture -> -0.6797
    - area error -> -0.6284
    - worst concavity -> -0.5219
    - worst smoothness -> -0.3727
