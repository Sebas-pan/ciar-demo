# Instancia explicativa

## ¿Qué está haciendo este modelo?

El modelo de regresión logística que utilicé para predecir el objetivo logró una exactitud del 98.8%, lo que significa que acertó casi todas las veces. Además, su puntuación F1 de 0.988 muestra que equilibra muy bien la precisión y el recuerdo, por lo que es confiable para la tarea.

Las características más importantes del modelo son worst concave points, worst texture, radius error y worst radius. Estas variables capturan información clave sobre la forma y el tamaño de las células, lo que permite al modelo predecir con mayor precisión.

## ¿Por qué el modelo predijo lo que predijo?

### Muestra (index 102)
El modelo predijo la clase **1**.
Asignó una probabilidad del **99.2%** a esa clase.
¿Por qué? los factores que más la empujaron hacia arriba fueron **worst concave points** (+0.0462), **area error** (+0.0383), **worst radius** (+0.0372).

### Muestra (index 17)
El modelo predijo la clase **0**.
Asignó una probabilidad del **100.0%** a esa clase.
¿Por qué? los que más la empujaron hacia abajo fueron **worst smoothness** (-0.0863), **worst concave points** (-0.0833), **worst texture** (-0.0775).

### Muestra (index 370)
El modelo predijo la clase **0**.
Asignó una probabilidad del **100.0%** a esa clase.
¿Por qué? los que más la empujaron hacia abajo fueron **worst symmetry** (-0.1321), **worst concave points** (-0.0992), **worst concavity** (-0.0846).

## Detalle técnico — Muestra (index 102)
- True: 1 | Pred: 1
- Probabilidades: {'0': 0.0084, '1': 0.9916}
- Top features LIME:
    - worst texture > 0.58 -> -0.2279
    - area error <= -0.60 -> 0.1171
    - worst smoothness <= -0.75 -> 0.1106
    - radius error <= -0.68 -> 0.0962
    - mean compactness <= -0.76 -> -0.0931
    - compactness error <= -0.69 -> -0.0899
    - texture error > 0.47 -> 0.0834
    - mean concave points <= -0.73 -> 0.0827
    - mean concavity <= -0.74 -> 0.0799
    - -0.66 < worst symmetry <= -0.15 -> 0.0782
- Top SHAP:
    - worst concave points -> 0.0462
    - area error -> 0.0383
    - worst radius -> 0.0372
    - worst area -> 0.0340
    - radius error -> 0.0318
    - worst concavity -> 0.0314
    - mean concave points -> 0.0303
    - worst perimeter -> 0.0298

## Detalle técnico — Muestra (index 17)
- True: 0 | Pred: 0
- Probabilidades: {'0': 1.0, '1': 0.0}
- Top features LIME:
    - worst texture > 0.58 -> -0.2530
    - worst radius > 0.52 -> -0.2095
    - worst area > 0.38 -> -0.2034
    - area error > 0.20 -> -0.2007
    - worst concave points > 0.78 -> -0.1827
    - radius error > 0.36 -> -0.1801
    - worst symmetry > 0.43 -> -0.1777
    - worst perimeter > 0.56 -> -0.1568
    - worst concavity > 0.48 -> -0.1542
    - mean compactness > 0.50 -> 0.1458
- Top SHAP:
    - worst smoothness -> -0.0863
    - worst concave points -> -0.0833
    - worst texture -> -0.0775
    - worst radius -> -0.0719
    - mean concave points -> -0.0683
    - worst area -> -0.0639
    - worst symmetry -> -0.0620
    - radius error -> -0.0563

## Detalle técnico — Muestra (index 370)
- True: 0 | Pred: 0
- Probabilidades: {'0': 1.0, '1': 0.0}
- Top features LIME:
    - worst texture > 0.58 -> -0.2342
    - area error > 0.20 -> -0.2134
    - worst area > 0.38 -> -0.2110
    - worst concave points > 0.78 -> -0.2064
    - worst radius > 0.52 -> -0.1919
    - worst symmetry > 0.43 -> -0.1683
    - worst concavity > 0.48 -> -0.1640
    - worst perimeter > 0.56 -> -0.1443
    - mean compactness > 0.50 -> 0.1406
    - mean concave points > 0.67 -> -0.1313
- Top SHAP:
    - worst symmetry -> -0.1321
    - worst concave points -> -0.0992
    - worst concavity -> -0.0846
    - worst texture -> -0.0696
    - worst radius -> -0.0590
    - worst area -> -0.0514
    - mean concave points -> -0.0498
    - worst perimeter -> -0.0455
