# Instancia explicativa

## ¿Qué está haciendo este modelo?

Utilizamos un modelo de máquina de soporte vectorial para clasificar los datos. Al evaluarlo, obtuvo una precisión del 97.7 % y un F1 de 0.977, lo que muestra que sus predicciones son muy fiables.

Las características más influyentes son el peor radio, los puntos cóncavos y el perímetro, ya que su variación indica el tamaño y la forma de la lesión, facilitando la predicción de su malignidad. Además, el área, la textura y la simetría aportan información complementaria que mejora la precisión del modelo.

## ¿Por qué el modelo predijo lo que predijo?

### Muestra (index 102)
El modelo predijo la clase **1**.
Asignó una probabilidad del **98.3%** a esa clase.
¿Por qué? los factores que más la empujaron hacia arriba fueron **worst concave points** (+0.0409), **worst area** (+0.0389), **worst radius** (+0.0364).

### Muestra (index 17)
El modelo predijo la clase **0**.
Asignó una probabilidad del **99.8%** a esa clase.
¿Por qué? los que más la empujaron hacia abajo fueron **worst smoothness** (-0.1016), **worst concave points** (-0.0823), **mean concave points** (-0.0717).

### Muestra (index 370)
El modelo predijo la clase **0**.
Asignó una probabilidad del **99.7%** a esa clase.
¿Por qué? los que más la empujaron hacia abajo fueron **worst symmetry** (-0.1361), **worst concave points** (-0.0829), **worst concavity** (-0.0794).

## Detalle técnico — Muestra (index 102)
- True: 1 | Pred: 1
- Probabilidades: {'0': 0.017, '1': 0.983}
- Top features LIME:
    - worst texture > 0.58 -> -0.1212
    - worst smoothness <= -0.75 -> 0.0885
    - mean concave points <= -0.73 -> 0.0653
    - mean concavity <= -0.74 -> 0.0559
    - area error <= -0.60 -> 0.0556
    - -0.64 < worst area <= -0.35 -> 0.0504
    - radius error <= -0.68 -> 0.0488
    - -0.66 < worst symmetry <= -0.15 -> 0.0473
    - -0.69 < worst perimeter <= -0.28 -> 0.0440
    - texture error > 0.47 -> 0.0421
- Top SHAP:
    - worst concave points -> 0.0409
    - worst area -> 0.0389
    - worst radius -> 0.0364
    - worst perimeter -> 0.0356
    - worst smoothness -> 0.0343
    - area error -> 0.0304
    - mean concave points -> 0.0302
    - worst concavity -> 0.0280

## Detalle técnico — Muestra (index 17)
- True: 0 | Pred: 0
- Probabilidades: {'0': 0.998, '1': 0.002}
- Top features LIME:
    - worst radius > 0.52 -> -0.1426
    - worst texture > 0.58 -> -0.1338
    - worst area > 0.38 -> -0.1318
    - worst perimeter > 0.56 -> -0.1192
    - mean concave points > 0.67 -> -0.1155
    - area error > 0.20 -> -0.1142
    - mean radius > 0.48 -> -0.1136
    - mean area > 0.37 -> -0.1126
    - worst concave points > 0.78 -> -0.1124
    - worst smoothness > 0.62 -> -0.1039
- Top SHAP:
    - worst smoothness -> -0.1016
    - worst concave points -> -0.0823
    - mean concave points -> -0.0717
    - worst perimeter -> -0.0639
    - worst radius -> -0.0639
    - worst texture -> -0.0636
    - worst symmetry -> -0.0579
    - worst concavity -> -0.0573

## Detalle técnico — Muestra (index 370)
- True: 0 | Pred: 0
- Probabilidades: {'0': 0.9969, '1': 0.0031}
- Top features LIME:
    - worst area > 0.38 -> -0.1347
    - worst radius > 0.52 -> -0.1306
    - worst concave points > 0.78 -> -0.1251
    - worst texture > 0.58 -> -0.1208
    - worst perimeter > 0.56 -> -0.1192
    - mean concave points > 0.67 -> -0.1090
    - area error > 0.20 -> -0.1079
    - mean area > 0.37 -> -0.1024
    - mean radius > 0.48 -> -0.1016
    - mean texture > 0.55 -> -0.1004
- Top SHAP:
    - worst symmetry -> -0.1361
    - worst concave points -> -0.0829
    - worst concavity -> -0.0794
    - worst texture -> -0.0665
    - mean concave points -> -0.0664
    - worst radius -> -0.0494
    - worst area -> -0.0484
    - mean concavity -> -0.0478
