# Reporte — Detección YOLO11

## Entrenamiento
- Modelo: `yolo11` (variante `yolov11n`)
- Dispositivo: `0` · tiempo: 2.02 min
- Clases: Cat, Dog
- EDA: `class_distribution.png`, `sample_grid.jpg` y `eda_image.json` (distribución por split, stats de tamaño)

## Métricas (test)
- mAP50: **0.7980**
- mAP50-95: **0.6025**
- Precision (mean): 0.9256
- Recall (mean): 0.8023
    - Cat · AP50=0.8950 · AP50-95=0.7158
    - Dog · AP50=0.7009 · AP50-95=0.4893

## ¿Qué está haciendo este modelo?

El modelo alcanzó un **mAP50 de 0.798** y un **mAP50-95 de 0.603** en el conjunto de test, lo que indica que detecta los objetos con buena precisión.
La clase mejor detectada es **Cat** (AP50 = 0.895)
y la más difícil es **Dog** (AP50 = 0.701).

En `C:\Users\I13310\Desktop\Proyecto\unified_pipeline\data\CatsDogs_YOLO11\test\images\cat_111_jpg.rf.XSCmTLQPKBr7ZNZzXhm0.jpg` el modelo **no encontró ninguna detección** por encima del umbral de confianza.
En `C:\Users\I13310\Desktop\Proyecto\unified_pipeline\data\CatsDogs_YOLO11\test\images\cat_14_jpg.rf.2aEiCGe3AlOrkQg6hHz8.jpg` el modelo detectó **1× Cat** con una confianza media de **91%**.
El mapa de calor **Grad-CAM** señala las regiones de la imagen que más respaldan la clase detectada (Cat).
En `C:\Users\I13310\Desktop\Proyecto\unified_pipeline\data\CatsDogs_YOLO11\test\images\cat_170_jpg.rf.rgEfrCyIhPrJGznb5SJ8.jpg` el modelo detectó **1× Cat** con una confianza media de **95%**.
El mapa de calor **Grad-CAM** señala las regiones de la imagen que más respaldan la clase detectada (Cat).
En `C:\Users\I13310\Desktop\Proyecto\unified_pipeline\data\CatsDogs_YOLO11\test\images\cat_208_jpg.rf.lXeTKCfgfVEp2CUm37CX.jpg` el modelo detectó **1× Cat, 1× Dog** con una confianza media de **66%**.
El mapa de calor **Grad-CAM** señala las regiones de la imagen que más respaldan la clase detectada (Cat).

## Predicciones de ejemplo (`predictions/`)
- `cat_111_jpg.rf.XSCmTLQPKBr7ZNZzXhm0.jpg` → 0 detecciones
- `cat_14_jpg.rf.2aEiCGe3AlOrkQg6hHz8.jpg` → 1 detecciones
    - Cat (0.91) bbox [322.7, 10.1, 665.3, 350.1]
- `cat_170_jpg.rf.rgEfrCyIhPrJGznb5SJ8.jpg` → 1 detecciones
    - Cat (0.95) bbox [312.8, 23.6, 960.5, 700.6]
- `cat_208_jpg.rf.lXeTKCfgfVEp2CUm37CX.jpg` → 2 detecciones
    - Cat (0.90) bbox [40.9, 2.2, 241.7, 217.1]
    - Dog (0.42) bbox [328.7, 79.9, 422.4, 177.3]

## Explicabilidad (`xai/`)

### GRADCAM
Artefactos:
- `gradcam_cat_111_jpg.rf.XSCmTLQPKBr7ZNZzXhm0.png` (clase objetivo: Cat)
- `gradcam_cat_14_jpg.rf.2aEiCGe3AlOrkQg6hHz8.png` (clase objetivo: Cat)
- `gradcam_cat_170_jpg.rf.rgEfrCyIhPrJGznb5SJ8.png` (clase objetivo: Cat)
- `gradcam_cat_208_jpg.rf.lXeTKCfgfVEp2CUm37CX.png` (clase objetivo: Cat)

### OCCLUSION
Artefactos:
- `occlusion_cat_14_jpg.rf.2aEiCGe3AlOrkQg6hHz8.png` (clase objetivo: None)
- `occlusion_cat_170_jpg.rf.rgEfrCyIhPrJGznb5SJ8.png` (clase objetivo: None)
- `occlusion_cat_208_jpg.rf.lXeTKCfgfVEp2CUm37CX.png` (clase objetivo: None)

### LIME_IMAGE
Artefactos:
- `lime_image_cat_111_jpg.rf.XSCmTLQPKBr7ZNZzXhm0.png` (clase objetivo: Cat)
- `lime_image_cat_14_jpg.rf.2aEiCGe3AlOrkQg6hHz8.png` (clase objetivo: Cat)
- `lime_image_cat_170_jpg.rf.rgEfrCyIhPrJGznb5SJ8.png` (clase objetivo: Cat)
- `lime_image_cat_208_jpg.rf.lXeTKCfgfVEp2CUm37CX.png` (clase objetivo: Cat)

## Cómo interpretar los mapas
- **Grad-CAM**: mapa de calor sobre la última conv de la rama de clases.
    Las zonas rojas muestran qué regiones **soportan** la clase detectada.
- **Occlusion Sensitivity**: celda en gris → caída de confianza de la caja.
    Celdas rojas = la confianza Depende de esa zona.
- **LIME Image**: superpíxeles que, al ocultarse, aumentaron la clase objetivo
    (verde) o la disminuyeron (rojo). Se agrega la confianza por clase como
    "probabilidad" para el clasificador LIME.

Artículo de referencia: métricas en `metrics.json`, predicciones en `predictions/predictions.json`, detalle XAI en `xai/*.json` y mapas `.npy`.