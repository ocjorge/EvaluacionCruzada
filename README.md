# Evaluación Cruzada: Modelo EMG en Dataset NinaPro DB3

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Dataset](https://img.shields.io/badge/Dataset-NinaPro_DB3-yellow)

Este proyecto realiza una evaluación cruzada de un modelo pre-entrenado de clasificación de gestos EMG, probando su rendimiento en el dataset NinaPro DB3 (sujetos amputados).

## 📋 Descripción

![EMG](https://img.shields.io/badge/EMG-Classification-purple)
![Cross-Dataset](https://img.shields.io/badge/Evaluation-Cross--Dataset-red)

El código evalúa un modelo híbrido CNN-LSTM pre-entrenado en el dataset DB1 (sujetos sanos) contra el dataset DB3 (sujetos amputados), analizando la capacidad de generalización del modelo entre poblaciones diferentes.

## 🎯 Objetivo

![Target](https://img.shields.io/badge/Goal-Model_Transferability-blueviolet)

Evaluar la transferibilidad de un modelo de clasificación de gestos EMG entrenado con sujetos sanos cuando se aplica a sujetos amputados.

## 🛠️ Tecnologías Utilizadas

![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Metrics-F7931E)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)
![Seaborn](https://img.shields.io/badge/Seaborn-Heatmaps-4BB5B2)
![NumPy](https://img.shields.io/badge/NumPy-Arrays-013243)
![SciPy](https://img.shields.io/badge/SciPy-.mat_loading-8CAAE6)

## ⚙️ Instalación

```bash
# Instalar dependencias
pip install tensorflow scipy matplotlib seaborn scikit-learn numpy

# Si hay problemas con la carga del modelo, instalar:
pip install h5py==3.1.0
```

## 📊 Datasets

### Dataset de Entrenamiento (Base)
![DB1](https://img.shields.io/badge/DB1-Healthy_Subjects-green)
- **NinaPro DB1**: Sujetos sanos
- 52 gestos diferentes
- 10 electrodos EMG

### Dataset de Evaluación
![DB3](https://img.shields.io/badge/DB3-Amputated_Subjects-orange)
- **NinaPro DB3**: Sujetos amputados
- 11 sujetos
- Mismos 52 gestos que DB1
- 10 electrodos EMG

## 🏗️ Estructura del Proyecto

```
├── mejor_modelo_base.h5          # Modelo pre-entrenado en DB1
├── ninapro_db3_data/             # Dataset DB3
│   ├── S1_E1_A1.mat
│   ├── S1_E2_A1.mat
│   └── ...
└── evaluation_cross_dataset.py    # Script de evaluación
```

## ⚙️ Parámetros de Configuración

```python
# Ruta del dataset DB3
DATASET_PATH_DB3 = 'ninapro_db3_data'

# Sujetos a procesar (DB3 tiene 11 sujetos)
SUBJECTS_TO_PROCESS_DB3 = list(range(1, 12))

# Parámetros de preprocesamiento (deben coincidir con el entrenamiento)
WINDOW_SIZE = 200
STEP = 50

# Ruta del modelo pre-entrenado
MODEL_PATH = 'mejor_modelo_base.h5'
NUM_CLASSES = 52
```

## 🏃‍♂️ Uso

### Ejecución básica:
```bash
python evaluation_cross_dataset.py
```

### Flujo de ejecución:

1. **Carga del modelo pre-entrenado**
   - Carga el modelo entrenado en DB1
   - Compila el modelo con optimizador Adam

2. **Procesamiento de datos DB3**
   - Carga datos de 11 sujetos amputados
   - Aplica preprocesamiento idéntico al entrenamiento
   - Crea ventanas temporales de 200 muestras

3. **Evaluación del modelo**
   - Realiza predicciones sobre DB3
   - Calcula precisión general
   - Genera matriz de confusión

4. **Visualización de resultados**
   - Muestra precisión en porcentaje
   - Genera heatmap de matriz de confusión

## 📈 Métricas de Evaluación

![Accuracy](https://img.shields.io/badge/Metric-Accuracy-blue)
![Confusion Matrix](https://img.shields.io/badge/Metric-Confusion_Matrix-green)

- **Precisión General**: Porcentaje de clasificaciones correctas
- **Matriz de Confusión**: Visualización de aciertos/errores por clase

## 🔧 Funciones Principales

### `load_ninapro_db3_data()`
![Function](https://img.shields.io/badge/Function-Data_Loading-ff69b4)
Carga datos EMG de un sujeto específico del DB3, combinando los 3 ejercicios disponibles.

### `create_windows_for_eval()`
![Function](https://img.shields.io/badge/Function-Windowing-9cf)
Crea ventanas temporales para evaluación sin aumentación de datos.

### Procesamiento de características:
- Normalización por ventana
- Exclusión de gestos de reposo (clase 0)
- Ajuste de etiquetas (resta 1 para compatibilidad)

## 📊 Salidas

### Resultados numéricos:
```
Precisión General del Modelo Base en NinaPro DB3: 68.45%
```

### Visualización:
- Matriz de confusión interactiva
- Heatmap con colores viridis
- Título con precisión general

## ⚠️ Consideraciones Importantes

![Warning](https://img.shields.io/badge/⚠️-Compatibility_Critical-red)

1. **Compatibilidad de preprocesamiento**: Los parámetros deben ser idénticos a los usados en entrenamiento
2. **Estructura de archivos**: Los archivos .mat deben seguir el naming convention de DB3
3. **Memoria**: Procesar 11 sujetos requiere memoria RAM suficiente
4. **Modelo**: El modelo debe estar entrenado con la misma arquitectura y número de clases

## 🎯 Aplicaciones

![Research](https://img.shields.io/badge/Application-Research-8A2BE2)
![HMI](https://img.shields.io/badge/Application-Human_Machine_Interface-20B2AA)

- Evaluación de robustez de modelos EMG
- Estudio de transfer learning entre poblaciones
- Benchmarking de algoritmos de clasificación
- Investigación en interfaces hombre-máquina

## 📄 Licencia

![License](https://img.shields.io/badge/License-MIT-lightgrey)

Este proyecto es para fines de investigación y educación. Los datasets NinaPro requieren citación apropiada.

## 🤝 Contribuciones

![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

Las contribuciones son bienvenidas para:
- Mejoras en la visualización
- Adición de más métricas de evaluación
- Soporte para otros datasets
- Optimización del procesamiento

---

**Nota**: Este código asume que el modelo pre-entrenado fue entrenado con la misma arquitectura y parámetros de preprocesamiento. Asegúrese de que `mejor_modelo_base.h5` existe en el directorio antes de ejecutar.
