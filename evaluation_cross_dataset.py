# =============================================================================
# SECCIÓN 1: IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import os
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

# =============================================================================
# SECCIÓN 2: CONFIGURACIÓN Y PARÁMETROS GLOBALES
# =============================================================================
# --- Parámetros de datos (DB3) ---
DATASET_PATH_DB3 = 'ninapro_db3_data'
SUBJECTS_TO_PROCESS_DB3 = list(range(1, 12))  # 11 sujetos en DB3

# --- Parámetros de preprocesamiento (DEBEN SER IDÉNTICOS A LOS DEL ENTRENAMIENTO) ---
WINDOW_SIZE = 200
STEP = 50

# --- Parámetros del modelo ---
MODEL_PATH = 'mejor_modelo_base.h5'  # Se define aquí, se usa en el main
NUM_CLASSES = 52


# =============================================================================
# SECCIÓN 3: FUNCIONES DE CARGA Y PREPROCESAMIENTO (ADAPTADAS PARA DB3)
# =============================================================================
# (Las funciones load_ninapro_db3_data y create_windows_for_eval se mantienen exactamente igual que en la versión corregida anterior)
def load_ninapro_db3_data(base_path, subject):
    """Carga los datos de un sujeto de DB3, iterando a través de los 3 ejercicios."""
    all_emg = np.array([])
    all_gestures = np.array([])
    exercises_in_db3 = [1, 2, 3]
    for exercise in exercises_in_db3:
        file_path = os.path.join(base_path, f'S{subject}_E{exercise}_A1.mat')
        try:
            data = loadmat(file_path)
            if all_emg.size == 0:
                all_emg, all_gestures = data['emg'], data['restimulus']
            else:
                all_emg = np.vstack((all_emg, data['emg']))
                all_gestures = np.vstack((all_gestures, data['restimulus']))
        except FileNotFoundError:
            print(f"¡ADVERTENCIA! No se encontró el archivo: {file_path}.")
            continue
    if all_emg.size == 0: return None, None
    return all_emg, all_gestures


def create_windows_for_eval(emg, gestures, window_size, step):
    """Crea ventanas para evaluación (sin aumentación)."""
    X, y = [], []
    active_indices = np.where(gestures.flatten() != 0)[0]
    for i in range(0, len(active_indices) - window_size, step):
        window_indices = active_indices[i: i + window_size]
        if window_indices[-1] - window_indices[0] != window_size - 1: continue
        window_emg = emg[window_indices]
        label = np.bincount(gestures[window_indices].flatten()).argmax()
        mean, std = np.mean(window_emg, axis=0), np.std(window_emg, axis=0)
        window_normalized = (window_emg - mean) / (std + 1e-8)
        X.append(window_normalized)
        y.append(label)
    return np.array(X), np.array(y)


# =============================================================================
# SECCIÓN 4: SCRIPT PRINCIPAL DE EVALUACIÓN
# =============================================================================
if __name__ == "__main__":
    # 1. Cargar el modelo base pre-entrenado
    print(f"Cargando modelo desde: {MODEL_PATH}")
    try:
        # --- ¡CORRECCIÓN AQUÍ! ---
        # Se aplica la solución alternativa de cargar sin compilar y luego compilar
        model = load_model(MODEL_PATH, compile=False)

        # Compilamos manualmente el modelo después de cargarlo
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("\nModelo cargado y compilado exitosamente.")
        model.summary()

    except Exception as e:
        print(f"Error al cargar o compilar el modelo: {e}")
        # Si esto falla, recuerda probar la solución principal: pip install h5py==3.1.0
        exit()

    # (El resto del script, del paso 2 al 5, se mantiene exactamente igual)

    # 2. Cargar y procesar los datos de DB3 para todos los sujetos
    print("\n--- Cargando y procesando datos de NinaPro DB3 ---")
    all_X_db3 = []
    all_y_db3 = []
    for subject_id in SUBJECTS_TO_PROCESS_DB3:
        print(f"Procesando Sujeto {subject_id}/{len(SUBJECTS_TO_PROCESS_DB3)} de DB3...")
        emg_signals, gesture_labels = load_ninapro_db3_data(DATASET_PATH_DB3, subject_id)
        if emg_signals is None: continue

        X, y = create_windows_for_eval(emg_signals, gesture_labels, WINDOW_SIZE, STEP)
        all_X_db3.append(X)
        all_y_db3.append(y)

    X_db3 = np.concatenate(all_X_db3, axis=0)
    y_db3_true = np.concatenate(all_y_db3, axis=0)

    y_db3_true = y_db3_true - 1

    X_db3 = X_db3[:, :, :10]

    print(f"\nDatos de DB3 listos para evaluación.")
    print(f"Forma total de X_db3: {X_db3.shape}")
    print(f"Forma total de y_db3_true: {y_db3_true.shape}")

    # 3. Realizar predicciones
    print("\nRealizando predicciones en el conjunto de datos de amputados (DB3)...")
    y_db3_pred_probs = model.predict(X_db3)
    y_db3_pred_labels = np.argmax(y_db3_pred_probs, axis=1)

    # 4. Calcular y mostrar métricas
    print("\n--- Resultados de la Evaluación Cruzada (DB1 -> DB3) ---")
    accuracy = accuracy_score(y_db3_true, y_db3_pred_labels)
    print(f"Precisión General del Modelo Base en NinaPro DB3: {accuracy * 100:.2f}%")

    # 5. Generar y visualizar la Matriz de Confusión
    print("\nGenerando matriz de confusión...")
    conf_matrix = confusion_matrix(y_db3_true, y_db3_pred_labels)

    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='viridis')
    plt.title(f'Matriz de Confusión - Modelo (Entrenado en DB1) vs. Datos (DB3)\nPrecisión: {accuracy * 100:.2f}%')
    plt.xlabel('Etiqueta Predicha')
    plt.ylabel('Etiqueta Verdadera')
    plt.show()
