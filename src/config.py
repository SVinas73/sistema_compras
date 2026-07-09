"""
config.py — Parámetros del negocio y del sistema (v2).
TODO lo ajustable vive acá: nunca toques la lógica para cambiar una política.
"""
from pathlib import Path

# ---------- Rutas ----------
RAIZ = Path(__file__).parent.parent
RUTA_RAW = RAIZ / "data" / "raw"
RUTA_PROCESSED = RAIZ / "data" / "processed"

# ---------- Negocio ----------
LEAD_TIME_MESES = 3
PERIODO_REVISION = 1
HORIZONTE = LEAD_TIME_MESES + PERIODO_REVISION   # 4 meses de protección

# Nivel de servicio por clase ABC → percentil del modelo quantile
# A: cubre el 95% de los escenarios | B: 90% | C: 80%
PERCENTIL_POR_CLASE = {"A": 0.95, "B": 0.90, "C": 0.80}

ABC_CORTE_A = 80
ABC_CORTE_B = 95

EXCLUIR = [
    "PMDSBD02-SP-P.9",
    "PMST01A-SP-HOOK12.J",
    "PMDSBD02-SP-P.11",
    "PMST01A-SP-HOOK7.J",
    "PMDSBD02-SP-P.10",
    "PMDSBD02-SP-C.B.1",
]

# ---------- Modelo (v2: features depuradas, medidas sobre datos reales) ----------
# Se quitaron lag_6 y lag_12 (con 17 meses de historia eran casi todo NaN
# y solo agregaban ruido). Se agregó "tendencia" (aceleración del SKU).
# Cuando haya 3+ años de historia: re-evaluar agregar lag_12.
# lag_6 y lag_12 reactivados al incorporar el histórico 2024
# (con 17 meses eran casi todo NaN; con 29 meses el lag_12 tiene dato
# en ~55% de las filas y es una de las features más usadas del modelo)
FEATURES = [
    # --- Pasado directo ---
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    # --- Nivel y volatilidad ---
    "media_movil_3", "media_movil_6", "media_movil_12", "std_movil_3",
    # --- Intermitencia / ciclo de vida (¿se apaga o despierta el SKU?) ---
    "meses_desde_ultima_venta", "tasa_actividad_6m", "tasa_actividad_12m",
    "ratio_actividad", "tendencia", "tendencia_larga",
    # --- Estacionalidad ---
    "mes", "mes_sin", "mes_cos", "indice_estacional",
]

# Hiperparámetros encontrados con Optuna (40 trials, validación temporal)
PARAMS_LGBM = dict(
    n_estimators=195,
    learning_rate=0.0194,
    num_leaves=31,
    min_child_samples=71,
    reg_lambda=9.93,
    subsample=0.62,
    colsample_bytree=0.62,
    subsample_freq=1,
    random_state=42,
    verbose=-1,
)

TARGET_MESES = 3

# Ensemble: cada modelo se entrena varias veces con distintas semillas
# y se promedian las predicciones. Reduce el ruido aleatorio del
# entrenamiento (medido: mejora MAE y calibración). Más semillas =
# más lento y más estable; 3 es buen equilibrio.
SEEDS = [42, 7, 123]
