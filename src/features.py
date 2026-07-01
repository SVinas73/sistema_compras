"""
features.py — Fase 3: Feature Engineering.
Crea las variables que le dan "memoria" al modelo y el target.
Regla sagrada: ninguna feature puede mirar el futuro (data leakage).
"""
import numpy as np
import pandas as pd

from . import config


def _meses_sin_venta(s: pd.Series) -> pd.Series:
    """Racha de meses consecutivos sin vender, mirando solo el pasado."""
    resultado, contador = [], np.nan
    for v in s.shift(1):  # shift(1): solo info del mes anterior hacia atrás
        if np.isnan(v):
            resultado.append(np.nan)
        elif v > 0:
            contador = 0
            resultado.append(0)
        else:
            contador = (0 if np.isnan(contador) else contador) + 1
            resultado.append(contador)
    return pd.Series(resultado, index=s.index)


def construir(largo: pd.DataFrame) -> pd.DataFrame:
    """Agrega todas las features y el target al formato largo."""
    print("[FASE 3] Feature engineering...")
    g = largo.groupby("cod_articulo")["ventas"]

    # --- Lags: el pasado directo ---
    for lag in (1, 2, 3, 6, 12):
        largo[f"lag_{lag}"] = g.shift(lag)

    # --- Rolling: nivel y volatilidad recientes ---
    # shift(1) ANTES de rolling: la ventana no debe incluir el mes actual
    largo["media_movil_3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    largo["media_movil_6"] = g.transform(lambda s: s.shift(1).rolling(6).mean())
    largo["std_movil_3"] = g.transform(lambda s: s.shift(1).rolling(3).std())

    # --- Intermitencia ---
    largo["meses_desde_ultima_venta"] = g.transform(_meses_sin_venta)
    largo["tasa_actividad_6m"] = g.transform(
        lambda s: (s.shift(1) > 0).rolling(6).mean()
    )

    # --- Tendencia: ¿el SKU acelera o se apaga? ---
    # >1 = vendiendo por encima de su nivel de 6 meses; <1 = enfriándose
    largo["tendencia"] = largo["media_movil_3"] / largo["media_movil_6"].replace(0, np.nan)

    # --- Calendario ---
    largo["mes"] = largo["fecha"].dt.month

    # --- Target: demanda acumulada de los próximos N meses ---
    n = config.TARGET_MESES
    largo[f"demanda_{n}m"] = g.transform(
        lambda s: s.shift(-1).rolling(n).sum().shift(-(n - 1))
    )

    # Filas entrenables: con historia suficiente y con futuro conocido
    dataset = largo.dropna(
        subset=["lag_3", "media_movil_3", f"demanda_{n}m"]
    ).copy()
    dataset["habra_demanda"] = (dataset[f"demanda_{n}m"] > 0).astype(int)

    dataset.to_parquet(
        config.RUTA_PROCESSED / "dataset_entrenamiento.parquet", index=False
    )
    print(f"  ✔ {len(dataset)} filas entrenables")
    return dataset


def foto_actual(largo: pd.DataFrame) -> pd.DataFrame:
    """Una fila por SKU con las features calculadas AL DÍA DE HOY,
    para alimentar la predicción de producción."""
    def msv_hoy(s):
        for i, v in enumerate(s.values[::-1]):
            if v > 0:
                return i
        return len(s)

    ult = largo.groupby("cod_articulo").agg(
        nom_articulo=("nom_articulo", "last"),
        cant_master=("cant_master", "last"),
        Fob=("Fob", "last"),
        ingresos=("ingresos", "last"),
        stock=("stock", "last"),
        stock_nodisp=("stock_nodisp", "last"),
        pedidos_pendientes=("pedidos_pendientes", "last"),
        compras_encurso=("compras_encurso", "last"),
        lag_1=("ventas", lambda s: s.iloc[-1]),
        lag_2=("ventas", lambda s: s.iloc[-2] if len(s) > 1 else np.nan),
        lag_3=("ventas", lambda s: s.iloc[-3] if len(s) > 2 else np.nan),
        lag_6=("ventas", lambda s: s.iloc[-6] if len(s) > 5 else np.nan),
        lag_12=("ventas", lambda s: s.iloc[-12] if len(s) > 11 else np.nan),
        media_movil_3=("ventas", lambda s: s.iloc[-3:].mean()),
        media_movil_6=("ventas", lambda s: s.iloc[-6:].mean()),
        std_movil_3=("ventas", lambda s: s.iloc[-3:].std()),
        tasa_actividad_6m=("ventas", lambda s: (s.iloc[-6:] > 0).mean()),
        # Mejor trimestre histórico: la mayor demanda real que el SKU
        # mostró en 3 meses seguidos. Base del tope de sensatez.
        max_trim_historico=("ventas", lambda s: s.rolling(3).sum().max()),
    ).reset_index()

    ult["meses_desde_ultima_venta"] = (
        largo.groupby("cod_articulo")["ventas"].apply(msv_hoy).values
    )

    # Tendencia al día de hoy
    ult["tendencia"] = ult["media_movil_3"] / ult["media_movil_6"].replace(0, np.nan)

    # El mes desde el que se predice = el mes siguiente al último dato
    ultima_fecha = largo["fecha"].max()
    ult["mes"] = (ultima_fecha + pd.DateOffset(months=1)).month
    return ult
