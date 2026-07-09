"""
features.py — Fase 3: Feature Engineering.
Crea las variables que le dan "memoria" al modelo y el target.
Regla sagrada: ninguna feature puede mirar el futuro (data leakage).
"""
import numpy as np
import pandas as pd

from . import config


def tabla_estacional(df: pd.DataFrame):
    """Estima la forma de la temporada por FAMILIA (modelo de máquina).

    Para cada (familia, mes) calcula cuánto se vende ese mes respecto del
    promedio anual de la familia: >1 = mes fuerte, <1 = mes flojo. Agrupar
    por familia pool­ea decenas de SKUs, así la temporada se estima con datos
    de sobra aunque cada SKU tenga solo ~2 años.

    Devuelve (indice_familia, indice_global, conteo) para poder aplicar el
    índice de familia donde hay evidencia y caer al global donde no la hay.
    Se calcula SOLO sobre las filas que se le pasan: en validación se le
    pasa el train, evitando mirar el futuro.
    """
    glob = df.groupby("mes")["ventas"].mean()
    indice_global = (glob / max(glob.mean(), 1e-9)).clip(0.3, 3.0)

    fam_mes = df.groupby(["familia", "mes"])["ventas"].mean()
    fam = df.groupby("familia")["ventas"].mean().replace(0, np.nan)
    indice_familia = (fam_mes.div(fam, level="familia")).clip(0.3, 3.0)
    conteo = df.groupby(["familia", "mes"])["ventas"].size()
    return indice_familia, indice_global, conteo


def aplicar_estacional(df, indice_familia, indice_global, conteo, min_n=8):
    """Agrega la columna indice_estacional usando familia si hay evidencia
    suficiente (>= min_n observaciones en esa celda familia-mes), si no el
    índice global del mes, y 1.0 (neutro) como último recurso."""
    clave = list(zip(df["familia"], df["mes"]))
    idx_fam = pd.Series(indice_familia.reindex(clave).values, index=df.index)
    n = pd.Series(conteo.reindex(clave).values, index=df.index)
    idx_glob = df["mes"].map(indice_global)

    out = idx_fam.where(n >= min_n)
    out = out.fillna(idx_glob).fillna(1.0)
    df["indice_estacional"] = out.values
    return df


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
    largo["media_movil_12"] = g.transform(lambda s: s.shift(1).rolling(12).mean())
    largo["std_movil_3"] = g.transform(lambda s: s.shift(1).rolling(3).std())

    # --- Intermitencia y actividad ---
    largo["meses_desde_ultima_venta"] = g.transform(_meses_sin_venta)
    tasa_3m = g.transform(lambda s: (s.shift(1) > 0).rolling(3).mean())
    largo["tasa_actividad_6m"] = g.transform(
        lambda s: (s.shift(1) > 0).rolling(6).mean()
    )
    largo["tasa_actividad_12m"] = g.transform(
        lambda s: (s.shift(1) > 0).rolling(12).mean()
    )

    # --- Ciclo de vida: ¿el SKU se está apagando? ---
    # ratio_actividad < 1 = vende menos meses que en su último año → enfriándose
    # o muriendo (el caso "vendió un año y al siguiente no"). > 1 = despertando.
    largo["ratio_actividad"] = tasa_3m / largo["tasa_actividad_12m"].replace(0, np.nan)

    # --- Tendencia de nivel: ¿acelera o se apaga? ---
    # corta (3 vs 6 meses) y larga (3 vs 12 meses): detecta el "vendió 3 meses
    # y paró" (corta) y el declive lento de todo un año (larga).
    largo["tendencia"] = largo["media_movil_3"] / largo["media_movil_6"].replace(0, np.nan)
    largo["tendencia_larga"] = largo["media_movil_3"] / largo["media_movil_12"].replace(0, np.nan)

    # --- Calendario (cíclico: dic y ene quedan "cerca") ---
    largo["mes"] = largo["fecha"].dt.month
    largo["mes_sin"] = np.sin(2 * np.pi * largo["mes"] / 12)
    largo["mes_cos"] = np.cos(2 * np.pi * largo["mes"] / 12)

    # --- Estacionalidad por familia (índice de temporada) ---
    idx_fam, idx_glob, cnt = tabla_estacional(largo)
    largo = aplicar_estacional(largo, idx_fam, idx_glob, cnt)

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
        familia=("familia", "last"),
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
        media_movil_12=("ventas", lambda s: s.iloc[-12:].mean()),
        std_movil_3=("ventas", lambda s: s.iloc[-3:].std()),
        tasa_actividad_6m=("ventas", lambda s: (s.iloc[-6:] > 0).mean()),
        tasa_actividad_12m=("ventas", lambda s: (s.iloc[-12:] > 0).mean()),
        _tasa_3m=("ventas", lambda s: (s.iloc[-3:] > 0).mean()),
        # Mejor trimestre histórico: la mayor demanda real que el SKU
        # mostró en 3 meses seguidos. Base del tope de sensatez.
        max_trim_historico=("ventas", lambda s: s.rolling(3).sum().max()),
    ).reset_index()

    ult["meses_desde_ultima_venta"] = (
        largo.groupby("cod_articulo")["ventas"].apply(msv_hoy).values
    )

    # Tendencias y ciclo de vida al día de hoy
    ult["tendencia"] = ult["media_movil_3"] / ult["media_movil_6"].replace(0, np.nan)
    ult["tendencia_larga"] = ult["media_movil_3"] / ult["media_movil_12"].replace(0, np.nan)
    ult["ratio_actividad"] = ult["_tasa_3m"] / ult["tasa_actividad_12m"].replace(0, np.nan)

    # El mes desde el que se predice = el mes siguiente al último dato
    ultima_fecha = largo["fecha"].max()
    ult["mes"] = (ultima_fecha + pd.DateOffset(months=1)).month
    ult["mes_sin"] = np.sin(2 * np.pi * ult["mes"] / 12)
    ult["mes_cos"] = np.cos(2 * np.pi * ult["mes"] / 12)

    # Índice estacional de la familia para el mes que se va a predecir.
    # Se estima con TODO el histórico (misma lógica que en entrenamiento).
    largo_mes = largo.assign(mes=largo["fecha"].dt.month)
    idx_fam, idx_glob, cnt = tabla_estacional(largo_mes)
    ult = aplicar_estacional(ult, idx_fam, idx_glob, cnt)
    return ult
