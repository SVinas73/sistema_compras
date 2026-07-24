"""
features.py — Fase 3: Feature Engineering.
Crea las variables que le dan "memoria" al modelo y el target.
Regla sagrada: ninguna feature puede mirar el futuro (data leakage).
"""
import holidays as _holidays
import numpy as np
import pandas as pd

from . import config, taller


def _tabla_calendario(periodos):
    """Días hábiles (lun-vie sin feriados) y feriados de cada mes (Period)."""
    hab, fer = {}, {}
    for p in periodos:
        h = _holidays.Uruguay(years=p.year)
        dias = pd.date_range(p.start_time, p.end_time, freq="D")
        habil = [(d.weekday() < 5) and (d.date() not in h) for d in dias]
        feriado = [(d.weekday() < 5) and (d.date() in h) for d in dias]
        hab[p], fer[p] = int(sum(habil)), int(sum(feriado))
    return hab, fer


def _agregar_calendario(largo: pd.DataFrame) -> pd.DataFrame:
    """habiles_objetivo / feriados_objetivo: días hábiles y feriados de los 3
    meses que se van a predecir (t+1..t+3). Determinista y conocido de
    antemano → sin fuga de datos. Un período con feriados/menos días hábiles
    tiende a vender menos."""
    fechas = pd.PeriodIndex(pd.to_datetime(largo["fecha"]).dt.to_period("M")).unique()
    needed = {p + k for p in fechas for k in (1, 2, 3)}
    hab, fer = _tabla_calendario(sorted(needed))
    map_h = {p.to_timestamp(): sum(hab[p + k] for k in (1, 2, 3)) for p in fechas}
    map_f = {p.to_timestamp(): sum(fer[p + k] for k in (1, 2, 3)) for p in fechas}
    largo["habiles_objetivo"] = largo["fecha"].map(map_h)
    largo["feriados_objetivo"] = largo["fecha"].map(map_f)
    return largo


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


def factor_horizonte(familias, p0, indice_familia, indice_global, conteo, min_n=8):
    """Cuánto estirar la predicción de TARGET_MESES para cubrir el HORIZONTE,
    respetando la forma de la temporada de cada familia.

    El uniforme HORIZONTE/TARGET (ej. 5/3) asume que los meses 4-5 venden al
    mismo ritmo que los 3 predichos; para un producto de invierno comprado en
    mayo eso infla el objetivo (sep-oct venden poco) y al revés lo achica.
    Acá: suma de índices estacionales de los meses del horizonte / suma de los
    del target. Con estacionalidad plana equivale exactamente al uniforme.
    Medido sobre el histórico: reduce ~9% el error de esa extrapolación
    (~15% en SKUs estacionales).
    """
    iff, cc, ig = indice_familia.to_dict(), conteo.to_dict(), indice_global.to_dict()
    meses = [(p0 + k).month for k in range(1, config.HORIZONTE + 1)]

    def _idx(fam, mes):
        v, n = iff.get((fam, mes)), cc.get((fam, mes), 0)
        if v is not None and not np.isnan(v) and n >= min_n:
            return v
        return ig.get(mes, 1.0)

    factores = {}
    for fam in familias.unique():
        indices = [_idx(fam, m) for m in meses]
        den = sum(indices[: config.TARGET_MESES])
        factores[fam] = sum(indices) / max(den, 1e-9)

    # Sensatez del propio factor: como mínimo 1 (los meses extra no venden
    # nada) y como máximo 2x el uniforme (los meses extra son doblemente
    # fuertes). El tope de max_trim_historico sigue aplicando después.
    maximo = 2 * config.HORIZONTE / config.TARGET_MESES
    return familias.map(factores).clip(1.0, maximo)


def _taller_mensual(largo: pd.DataFrame):
    """Carga la demanda de taller alineada al panel y devuelve (serie mensual
    por SKU, mes desde el que taller está activo). Devuelve (None, None) si no
    hay planilla de taller."""
    cant_master = largo.groupby("cod_articulo")["cant_master"].last().to_dict()
    tall = taller.cargar(cant_master)
    if tall is None or tall.empty:
        return None, None
    # Mes desde el que taller se registra en serio (>=5 pedidos en el mes):
    # antes de eso "sin dato" es desconocido, no un cero real.
    por_mes = tall.groupby("fecha").size()
    activos = por_mes[por_mes >= 5].index
    inicio = activos.min() if len(activos) else tall["fecha"].min()
    return tall, inicio


def _agregar_taller(largo: pd.DataFrame) -> pd.DataFrame:
    """Agrega taller_mm3: demanda de reparación de los 3 meses previos.
    NaN antes de que taller empiece a registrarse (dato desconocido, no cero)."""
    tall, inicio = _taller_mensual(largo)
    if tall is None:
        largo["taller_mm3"] = np.nan
        return largo

    largo = largo.merge(tall, on=["cod_articulo", "fecha"], how="left")
    # Dentro del período activo, "sin registro" = 0 pedidos reales.
    en_rango = largo["fecha"] >= inicio
    largo.loc[en_rango, "demanda_taller"] = largo.loc[en_rango, "demanda_taller"].fillna(0)
    largo = largo.sort_values(["cod_articulo", "fecha"]).reset_index(drop=True)

    gt = largo.groupby("cod_articulo")["demanda_taller"]
    largo["taller_mm3"] = gt.transform(lambda s: s.shift(1).rolling(3).sum())
    largo = largo.drop(columns=["demanda_taller"])
    return largo


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
    # Normalizar orden e índice ANTES de todo: los transform de abajo se
    # asignan por índice, y los merges (taller/calendario) lo reinician.
    # Sin esta línea, un frame filtrado (índice con huecos) desalinearía el
    # target EN SILENCIO: cada fila recibiría el futuro de otro SKU.
    # (Detectado por el boletín de aprendizaje en un backtest.)
    largo = largo.sort_values(["cod_articulo", "fecha"]).reset_index(drop=True)
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
    # y paró" (corta) y el declive lento de todo un año (larga). El tope de
    # sensatez (max_horizonte_historico) se aplica después en inference.
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
    # OJO: calcularlo ANTES de los merges de taller/calendario, que crean un
    # frame nuevo; g quedó atado al índice original y una asignación después
    # de un merge podría desalinear target y features.
    n = config.TARGET_MESES
    largo[f"demanda_{n}m"] = g.transform(
        lambda s: s.shift(-1).rolling(n).sum().shift(-(n - 1))
    )

    # --- Demanda de taller (reparación) ---
    largo = _agregar_taller(largo)

    # --- Calendario del período a predecir (días hábiles / feriados) ---
    largo = _agregar_calendario(largo)

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
        # Ventas reales de los últimos 3 meses: el criterio "a ojo" del
        # comprador, para poder compararlo con la recomendación del modelo.
        ventas_ult_3m=("ventas", lambda s: s.iloc[-3:].sum()),
        std_movil_3=("ventas", lambda s: s.iloc[-3:].std()),
        tasa_actividad_6m=("ventas", lambda s: (s.iloc[-6:] > 0).mean()),
        tasa_actividad_12m=("ventas", lambda s: (s.iloc[-12:] > 0).mean()),
        _tasa_3m=("ventas", lambda s: (s.iloc[-3:] > 0).mean()),
        # Máxima demanda real del SKU en una ventana del largo del horizonte
        # de protección. Base del tope de sensatez (evita sobre-comprar por
        # picos que el cuantil extrapola de más).
        max_horizonte_historico=("ventas",
                                 lambda s: s.rolling(config.HORIZONTE).sum().max()),
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

    # Factor estacional para estirar la predicción de 3 meses al horizonte
    # completo (ver factor_horizonte). p0 = último mes con datos.
    p0 = pd.Period(ultima_fecha, "M")
    ult["factor_horizonte"] = factor_horizonte(
        ult["familia"], p0, idx_fam, idx_glob, cnt
    )

    # Demanda de taller de los últimos 3 meses hasta hoy (misma ventana que
    # taller_mm3 en entrenamiento).
    tall, inicio = _taller_mensual(largo)
    if tall is not None:
        fin = largo["fecha"].max()
        ventana = tall[
            (tall["fecha"] <= fin) & (tall["fecha"] > fin - pd.DateOffset(months=3))
        ]
        tmm3 = ventana.groupby("cod_articulo")["demanda_taller"].sum()
        relleno = 0.0 if fin >= inicio else np.nan
        ult["taller_mm3"] = ult["cod_articulo"].map(tmm3).fillna(relleno)
    else:
        ult["taller_mm3"] = np.nan

    # Calendario de los próximos 3 meses (igual que en entrenamiento)
    p0 = pd.Period(largo["fecha"].max(), "M")
    hab, fer = _tabla_calendario([p0 + k for k in (1, 2, 3)])
    ult["habiles_objetivo"] = sum(hab[p0 + k] for k in (1, 2, 3))
    ult["feriados_objetivo"] = sum(fer[p0 + k] for k in (1, 2, 3))
    return ult
