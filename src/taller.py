"""
taller.py — Ingesta OPCIONAL de la planilla propia "Solicitud de Repuestos".

Formato transaccional: una fila por pedido (fecha, código, descripción,
cantidad). Es demanda de reparación que YA está incluida dentro de las ventas
mensuales AMyyyymm, por eso NO se suma a las ventas: se usa como SEÑAL para que
el modelo distinga los repuestos movidos por reparación de los de góndola.

Regla de oro: si la carpeta data/taller/ está vacía o el archivo cambia de
forma, el resto del sistema debe seguir funcionando igual. Nada acá puede
romper la lectura de la planilla AMyyyymm.
"""
import re

import numpy as np
import pandas as pd

from . import config

# Nombre de la hoja que importa (el resto se ignora). Match tolerante.
HOJA = "solicitud de repuestos"


def _leer_archivo(path):
    """Lee un archivo de taller (.xlsx con la hoja correcta, o .csv)."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    xl = pd.ExcelFile(path)
    hoja = next(
        (h for h in xl.sheet_names if str(h).strip().lower() == HOJA),
        xl.sheet_names[0],
    )
    return pd.read_excel(path, sheet_name=hoja)


def _parse_cantidad(valor):
    """Extrae la cantidad de textos sucios: '10', '2--', '50 UNIDADES',
    '1 CAJA'. Devuelve (numero_o_nan, es_caja). 'TODOS/TODAS' -> nan
    (se imputa después con el promedio del código)."""
    s = str(valor).upper().strip()
    m = re.search(r"\d+", s)
    n = float(m.group()) if m else np.nan
    return n, ("CAJA" in s)


def cargar(cant_master: dict | None = None) -> pd.DataFrame | None:
    """Demanda de taller agregada a (cod_articulo, fecha=mes). None si no hay
    archivos. `cant_master` (código -> unidades por caja) convierte los pedidos
    expresados en cajas a unidades."""
    archivos = sorted(config.RUTA_TALLER.glob("*.xls*")) + sorted(
        config.RUTA_TALLER.glob("*.csv")
    )
    if not archivos:
        return None

    frames = []
    for a in archivos:
        df = _leer_archivo(a).iloc[:, :4]
        df.columns = ["fecha", "cod_articulo", "descripcion", "cantidad"]
        frames.append(df)
    t = pd.concat(frames, ignore_index=True)

    t["cod_articulo"] = t["cod_articulo"].astype(str).str.strip()
    t["fecha"] = pd.to_datetime(t["fecha"], errors="coerce")
    t = t[(t["cod_articulo"].str.lower() != "nan") & t["fecha"].notna()].copy()

    parsed = [_parse_cantidad(v) for v in t["cantidad"]]
    t["cant"] = [p[0] for p in parsed]
    t["es_caja"] = [p[1] for p in parsed]

    # Cajas -> unidades con las unidades por caja del código (default 1)
    if cant_master:
        cm = t["cod_articulo"].map(cant_master).fillna(1).clip(lower=1)
        t.loc[t["es_caja"], "cant"] = t.loc[t["es_caja"], "cant"] * cm[t["es_caja"]]

    # 'TODOS/TODAS' sin número -> promedio del propio código; si el código no
    # tiene ningún dato numérico, cae a 1 (mínimo razonable).
    prom = t.groupby("cod_articulo")["cant"].transform("mean")
    t["cant"] = t["cant"].fillna(prom).fillna(1.0)

    t["fecha"] = t["fecha"].dt.to_period("M").dt.to_timestamp()
    mensual = (
        t.groupby(["cod_articulo", "fecha"])["cant"]
        .sum()
        .rename("demanda_taller")
        .reset_index()
    )
    return mensual
