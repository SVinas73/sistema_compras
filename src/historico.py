"""
historico.py — Extiende la historia de ventas hacia atrás con el reporte
detallado de EDINTOR (data/historico/). OPCIONAL.

Las tres hojas que importan reconstruyen la demanda mensual por canal:
  VENTAREP = venta de repuestos al público
  CONSGAR  = repuestos consumidos en garantía
  CONSPRE  = repuestos consumidos en presupuesto

Medido sobre los meses en común: su SUMA ≈ las ventas mensuales AMyyyymm
(correlación 0.94; es el mismo dato, descompuesto por canal). Por eso NO se
suman a AM (sería doble conteo): se usan solo para los meses ANTERIORES a que
arranca la planilla AMyyyymm, agregando historia sin doble contar.

Si la carpeta está vacía, el sistema funciona igual.
"""
import pandas as pd

from . import config
from .extract import COLS_NEGOCIO

# hoja -> (columna_codigo, columna_cantidad, columna_fecha)
HOJAS = {
    "VENTAREP": ("Cod. Articulo", "Cantidad", "Fecha"),
    "CONSGAR": ("Codigo", "Cantidad", "Fecha"),
    "CONSPRE": ("Codigo", "Cantidad", "Fecha"),
}


def _demanda_mensual():
    """Demanda mensual por (cod_articulo, mes) = suma de los tres canales.
    None si no hay archivo."""
    archivos = sorted(config.RUTA_HISTORICO.glob("*.xls*"))
    if not archivos:
        return None

    partes = []
    for arch in archivos:
        xl = pd.ExcelFile(arch)
        for hoja, (cc, ca, cf) in HOJAS.items():
            if hoja not in xl.sheet_names:
                continue
            d = pd.read_excel(arch, sheet_name=hoja)
            d.columns = [str(c).strip() for c in d.columns]
            if not {cc, ca, cf} <= set(d.columns):
                continue
            partes.append(pd.DataFrame({
                "cod_articulo": d[cc].astype(str).str.strip(),
                "ventas": pd.to_numeric(d[ca], errors="coerce"),
                "fecha": pd.to_datetime(d[cf], errors="coerce"),
            }).dropna(subset=["cod_articulo", "fecha"]))

    if not partes:
        return None
    todo = pd.concat(partes, ignore_index=True)
    todo["ventas"] = todo["ventas"].clip(lower=0)
    todo["fecha"] = todo["fecha"].dt.to_period("M").dt.to_timestamp()
    return todo.groupby(["cod_articulo", "fecha"])["ventas"].sum().reset_index()


def prepend(largo: pd.DataFrame) -> pd.DataFrame:
    """Agrega los meses ANTERIORES a la planilla AMyyyymm (para los SKUs
    activos), reconstruidos del reporte EDINTOR. Sin archivo, devuelve largo
    igual."""
    dem = _demanda_mensual()
    if dem is None:
        return largo

    inicio_am = largo["fecha"].min()
    dem = dem[dem["fecha"] < inicio_am]              # solo meses previos a AM
    activos = sorted(largo["cod_articulo"].unique())
    dem = dem[dem["cod_articulo"].isin(activos)]
    if dem.empty:
        return largo

    meses = sorted(dem["fecha"].unique())
    # Datos de negocio (stock, familia, cant_master...) constantes por SKU:
    # se copian de la foto actual, igual que en las filas de la planilla AM.
    neg_cols = [c for c in COLS_NEGOCIO if c != "cod_articulo"]
    neg = largo.drop_duplicates("cod_articulo").set_index("cod_articulo")[neg_cols]

    grid = pd.MultiIndex.from_product(
        [activos, meses], names=["cod_articulo", "fecha"]
    ).to_frame(index=False)
    grid = grid.merge(dem, on=["cod_articulo", "fecha"], how="left")
    grid["ventas"] = grid["ventas"].fillna(0.0)
    grid = grid.join(neg, on="cod_articulo")
    grid["periodo"] = "AM" + grid["fecha"].dt.strftime("%Y%m")

    out = pd.concat([grid[largo.columns], largo], ignore_index=True)
    out = out.sort_values(["cod_articulo", "fecha"]).reset_index(drop=True)
    print(f"  ✔ histórico EDINTOR: +{len(meses)} meses previos "
          f"({pd.Timestamp(meses[0]).date()} .. {pd.Timestamp(meses[-1]).date()})")
    return out
