"""
extract.py — Fase 1: Ingesta.
Lee TODOS los .xls de data/raw, los unifica y devuelve
el histórico en formato largo (una fila por SKU-mes).
"""
import pandas as pd

from . import config

# Columnas de negocio que arrastramos junto al histórico.
# "familia" se deriva del código (prefijo antes de -SP- = modelo de máquina):
# agrupa repuestos que comparten estacionalidad y permite estimar la forma
# de la temporada con muchos más datos que SKU por SKU.
COLS_NEGOCIO = [
    "cod_articulo", "familia", "nom_articulo", "cant_master", "Fob", "ingresos",
    "stock", "stock_nodisp", "pedidos_pendientes", "compras_encurso",
]


def leer_planillas() -> pd.DataFrame:
    """Lee y concatena las planillas de ventas (formato AMyyyymm) de data/raw.
    Ignora otros Excel que puedan estar ahí (p.ej. el maestro de artículos),
    que no tienen ese formato."""
    archivos = sorted(config.RUTA_RAW.glob("*.xls*"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos Excel en {config.RUTA_RAW}")

    tablas = []
    for archivo in archivos:
        # engine según extensión: .xls viejo usa xlrd, .xlsx usa openpyxl
        engine = "xlrd" if archivo.suffix == ".xls" else "openpyxl"
        df = pd.read_excel(archivo, engine=engine, skiprows=4)
        # Solo planillas de ventas: deben tener cod_articulo y columnas AMyyyymm.
        tiene_am = any(str(c).startswith("AM2") for c in df.columns)
        if "cod_articulo" not in df.columns or not tiene_am:
            print(f"  - {archivo.name}: ignorado (no es planilla de ventas AMyyyymm)")
            continue
        tablas.append(df)

    if not tablas:
        raise FileNotFoundError(
            f"No hay planillas de ventas (formato AMyyyymm) en {config.RUTA_RAW}")
        print(f"  - {archivo.name}: {len(df)} filas")

    # --- Unificación correcta de múltiples archivos con períodos distintos ---
    # Datos de NEGOCIO (stock, pendientes, Fob...): del archivo MÁS NUEVO,
    # porque solo la foto más reciente es válida.
    # Datos de VENTAS (columnas AMyyyymm): la UNIÓN de todos los archivos;
    # si un mes aparece en dos archivos, gana el archivo más nuevo.
    base = tablas[-1].drop_duplicates(subset="cod_articulo", keep="last")
    base = base.set_index("cod_articulo")

    for tabla in reversed(tablas[:-1]):          # del más nuevo al más viejo
        tabla = tabla.drop_duplicates(subset="cod_articulo", keep="last")
        tabla = tabla.set_index("cod_articulo")
        cols_am_viejas = [c for c in tabla.columns if c.startswith("AM2")]
        # Meses que el archivo viejo tiene y la base todavía no
        nuevas = [c for c in cols_am_viejas if c not in base.columns]
        if nuevas:
            base = base.join(tabla[nuevas], how="left")
        # Meses compartidos: rellenar solo donde la base no tiene dato
        compartidas = [c for c in cols_am_viejas if c in base.columns]
        for c in compartidas:
            base[c] = base[c].fillna(tabla[c])
        # SKUs que existen solo en el archivo viejo: agregarlos enteros
        solo_viejo = tabla.index.difference(base.index)
        if len(solo_viejo):
            base = pd.concat([base, tabla.loc[solo_viejo]])

    df = base.reset_index()
    # Meses sin dato para un SKU (no existía aún) quedan en 0
    cols_am = [c for c in df.columns if c.startswith("AM2")]
    df[cols_am] = df[cols_am].fillna(0)

    # Exclusiones del negocio
    df = df[~df["cod_articulo"].isin(config.EXCLUIR)].copy()

    # Familia = modelo de máquina (prefijo antes de -SP-). Los SKUs sin ese
    # patrón quedan como su propio código (familia unipersonal).
    cod = df["cod_articulo"].astype(str)
    df["familia"] = cod.str.split("-SP-").str[0]
    return df


def a_formato_largo(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte las columnas AMyyyymm en filas (formato largo)."""
    cols_ventas = sorted(c for c in df.columns if c.startswith("AM2"))

    # Solo SKUs con al menos una venta en el período
    df["ventas_totales"] = df[cols_ventas].sum(axis=1)
    activos = df[df["ventas_totales"] > 0].copy()

    largo = activos[COLS_NEGOCIO + cols_ventas].melt(
        id_vars=COLS_NEGOCIO,
        value_vars=cols_ventas,
        var_name="periodo",
        value_name="ventas",
    )
    largo["fecha"] = pd.to_datetime(
        largo["periodo"].str.replace("AM", ""), format="%Y%m"
    )

    # Devoluciones: meses con ventas negativas (notas de crédito).
    # Para modelar DEMANDA las llevamos a 0: una devolución no es
    # demanda negativa futura, es un ajuste contable del pasado.
    largo["ventas"] = largo["ventas"].clip(lower=0)
    largo = largo.sort_values(["cod_articulo", "fecha"]).reset_index(drop=True)
    return largo


def run() -> pd.DataFrame:
    """Punto de entrada de la fase. Devuelve el formato largo y lo guarda."""
    from . import historico  # import local: evita ciclo (historico usa COLS_NEGOCIO)

    print("[FASE 1] Ingesta...")
    df = leer_planillas()
    largo = a_formato_largo(df)
    # Extiende la historia hacia atrás con el reporte EDINTOR (si está). No
    # toca los meses de la planilla AMyyyymm; solo agrega los previos.
    largo = historico.prepend(largo)
    largo.to_parquet(config.RUTA_PROCESSED / "ventas_largo.parquet", index=False)
    print(f"  ✔ {largo['cod_articulo'].nunique()} SKUs activos, {len(largo)} filas")
    return largo
