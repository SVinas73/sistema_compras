"""
monitor.py — Fase 6: Boletín de aprendizaje (retroalimentación).

Cada corrida hace dos cosas:
 1. EVALÚA las predicciones guardadas por corridas anteriores cuyos 3 meses
    objetivo ya se conocen, comparándolas contra las ventas reales:
      - MAE del modelo vs MAE del baseline ingenuo (¿aporta el modelo?)
      - cobertura real de cada percentil (¿el P90 cubre ~90%?)
      - sesgo (¿sobreestima o subestima?)
 2. REGISTRA las predicciones de hoy para que una corrida futura las evalúe.

Así el sistema se califica solo con cada planilla nueva: la confianza deja
de ser una opinión y pasa a ser una serie medida (historial/boletin.csv).
Nota: un corte se vuelve evaluable recién cuando los datos cubren sus 3
meses siguientes (el boletín corre naturalmente 3 meses detrás).
"""
import re

import pandas as pd

from . import config

RUTA_HISTORIAL = config.RUTA_PROCESSED / "historial"

# Columnas de contexto que viajan al snapshot si están disponibles.
# (prob_demanda NO: en el panel queda escalada 0-100 para el reporte y
# generaría snapshots con escalas mezcladas; ya está embebida en pred_3m.)
_EXTRAS = ["clase_abc", "comprar", "posicion"]


def _periodo(texto: str) -> pd.Period:
    """'202602' -> Period 2026-02 (parseo explícito, sin ambigüedad)."""
    return pd.Period(year=int(texto[:4]), month=int(texto[4:6]), freq="M")


def registrar(panel: pd.DataFrame, largo: pd.DataFrame) -> None:
    """Guarda la foto de predicciones del corte actual (idempotente: si se
    corre dos veces con la misma planilla, se sobreescribe la misma foto)."""
    RUTA_HISTORIAL.mkdir(parents=True, exist_ok=True)
    corte = pd.Period(largo["fecha"].max(), freq="M")
    cols_p = sorted(c for c in panel.columns if re.fullmatch(r"p\d+_3m", c))
    cols = ["cod_articulo", "pred_3m", "baseline_3m"] + cols_p + [
        c for c in _EXTRAS if c in panel.columns
    ]
    destino = RUTA_HISTORIAL / f"pred_{corte.strftime('%Y%m')}.parquet"
    panel[cols].to_parquet(destino, index=False)
    print(f"  ✔ Predicciones del corte {corte} registradas ({destino.name})")


def evaluar(largo: pd.DataFrame) -> None:
    """Compara cada snapshot con futuro ya conocido contra las ventas reales
    y reescribe historial/boletin.csv completo (idempotente)."""
    archivos = sorted(RUTA_HISTORIAL.glob("pred_*.parquet"))
    corte_actual = pd.Period(largo["fecha"].max(), freq="M")
    ventas = largo.assign(periodo=pd.PeriodIndex(largo["fecha"], freq="M"))

    filas = []
    for arch in archivos:
        corte = _periodo(arch.stem.replace("pred_", ""))
        meses_obj = [corte + k for k in range(1, config.TARGET_MESES + 1)]
        if meses_obj[-1] > corte_actual:
            continue  # su futuro todavía no se conoce completo
        snap = pd.read_parquet(arch)
        real = (
            ventas[ventas["periodo"].isin(meses_obj)]
            .groupby("cod_articulo")["ventas"].sum().rename("real")
        )
        df = snap.merge(real, on="cod_articulo", how="inner")
        if df.empty:
            continue
        fila = {
            "corte": str(corte),
            "evaluado_con_datos_hasta": str(corte_actual),
            "n_skus": len(df),
            "mae_modelo": (df["pred_3m"] - df["real"]).abs().mean(),
            "mae_baseline": (df["baseline_3m"] - df["real"]).abs().mean(),
            "sesgo": (df["pred_3m"] - df["real"]).mean(),
        }
        for c in sorted(c for c in df.columns if re.fullmatch(r"p\d+_3m", c)):
            fila[f"cobertura_{c[:-3]}"] = (df["real"] <= df[c]).mean()
        filas.append(fila)

    if not filas:
        print("  (aún sin predicciones evaluables: el primer boletín sale "
              "cuando una planilla nueva cubra los 3 meses posteriores a un registro)")
        return

    boletin = pd.DataFrame(filas)
    boletin.to_csv(RUTA_HISTORIAL / "boletin.csv", index=False)
    for _, f in boletin.tail(6).iterrows():
        gana = ("✓ le gana al baseline" if f["mae_modelo"] <= f["mae_baseline"]
                else "✗ pierde con el baseline")
        print(f"  Corte {f['corte']} ({int(f['n_skus'])} SKUs): "
              f"MAE modelo {f['mae_modelo']:.2f} vs baseline {f['mae_baseline']:.2f} "
              f"({gana}) | sesgo {f['sesgo']:+.2f}")
        cob = {k[len("cobertura_p"):]: v for k, v in f.items()
               if str(k).startswith("cobertura_")}
        print("    cobertura real: " + " | ".join(
            f"P{k}: {v * 100:.1f}%" for k, v in sorted(cob.items(), key=lambda x: int(x[0]))))
    print(f"  ✔ Boletín completo en {RUTA_HISTORIAL / 'boletin.csv'}")


def run(largo: pd.DataFrame, panel: pd.DataFrame) -> None:
    """Punto de entrada de la fase: primero evalúa lo viejo, después registra
    lo nuevo."""
    print("[FASE 6] Boletín de aprendizaje...")
    evaluar(largo)
    registrar(panel, largo)
