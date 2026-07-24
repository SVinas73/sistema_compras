"""
inference.py — Fase 5: De la predicción a la orden de compra (v2.2).

COMPRAR = objetivo_de_cobertura (percentil según clase ABC, con tope
de sensatez) - posición de inventario, redondeado a cajas completas.

El reporte de salida incluye columnas de interpretación (probabilidad
de demanda, tendencia, meses sin venta, urgencia categórica) para que
el comprador entienda POR QUÉ el sistema recomienda cada compra.
"""
import numpy as np
import pandas as pd

from . import config
from .train import predecir, predecir_proba


def run(ult: pd.DataFrame, clfs, regs, quantiles):
    """Genera la orden de compra. Devuelve (orden, panel): la orden filtrada
    a lo que hay que comprar, y el panel completo de SKUs con todas las
    predicciones (lo consume el monitor de la fase 6)."""
    print("[FASE 5] Orden de compra...")
    ult = ult.copy()
    # Estiramiento de la predicción (3 meses) al horizonte: por SKU, según la
    # forma de la temporada de su familia (features.factor_horizonte). Si la
    # columna no está, cae al uniforme de siempre.
    factor = ult.get(
        "factor_horizonte",
        pd.Series(config.HORIZONTE / config.TARGET_MESES, index=ult.index),
    ).fillna(config.HORIZONTE / config.TARGET_MESES)

    # --- Predicciones del ensemble ---
    # Se guardan también las NATIVAS a 3 meses (sin estirar al horizonte):
    # son las que el monitor (fase 6) compara después contra la realidad.
    ult["prob_demanda"] = predecir_proba(clfs, ult[config.FEATURES])
    ult["pred_3m"] = ult["prob_demanda"] * predecir(regs, ult[config.FEATURES])
    ult["pred_esperada"] = ult["pred_3m"] * factor
    for alpha, modelos in quantiles.items():
        q3 = predecir(modelos, ult[config.FEATURES])
        ult[f"p{int(alpha*100)}_3m"] = q3
        ult[f"p{int(alpha*100)}"] = q3 * factor
    # Baseline ingenuo (últimos 3 meses proyectados): la vara a superar.
    ult["baseline_3m"] = (ult["media_movil_3"] * config.TARGET_MESES).fillna(0)

    # --- Clase ABC por ingresos ---
    ult["valor_abc"] = ult["ingresos"].fillna(0)
    ult = ult.sort_values("valor_abc", ascending=False).reset_index(drop=True)
    total = max(ult["valor_abc"].sum(), 1e-9)
    ult["pct_acum"] = ult["valor_abc"].cumsum() / total * 100
    ult["clase_abc"] = np.where(
        ult["pct_acum"] <= config.ABC_CORTE_A, "A",
        np.where(ult["pct_acum"] <= config.ABC_CORTE_B, "B", "C"),
    )

    # --- Objetivo de cobertura según clase + tope de sensatez ---
    ult["objetivo"] = ult.apply(
        lambda fila: fila[f"p{int(config.PERCENTIL_POR_CLASE[fila['clase_abc']] * 100)}"],
        axis=1,
    )
    # Tope de sensatez: el objetivo no puede superar lo MÁS que el SKU consumió
    # en una ventana del largo del horizonte (best rolling-HORIZONTE), por un
    # margen de seguridad. Frena la sobre-compra cuando el cuantil se dispara
    # por un pico reciente en items que casi no venden (ej. GMT55231: vendió 4
    # en 40 meses, el modelo pedía 6). Un item que crece de verdad tiene su
    # mejor ventana reciente y alta, así que el tope no lo limita.
    tope = (ult["max_horizonte_historico"] * config.TOPE_FACTOR).fillna(np.inf)
    ult["objetivo"] = np.minimum(ult["objetivo"], tope.clip(lower=1))
    ult["stock_seguridad"] = (ult["objetivo"] - ult["pred_esperada"]).clip(0).round(1)

    # --- Posición de inventario ---
    ult["posicion"] = (
        ult["stock"].fillna(0)
        + ult["stock_nodisp"].fillna(0)
        + ult["compras_encurso"].fillna(0)
        - ult["pedidos_pendientes"].fillna(0)
    )

    # --- Fórmula maestra y cajas ---
    ult["necesidad"] = ult["objetivo"] - ult["posicion"]
    cajas = ult["cant_master"].replace(0, 1).fillna(1)
    ult["comprar"] = (np.ceil(ult["necesidad"].clip(0) / cajas) * cajas).astype(int)

    # --- Urgencia ---
    dem_mensual = (ult["pred_esperada"] / config.HORIZONTE).replace(0, np.nan)
    ult["alcance_meses"] = (ult["posicion"] / dem_mensual).round(1)
    # Categoría legible: ¿llega la cobertura actual hasta que arribe el pedido?
    ult["urgencia"] = np.select(
        [
            ult["alcance_meses"] < 0,                      # ya debe unidades
            ult["alcance_meses"] < config.LEAD_TIME_MESES, # quiebra antes de que llegue
            ult["alcance_meses"] < config.HORIZONTE,       # justo
        ],
        ["CRITICO: sin stock y con pedidos", "ALTO: quiebra antes de que llegue el pedido",
         "MEDIO: cubre el viaje, sin margen"],
        default="BAJO: con cobertura",
    )
    ult["inversion_usd"] = (ult["comprar"] * ult["Fob"].fillna(0)).round(0)
    ult["prob_demanda"] = (ult["prob_demanda"] * 100).round(0)
    ult["tendencia"] = ult["tendencia"].round(2)
    # Señal de reparación para el comprador: unidades pedidas por taller en los
    # últimos 3 meses (vacío si no hay planilla de taller cargada).
    ult["pedidos_taller_3m"] = ult.get("taller_mm3", pd.Series(np.nan, index=ult.index)).round(0)

    # Desglose de la posición, para que se pueda VERIFICAR qué consideró el
    # modelo: posicion = stock_fisico + en_camino - pendiente_entregar.
    # 'en_camino' = compras_encurso (lo ya pedido al proveedor, que está por
    # llegar): el modelo lo descuenta para no volver a comprar lo que ya viene.
    ult["stock_fisico"] = ult["stock"].fillna(0) + ult["stock_nodisp"].fillna(0)
    ult["en_camino"] = ult["compras_encurso"].fillna(0)
    ult["pendiente_entregar"] = ult["pedidos_pendientes"].fillna(0)

    # Alerta de demanda oculta: si no hay stock y hace meses que "no vende",
    # esos ceros pueden ser venta PERDIDA (no había qué vender), no falta de
    # demanda. El modelo tiende a SUBestimar estos casos: revisarlos a mano.
    ult["alerta"] = np.where(
        (ult["stock_fisico"] <= 0)
        & (ult["en_camino"] <= 0)
        & (ult["meses_desde_ultima_venta"] >= 1),
        "SIN STOCK: meses sin venta pueden ser venta perdida",
        "",
    )

    orden = (
        ult[ult["comprar"] > 0]
        .sort_values(["alcance_meses", "inversion_usd"], ascending=[True, False])
        .reset_index(drop=True)
    )

    cols = ["cod_articulo", "nom_articulo", "clase_abc", "urgencia", "alerta",
            "comprar", "cant_master", "inversion_usd",
            "ventas_ult_3m", "pred_esperada", "objetivo", "stock_seguridad",
            "posicion", "stock_fisico", "en_camino", "pendiente_entregar",
            "alcance_meses", "prob_demanda", "tendencia", "pedidos_taller_3m",
            "meses_desde_ultima_venta", "Fob"]
    salida = config.RUTA_PROCESSED / "orden_de_compra.csv"
    orden[cols].round(1).to_csv(salida, index=False)

    resumen = orden.groupby(["urgencia"]).agg(
        SKUs=("comprar", "count"), USD=("inversion_usd", "sum"))
    print(f"  ✔ {len(orden)} SKUs a comprar | USD {orden['inversion_usd'].sum():,.0f}")
    print(resumen.to_string())
    print(f"  ✔ Exportado a {salida}")
    return orden, ult
