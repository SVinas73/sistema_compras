"""
train.py — Fase 4: Modelado (v2.2, con ensemble de semillas).
Cada modelo se entrena len(SEEDS) veces y las predicciones se promedian:
reduce la varianza del entrenamiento (como pedir la opinión de varios
peritos en vez de uno). Medido: mejora MAE y deja la calibración de los
percentiles prácticamente en su valor nominal.
"""
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from . import config, features

TARGET = f"demanda_{config.TARGET_MESES}m"
PERCENTILES = sorted(set(config.PERCENTIL_POR_CLASE.values()))


def _params(seed):
    p = dict(config.PARAMS_LGBM)
    p["random_state"] = seed
    return p


def _fit_ensemble(modelo_cls, X, y, **kw):
    """Entrena una lista de modelos idénticos salvo la semilla."""
    return [modelo_cls(**_params(s), **kw).fit(X, y) for s in config.SEEDS]


def predecir_proba(modelos, X):
    """Promedio de probabilidades del ensemble de clasificadores."""
    return np.mean([m.predict_proba(X)[:, 1] for m in modelos], axis=0)


def predecir(modelos, X):
    """Promedio de predicciones del ensemble de regresores."""
    return np.mean([m.predict(X) for m in modelos], axis=0).clip(0)


def validar(dataset: pd.DataFrame, meses_test: int = 3) -> None:
    fechas = sorted(dataset["fecha"].unique())
    corte = fechas[-meses_test]
    train = dataset[dataset["fecha"] < corte].copy()
    test = dataset[dataset["fecha"] >= corte].copy()

    # Índice estacional recalculado SOLO con el train, para que la validación
    # no mire el futuro (en producción se estima con todo el histórico).
    idx_fam, idx_glob, cnt = features.tabla_estacional(train)
    train = features.aplicar_estacional(train, idx_fam, idx_glob, cnt)
    test = features.aplicar_estacional(test, idx_fam, idx_glob, cnt)

    clfs = _fit_ensemble(lgb.LGBMClassifier, train[config.FEATURES], train["habra_demanda"])
    proba = predecir_proba(clfs, test[config.FEATURES])

    pos = train[train[TARGET] > 0]
    regs = _fit_ensemble(lgb.LGBMRegressor, pos[config.FEATURES], pos[TARGET], objective="tweedie")
    pred = proba * predecir(regs, test[config.FEATURES])

    auc = roc_auc_score(test["habra_demanda"], proba)
    mae = mean_absolute_error(test[TARGET], pred)
    baseline = mean_absolute_error(
        test[TARGET], (test["media_movil_3"] * config.TARGET_MESES).fillna(0)
    )
    print(f"  [validación] AUC clasificador: {auc:.3f}")
    print(f"  [validación] MAE modelo: {mae:.2f} | MAE baseline: {baseline:.2f}")

    for alpha in PERCENTILES:
        qs = _fit_ensemble(lgb.LGBMRegressor, train[config.FEATURES], train[TARGET],
                           objective="quantile", alpha=alpha)
        cob = (test[TARGET] <= predecir(qs, test[config.FEATURES])).mean()
        print(f"  [validación] P{int(alpha*100)}: cobertura real {cob*100:.1f}%")


def run(dataset: pd.DataFrame, con_validacion: bool = True):
    print("[FASE 4] Modelado (ensemble de {} semillas)...".format(len(config.SEEDS)))
    if con_validacion:
        validar(dataset)

    clfs = _fit_ensemble(lgb.LGBMClassifier, dataset[config.FEATURES], dataset["habra_demanda"])
    pos = dataset[dataset[TARGET] > 0]
    regs = _fit_ensemble(lgb.LGBMRegressor, pos[config.FEATURES], pos[TARGET], objective="tweedie")

    quantiles = {}
    for alpha in PERCENTILES:
        quantiles[alpha] = _fit_ensemble(lgb.LGBMRegressor, dataset[config.FEATURES],
                                         dataset[TARGET], objective="quantile", alpha=alpha)

    joblib.dump(clfs, config.RUTA_PROCESSED / "modelo_clasificador.pkl")
    joblib.dump(regs, config.RUTA_PROCESSED / "modelo_regresor.pkl")
    joblib.dump(quantiles, config.RUTA_PROCESSED / "modelos_quantile.pkl")
    print(f"  ✔ Ensembles entrenados: clasificador + regresor + {len(quantiles)} quantile")
    return clfs, regs, quantiles
