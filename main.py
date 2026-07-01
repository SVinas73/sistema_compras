"""
main.py — Orquestador del sistema completo.

Uso mensual:
  1. Exportar la planilla nueva del sistema a data/raw/
  2. Ejecutar:  python main.py
  3. Revisar:   data/processed/orden_de_compra.csv
"""
from src import extract, features, inference, train


def main():
    print("=" * 50)
    print("SISTEMA DE OPTIMIZACIÓN DE COMPRAS — REPUESTOS")
    print("=" * 50)

    # Fase 1: leer y unificar los Excel
    largo = extract.run()

    # Fase 3: features + target (la fase 2 de segmentación vive
    # dentro de inference como clase ABC; los inmovilizados quedan
    # fuera automáticamente al filtrar SKUs sin ventas)
    dataset = features.construir(largo)

    # Fase 4: entrenar (con validación informativa previa)
    clfs, regs, quantiles = train.run(dataset, con_validacion=True)

    # Fase 5: predicción al día de hoy y orden de compra
    hoy = features.foto_actual(largo)
    inference.run(hoy, clfs, regs, quantiles)

    print("=" * 50)
    print("LISTO ✔")


if __name__ == "__main__":
    main()
