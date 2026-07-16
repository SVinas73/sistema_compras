"""Test de lectura del sustantivo principal de la descripción del repuesto.
Protege la regla: la PIEZA es la última palabra significativa, no el
modificador ('Engine Switch' es un switch, no un motor)."""
from src.tamano import tipo_pieza, clasificar

CASOS_TIER = {
    "EngineSwitch": "TINY", "Engine Switch": "TINY", "CARBONBRUSH": "TINY",
    "ConnectingRod": "MEDIO", "OilSeal": "TINY", "Machine Head": "STRUCT",
    "Cylinder Head": "STRUCT", "Motor Housing": "STRUCT", "Fuel Tank": "FULL",
    "Gear Box": "MEDIO", "Water Pump": "MEDIO", "Switch Assembly": "TINY",
    "Left and Right Case": "STRUCT", "Pressure pipe Assy": "SMALL",
    "Motor": "STRUCT", "Rotor": "MEDIO", "Bearing": "TINY",
}


def test_lectura_sustantivo_principal():
    for desc, esperado in CASOS_TIER.items():
        assert tipo_pieza(desc) == esperado, f"{desc}: {tipo_pieza(desc)} != {esperado}"


def test_casos_reportados_por_negocio():
    # Engine Switch de un generador = chico (es un switch, no un motor)
    assert clasificar("EngineSwitch", "GENERADOR 1200W INGCO GE15002")[0] == "Chico"
    # Machine Head (cabezal) de un compresor grande = grande
    assert clasificar("Machine Head", "COMPRESOR DE AIRE 100L 3.0HP")[0] == "Grande"
    # Motor de un atornillador = chico; el mismo motor en un generador, no
    assert clasificar("Motor", "ATORNILLADOR 20V CDLI20028 INGCO")[0] == "Chico"
    assert clasificar("Motor", "GENERADOR 12KW INGCO GE12000")[0] in ("Grande", "Extradimensional")


if __name__ == "__main__":
    test_lectura_sustantivo_principal()
    test_casos_reportados_por_negocio()
    print("OK: todos los tests de tamaño pasan")
