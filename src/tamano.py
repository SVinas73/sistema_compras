"""
tamano.py — Clasificación de tamaño de cada repuesto (tarea de una sola vez).

No hay volumetría real: se estima el tamaño físico de cada repuesto a partir de
DOS señales combinadas, como pidió el negocio:
  1. qué MÁQUINA es (del maestro Planilla_articulos2) → da la escala
  2. qué PIEZA es (descripción del repuesto en planilla_compras) → da la fracción

Un "Motor" de atornillador es chico; el de un generador, extradimensional.
Eso lo captura la MATRIZ clase_maquina × tipo_pieza de abajo.

Salida: data/processed/clasificacion_tamano.csv, con una marca 'revisar' para
los casos de baja confianza (cola larga de descripciones raras, máquina no
encontrada, o resultados Grande/Extradimensional que conviene chequear a mano).

Referencia del negocio (cubeta 600×400×300 mm = 72 L):
  Chico          — hasta 1/8 de cubeta (~9 L)
  Mediano        — hasta 1/2 cubeta (~36 L)
  Grande         — hasta 1 cubeta (~72 L)
  Extradimensional — excede la cubeta
"""
import re
import unicodedata

import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# 1) CLASE DE MÁQUINA: qué tan grande es el equipo entero.
#    MICRO < HAND < MEDIUM < LARGE < XLARGE
# ---------------------------------------------------------------------------
CLASE_MAQUINA = {
    # MICRO: cabe en una mano / caja de zapatos
    "ATORNILLADOR": "MICRO", "MINI": "MICRO", "MULTI": "MICRO",
    "LINTERNA": "MICRO", "ENGRASADOR": "MICRO", "CRIMPEADORA": "MICRO",
    "REMACHADORA": "MICRO", "NIVEL": "MICRO", "AFILADOR": "MICRO",
    "CARGADOR": "MICRO", "INFLADOR": "MICRO", "BATIDORA": "MICRO",
    "LICUADORA": "MICRO", "FOCO": "MICRO", "PISTOLA": "MICRO",
    "VENTOSA": "MICRO", "REGULADOR": "MICRO", "PULVERIZADOR": "MICRO",
    "LINTERNAS": "MICRO",
    # HAND: herramienta de mano grande (~media cubeta el equipo)
    "TALADRO": "HAND", "AMOLADORA": "HAND", "AMOLADURA": "HAND",
    "LIJADORA": "HAND", "ROUTER": "HAND", "LLAVE": "HAND",
    "CLAVADORA": "HAND", "TIJERA": "HAND", "CEPILLO": "HAND",
    "ACANALADORA": "HAND", "PLANCHA": "HAND", "SOPLADORA": "HAND",
    "SOPLA": "HAND", "CALADORA": "HAND", "CORTA": "HAND",
    "CORTADOR": "HAND", "BORDEADORA": "HAND", "ELECTROSIERRA": "HAND",
    "PODADORA": "HAND", "CIZALLA": "HAND", "HERRAMIENTA": "HAND",
    "HERRAMIENTAS": "HAND", "FUMIGADOR": "HAND", "MOCHILA": "HAND",
    "CARETA": "HAND", "SIERRA": "HAND", "ROTO": "HAND",
    # MEDIUM: máquina mediana (~una cubeta el equipo)
    "ROTOMARTILLO": "MEDIUM", "MARTILLO": "MEDIUM", "MOTOSIERRA": "MEDIUM",
    "HIDROLAVADORA": "MEDIUM", "ASPIRADORA": "MEDIUM", "SOLDADORA": "MEDIUM",
    "DESMALEZADORA": "MEDIUM", "TERMOFUSORA": "MEDIUM", "CALEFACTOR": "MEDIUM",
    "VIBRADOR": "MEDIUM", "MIXER": "MEDIUM", "LUSTRA": "MEDIUM",
    "LUSTRA-PULIDORA": "MEDIUM", "GARLOPA": "MEDIUM", "MOTO": "MEDIUM",
    "VENTILADOR": "MEDIUM", "LIMPIADORA": "MEDIUM", "ENSAMBLADORA": "MEDIUM",
    "COMPRESOR": "MEDIUM", "BOMBA": "MEDIUM", "MOTOBOMBA": "MEDIUM",
    "PATA": "MEDIUM", "PLANCHADORA": "MEDIUM", "TRITURADORA": "MEDIUM",
    "SENSITIVA": "MEDIUM",  # tronzadora de banco para metal
    # LARGE: equipo pesado de piso
    "GENERADOR": "LARGE", "HORMIGONERA": "LARGE", "COMPACTADORA": "LARGE",
    "CORTADORA": "LARGE", "POCERA": "LARGE", "ELEVADOR": "LARGE",
    "APAREJO": "LARGE", "TRASPALETA": "LARGE", "CARRETILLA": "LARGE",
    "MEZCLADORA": "LARGE", "CHIPEADORA": "LARGE", "MOTOCULTIVADOR": "LARGE",
    "EQUIPO": "LARGE", "HELICOPTERO": "LARGE", "MAQUINA": "LARGE",
    "MOTOR": "LARGE", "CARRO": "LARGE", "BRAZO": "LARGE", "PILAR": "LARGE",
    # XLARGE: extradimensional por definición
    "ANDAMIO": "XLARGE", "ESTANTERIA": "XLARGE", "SOPORTE": "XLARGE",
    "TRIPODE": "XLARGE",
}
CLASE_DEFAULT = "HAND"  # la mayoría del catálogo son herramientas de mano

# Refinamientos por potencia/palabra dentro de la descripción de la máquina.
# (una lista de (categoria, subcadena, nueva_clase); primera que matchea gana)
REFINAR_MAQUINA = [
    # compresores con tanque grande
    ("COMPRESOR", r"\b(50|100|150|200|300)\s?L", "LARGE"),
    # generadores por potencia: chicos → MEDIUM, grandes → LARGE (default LARGE)
    ("GENERADOR", r"\b(0[.,]\d|1[.,]\d|1|2|2[.,]\d)\s?KW", "MEDIUM"),
    # sierras de banco / ingletadora / mesa → más grandes
    ("SIERRA", r"BANCO|INGLET|MESA|CIRCULAR DE MESA", "MEDIUM"),
    # bombas sumergibles/superficie grandes
    ("BOMBA", r"SUMERGIBLE|CENTRIFUGA|3\"|4\"", "LARGE"),
]

# ---------------------------------------------------------------------------
# 2) TIPO DE PIEZA: qué fracción de la máquina representa el repuesto.
#    TINY < SMALL < MEDIUM < STRUCT < FULL
#    El orden de la lista es de MAYOR a menor prioridad al clasificar.
# ---------------------------------------------------------------------------
FULL = [  # el conjunto completo / bastidor entero
    "COMPLETE", "COMPLETO", "FULL ASSEMBLY", "FULL SET", "CONJUNTO COMPLETO",
    "BASTIDOR", "STRUCTURE", "ESTRUCTURA", "FRAME ASSEMBLY", "CHASSIS ASSEMBLY",
]
STRUCT = [  # pieza estructural grande (escala fuerte con la máquina)
    "MOTOR", "ENGINE", "HOUSING", "CARCASA", "CASE", "CRANKCASE", "CARTER",
    "CYLINDER BLOCK", "CIGUENAL BLOCK", "BODY", "CUERPO", "FRAME", "CHASIS",
    "CHASSIS", "TANK", "TANQUE", "DRUM", "TAMBOR", "BASE", "CABEZAL",
    "HEAD ASSY", "HEAD ASSEMBLY", "GEARBOX HOUSING", "MAIN BODY", "STAND",
    "BOOM", "HOOD", "CAPOT", "CASING", "CUBA", "DEPOSITO", "BOWL", "RAM",
]
MEDIO = [  # pieza mediana (escala con la máquina)
    "ROTOR", "STATOR", "ESTATOR", "PISTON", "CYLINDER", "CILINDRO",
    "CARBURETOR", "CARBURADOR", "GEARBOX", "GEAR BOX", "CAJA", "PUMP",
    "BOMBA", "IMPELLER", "IMPULSOR", "FAN", "VENTILADOR", "WHEEL", "RUEDA",
    "HANDLE", "MANGO", "COVER", "TAPA", "CUBIERTA", "GUARD", "PROTECTOR",
    "CRANKSHAFT", "CIGUENAL", "CONNECTING ROD", "CONNECTINGROD", "BIELA",
    "FLYWHEEL", "VOLANTE", "CLUTCH", "EMBRAGUE", "TRANSFORMER",
    "TRANSFORMADOR", "REEL", "CARRETE", "PLATE", "PLACA", "ARM", "BRAZO",
    "COIL", "BOBINA", "MUFFLER", "SILENCIADOR", "RADIATOR", "CAMSHAFT",
    "AIR CLEANER", "MANIFOLD", "PANEL",
]
TINY = [  # pieza chica siempre, sin importar la máquina
    "SCREW", "TORNILLO", "BOLT", "PERNO", "NUT", "TUERCA", "WASHER",
    "ARANDELA", "SPRING", "RESORTE", "MUELLE", "O-RING", "ORING", "O RING",
    "SEAL", "SELLO", "RETEN", "RETAINER", "BEARING", "RODAMIENTO",
    "RULEMAN", "BUSH", "BUJE", "BRUSH", "CARBON", "ESCOBILLA", "KNOB",
    "PERILLA", "BUTTON", "BOTON", "SWITCH", "INTERRUPTOR", "CAPACITOR",
    "CAPACITANCE", "CONDENSADOR", "GASKET", "JUNTA", "PIN", "CLIP",
    "TERMINAL", "CABLE", "WIRE", "CONNECTOR", "CONECTOR", "RING", "ANILLO",
    "KEY", "CHAVETA", "SENSOR", "LED", "IGBT", "FUSE", "FUSIBLE", "DIODE",
    "RELAY", "RELE", "SPARK PLUG", "BUJIA", "NEEDLE", "AGUJA", "BALL",
    "BOLILLA", "SPACER", "ESPACIADOR", "NAMEPLATE", "LABEL", "STICKER",
    "POSTER", "CARTEL", "SPONGE", "ESPONJA", "FOAM", "GROMMET", "OIL SEAL",
    "GAUGE", "MANOMETRO", "DIPSTICK", "OUTLET", "PLUG", "SOCKET", "COLLAR",
    "CLAMP", "ABRAZADERA", "MEMBRANE", "DIAPHRAGM", "MEMBRANA", "BAFFLE",
]
SMALL = [  # pieza chica-mediana (chico salvo en máquinas grandes)
    "GEAR", "ENGRANAJE", "TRIGGER", "GATILLO", "CHUCK", "MANDRIL",
    "COLLET", "SPINDLE", "SHAFT", "EJE", "ROD", "VALVE", "VALVULA",
    "NOZZLE", "BOQUILLA", "FILTER", "FILTRO", "PULLEY", "POLEA", "BELT",
    "CORREA", "PCB", "BOARD", "PLAQUETA", "LEVER", "PALANCA", "CAM", "LEVA",
    "GRIP", "EMPUÑADURA", "HOSE", "MANGUERA", "PIPE", "TUBO", "BLADE",
    "CUCHILLA", "DISC", "DISCO", "ADAPTER", "ADAPTADOR", "BRACKET",
    "SOPORTE", "GUIDE", "GUIA", "ROLLER", "RODILLO", "FLANGE", "BRIDA",
    "WRENCH", "TELEFLEX", "BAG", "BOLSA", "CHAIN", "CADENA", "SPROCKET",
    "PINON", "ELBOW", "CODO", "FITTING", "COUPLING", "ACOPLE", "STRAINER",
    "BAR", "TRIGGER SET", "GEAR SET",
]

_ORDEN = [("FULL", FULL), ("STRUCT", STRUCT), ("MEDIO", MEDIO),
          ("TINY", TINY), ("SMALL", SMALL)]

# ---------------------------------------------------------------------------
# 3) MATRIZ clase_maquina × tipo_pieza → tamaño final
# ---------------------------------------------------------------------------
CH, ME, GR, EX = "Chico", "Mediano", "Grande", "Extradimensional"
MATRIZ = {
    #            MICRO HAND  MEDIUM LARGE XLARGE
    "TINY":   {"MICRO": CH, "HAND": CH, "MEDIUM": CH, "LARGE": CH, "XLARGE": ME},
    "SMALL":  {"MICRO": CH, "HAND": CH, "MEDIUM": CH, "LARGE": ME, "XLARGE": GR},
    "MEDIO":  {"MICRO": CH, "HAND": CH, "MEDIUM": ME, "LARGE": GR, "XLARGE": EX},
    "STRUCT": {"MICRO": CH, "HAND": ME, "MEDIUM": GR, "LARGE": EX, "XLARGE": EX},
    "FULL":   {"MICRO": ME, "HAND": GR, "MEDIUM": EX, "LARGE": EX, "XLARGE": EX},
}


def _norm(texto: str) -> str:
    """Mayúsculas sin acentos, para matchear en español e inglés."""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return t.upper()


def _padre_de_repuesto(cod: str) -> str:
    """AAC2508-SP-53/55/56 -> AAC2508 ; ING-MGT1601-SP-C -> MGT1601."""
    c = re.sub(r"^[A-Z]+[-_]", "", str(cod))
    return c.split("-SP-")[0]


def clase_maquina(nom_maquina: str) -> str:
    """Clase de tamaño del equipo a partir de su descripción."""
    if not nom_maquina or pd.isna(nom_maquina):
        return None
    t = _norm(nom_maquina)
    cat = t.split()[0] if t.split() else ""
    base = CLASE_MAQUINA.get(cat, CLASE_DEFAULT)
    for categoria, patron, nueva in REFINAR_MAQUINA:
        if cat == categoria and re.search(patron, t):
            return nueva
    return base


def tipo_pieza(nom_repuesto: str) -> str:
    """Tier de la pieza a partir de su descripción. None si no reconoce nada."""
    t = _norm(nom_repuesto)
    for tier, palabras in _ORDEN:
        for p in palabras:
            if p in t:
                return tier
    return None


def clasificar(nom_repuesto, nom_maquina):
    """Devuelve (tamano, clase_maquina, tier_pieza, confianza, motivo).

    confianza:
      alta  — máquina y pieza reconocidas, resultado Chico/Mediano
      media — reconocido pero resultado Grande/Extra (verificar por impacto)
      baja  — máquina no encontrada o pieza no reconocida (se usó default)
    """
    cm = clase_maquina(nom_maquina)
    tier = tipo_pieza(nom_repuesto)

    motivos = []
    if cm is None:
        cm = CLASE_DEFAULT
        motivos.append("maquina no encontrada")
    if tier is None:
        tier = "SMALL"  # default prudente: la mayoría de piezas son chicas
        motivos.append("pieza no reconocida")

    tamano = MATRIZ[tier][cm]
    if motivos:
        confianza = "baja"
    elif tamano in (GR, EX):
        confianza = "media"
        motivos.append("resultado grande: verificar por impacto")
    else:
        confianza = "alta"
    return tamano, cm, tier, confianza, "; ".join(motivos)


def run() -> pd.DataFrame:
    """Clasifica todos los repuestos y guarda el CSV. Tarea de una sola vez."""
    print("[TAMAÑO] Clasificando repuestos por tamaño...")
    art = pd.read_excel(config.RUTA_RAW / "Planilla_articulos2.xlsx", sheet_name="Grilla")
    art["cod_articulo"] = art["cod_articulo"].astype(str).str.strip()
    art["padre"] = art["cod_articulo"].apply(lambda c: re.sub(r"^[A-Z]+_", "", c))
    maestro = art.drop_duplicates("padre").set_index("padre")["nom_articulo"]

    comp = pd.read_excel(config.RUTA_RAW / "planilla_compras.xls",
                         engine="xlrd", skiprows=4)
    comp["cod_articulo"] = comp["cod_articulo"].astype(str).str.strip()
    comp = comp[["cod_articulo", "nom_articulo"]].drop_duplicates("cod_articulo")
    comp["padre"] = comp["cod_articulo"].apply(_padre_de_repuesto)
    comp["articulo_padre"] = comp["padre"].map(maestro)

    res = comp.apply(
        lambda r: clasificar(r["nom_articulo"], r["articulo_padre"]),
        axis=1, result_type="expand",
    )
    res.columns = ["tamano", "clase_maquina", "tipo_pieza", "confianza", "motivo"]
    salida = pd.concat([comp, res], axis=1)

    cols = ["cod_articulo", "nom_articulo", "padre", "articulo_padre",
            "clase_maquina", "tipo_pieza", "tamano", "confianza", "motivo"]
    destino = config.RUTA_PROCESSED / "clasificacion_tamano.csv"
    salida[cols].to_csv(destino, index=False)

    print(f"  ✔ {len(salida)} repuestos clasificados → {destino.name}")
    print("  Tamaño:")
    print(salida["tamano"].value_counts().to_string())
    print("  Confianza:")
    print(salida["confianza"].value_counts().to_string())
    return salida


if __name__ == "__main__":
    run()
