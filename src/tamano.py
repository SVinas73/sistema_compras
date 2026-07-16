"""
tamano.py — Clasificación de tamaño de cada repuesto (tarea de una sola vez).

Objetivo: un almacén vertical con CUBETAS de distintos tamaños. Se necesita saber
en qué cubeta entra cada repuesto (o si no entra en ninguna). Referencia
(cubeta estándar 600×400×300 mm = 72 L):
  Chico          — entra en 1/8 de cubeta (~150×200×300 / 9 L)
  Mediano        — entra en 1/2 cubeta   (~300×400×300 / 36 L)
  Grande         — entra en una cubeta    (600×400×300 / 72 L)
  Extradimensional — NO entra en ninguna cubeta

No hay volumetría real: se estima combinando DOS señales, en este orden de
importancia:
  1. QUÉ PIEZA es (descripción del repuesto) — lo más importante.
  2. QUÉ MÁQUINA es (maestro Planilla_articulos2) — da la escala.

Clave para leer bien inglés/español: la PIEZA es el sustantivo PRINCIPAL, que
suele ser la ÚLTIMA palabra. "Engine Switch" = un SWITCH (chico), no un motor.
"Machine Head" = un CABEZAL (grande). Por eso se lee de derecha a izquierda y se
parten las palabras pegadas ("EngineSwitch" -> "Engine" "Switch").
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
    # MICRO: cabe en 1/8 de cubeta el equipo entero
    "ATORNILLADOR": "MICRO", "MINI": "MICRO", "MULTI": "MICRO",
    "LINTERNA": "MICRO", "LINTERNAS": "MICRO", "ENGRASADOR": "MICRO",
    "CRIMPEADORA": "MICRO", "REMACHADORA": "MICRO", "NIVEL": "MICRO",
    "AFILADOR": "MICRO", "CARGADOR": "MICRO", "INFLADOR": "MICRO",
    "BATIDORA": "MICRO", "LICUADORA": "MICRO", "FOCO": "MICRO",
    "PISTOLA": "MICRO", "VENTOSA": "MICRO", "REGULADOR": "MICRO",
    "PULVERIZADOR": "MICRO",
    # HAND: herramienta de mano (equipo ~1/2 cubeta)
    "TALADRO": "HAND", "AMOLADORA": "HAND", "AMOLADURA": "HAND",
    "LIJADORA": "HAND", "ROUTER": "HAND", "LLAVE": "HAND",
    "CLAVADORA": "HAND", "TIJERA": "HAND", "CEPILLO": "HAND",
    "ACANALADORA": "HAND", "PLANCHA": "HAND", "SOPLADORA": "HAND",
    "SOPLA": "HAND", "CALADORA": "HAND", "CORTA": "HAND",
    "CORTADOR": "HAND", "BORDEADORA": "HAND", "ELECTROSIERRA": "HAND",
    "PODADORA": "HAND", "CIZALLA": "HAND", "HERRAMIENTA": "HAND",
    "HERRAMIENTAS": "HAND", "FUMIGADOR": "HAND", "CARETA": "HAND",
    "SIERRA": "HAND", "ROTO": "HAND", "ROTOMARTILLO": "HAND",
    "MOCHILA": "HAND",
    # MEDIUM: máquina mediana (equipo ~1 cubeta)
    "MARTILLO": "MEDIUM", "MOTOSIERRA": "MEDIUM", "HIDROLAVADORA": "MEDIUM",
    "ASPIRADORA": "MEDIUM", "SOLDADORA": "MEDIUM", "DESMALEZADORA": "MEDIUM",
    "TERMOFUSORA": "MEDIUM", "CALEFACTOR": "MEDIUM", "VIBRADOR": "MEDIUM",
    "MIXER": "MEDIUM", "LUSTRA": "MEDIUM", "LUSTRA-PULIDORA": "MEDIUM",
    "GARLOPA": "MEDIUM", "MOTO": "MEDIUM", "VENTILADOR": "MEDIUM",
    "LIMPIADORA": "MEDIUM", "ENSAMBLADORA": "MEDIUM", "COMPRESOR": "MEDIUM",
    "BOMBA": "MEDIUM", "MOTOBOMBA": "MEDIUM", "PATA": "MEDIUM",
    "PLANCHADORA": "MEDIUM", "TRITURADORA": "MEDIUM", "SENSITIVA": "MEDIUM",
    # LARGE: equipo pesado, más grande que una cubeta
    "GENERADOR": "LARGE", "HORMIGONERA": "LARGE", "COMPACTADORA": "LARGE",
    "CORTADORA": "LARGE", "POCERA": "LARGE", "ELEVADOR": "LARGE",
    "APAREJO": "LARGE", "TRASPALETA": "LARGE", "CARRETILLA": "LARGE",
    "MEZCLADORA": "LARGE", "CHIPEADORA": "LARGE", "MOTOCULTIVADOR": "LARGE",
    "EQUIPO": "LARGE", "HELICOPTERO": "LARGE", "MAQUINA": "LARGE",
    "MOTOR": "LARGE", "CARRO": "LARGE", "BRAZO": "LARGE", "PILAR": "LARGE",
    # XLARGE: enorme
    "ANDAMIO": "XLARGE", "ESTANTERIA": "XLARGE", "SOPORTE": "XLARGE",
    "TRIPODE": "XLARGE",
}
CLASE_DEFAULT = "HAND"  # la mayoría del catálogo son herramientas de mano


def _potencia_w(texto: str):
    """Primera potencia en watts que aparezca ('1200W', '3.0KW', '2 KW')."""
    m = re.search(r"(\d+[.,]?\d*)\s?(KW|W)\b", texto)
    if not m:
        return None
    val = float(m.group(1).replace(",", "."))
    return val * 1000 if m.group(2) == "KW" else val


def clase_maquina(nom_maquina):
    """Clase de tamaño del equipo a partir de su descripción."""
    if not nom_maquina or pd.isna(nom_maquina):
        return None
    t = _norm(nom_maquina)
    cat = t.split()[0] if t.split() else ""
    base = CLASE_MAQUINA.get(cat, CLASE_DEFAULT)

    # Refinamientos por potencia/tamaño reales
    if cat == "GENERADOR":
        w = _potencia_w(t)
        return "MEDIUM" if (w is not None and w <= 2500) else "LARGE"
    if cat == "COMPRESOR":
        if re.search(r"\b(50|80|100|120|150|200|300)\s?L", t):
            return "LARGE"
    if cat == "SIERRA" and re.search(r"BANCO|INGLET|MESA", t):
        return "MEDIUM"
    if cat == "BOMBA" and re.search(r"SUMERGIBLE|CENTRIFUGA|\b[34]\"", t):
        return "LARGE"
    return base


# ---------------------------------------------------------------------------
# 2) TIPO DE PIEZA: qué fracción de la máquina es el repuesto.
#    Se lee el SUSTANTIVO PRINCIPAL (última palabra significativa).
#    Tiers, de mayor a menor tamaño: FULL > STRUCT > MEDIO > SMALL > TINY
# ---------------------------------------------------------------------------
TIERS = {
    # Pieza enorme / conjunto entero: no entra en cubeta en máquinas medianas+
    "FULL": [
        "FRAME", "CHASIS", "CHASSIS", "BASTIDOR", "ESTRUCTURA", "TANK",
        "TANQUE", "DRUM", "TAMBOR", "BOOM", "STAND", "BASE", "CUBA",
        "DEPOSITO", "BOILER", "CALDERA", "COMPLETE", "COMPLETO", "CONJUNTO",
    ],
    # Pieza estructural grande (escala fuerte con la máquina)
    "STRUCT": [
        "MOTOR", "ENGINE", "HOUSING", "CARCASA", "CASE", "CASING",
        "CRANKCASE", "CARTER", "BLOCK", "BODY", "CUERPO", "HEAD", "CABEZAL",
        "HOOD", "CAPOT", "GEARCASE", "CYLINDER BLOCK", "MAIN BODY",
    ],
    # Pieza mediana (escala con la máquina)
    "MEDIO": [
        "ROTOR", "STATOR", "ESTATOR", "ARMATURE", "INDUCIDO", "PISTON",
        "CYLINDER", "CILINDRO", "CARBURETOR", "CARBURADOR", "GEARBOX",
        "GEAR BOX", "PUMP", "IMPELLER", "IMPULSOR", "FAN", "WHEEL", "RUEDA",
        "HANDLE", "MANGO", "COVER", "TAPA", "CUBIERTA", "GUARD", "PROTECTOR",
        "CRANKSHAFT", "CIGUENAL", "CONNECTING ROD", "BIELA", "FLYWHEEL",
        "VOLANTE", "CLUTCH", "EMBRAGUE", "TRANSFORMER", "TRANSFORMADOR",
        "REEL", "CARRETE", "PLATE", "PLACA", "COIL", "BOBINA", "MUFFLER",
        "SILENCIADOR", "RADIATOR", "CAMSHAFT", "AIR CLEANER", "MANIFOLD",
        "PANEL", "RAM", "HELICE",
    ],
    # Pieza chica-mediana (chico, salvo en máquinas grandes)
    "SMALL": [
        "GEAR", "ENGRANAJE", "TRIGGER", "GATILLO", "CHUCK", "MANDRIL",
        "COLLET", "SPINDLE", "SHAFT", "EJE", "ROD", "VALVE", "VALVULA",
        "NOZZLE", "BOQUILLA", "FILTER", "FILTRO", "PULLEY", "POLEA", "BELT",
        "CORREA", "PCB", "BOARD", "PLAQUETA", "LEVER", "PALANCA", "CAM",
        "GRIP", "HOSE", "MANGUERA", "PIPE", "TUBO", "BLADE", "CUCHILLA",
        "DISC", "DISCO", "ADAPTER", "ADAPTADOR", "BRACKET", "SOPORTE",
        "GUIDE", "GUIA", "ROLLER", "RODILLO", "FLANGE", "BRIDA", "WRENCH",
        "TELEFLEX", "BAG", "BOLSA", "CHAIN", "CADENA", "SPROCKET", "PINON",
        "ELBOW", "CODO", "FITTING", "COUPLING", "ACOPLE", "STRAINER", "BAR",
    ],
    # Pieza chica siempre, sin importar la máquina
    "TINY": [
        "SCREW", "TORNILLO", "BOLT", "PERNO", "NUT", "TUERCA", "WASHER",
        "ARANDELA", "SPRING", "RESORTE", "MUELLE", "O-RING", "ORING", "SEAL",
        "SELLO", "RETEN", "RETAINER", "BEARING", "RODAMIENTO", "RULEMAN",
        "BUSH", "BUJE", "BRUSH", "CARBON", "ESCOBILLA", "KNOB", "PERILLA",
        "BUTTON", "BOTON", "SWITCH", "INTERRUPTOR", "CAPACITOR",
        "CAPACITANCE", "CONDENSADOR", "GASKET", "JUNTA", "PIN", "CLIP",
        "TERMINAL", "CABLE", "WIRE", "CONNECTOR", "CONECTOR", "RING",
        "ANILLO", "KEY", "CHAVETA", "SENSOR", "LED", "IGBT", "FUSE",
        "FUSIBLE", "DIODE", "RELAY", "RELE", "SPARK PLUG", "BUJIA", "NEEDLE",
        "AGUJA", "BALL", "SPACER", "NAMEPLATE", "LABEL", "STICKER", "POSTER",
        "CARTEL", "SPONGE", "ESPONJA", "FOAM", "GROMMET", "GAUGE",
        "MANOMETRO", "DIPSTICK", "OUTLET", "PLUG", "SOCKET", "COLLAR",
        "CLAMP", "ABRAZADERA", "MEMBRANE", "DIAPHRAGM", "MEMBRANA", "BAFFLE",
        "CAP", "LID", "TAPON", "JOINT", "LENS", "LENTE", "GLASS", "VIDRIO",
        "LAMP", "LAMPARA", "BULB", "STOPPER", "TOPE", "NIPPLE", "JET",
    ],
}
# Piezas cuyo tier grande SOLO vale si son el sustantivo principal (última
# palabra): si aparecen como modificador ('Fuel TANK Cap') no cuentan.
_TIER_GRANDE = {"STRUCT", "FULL"}
# Piezas chicas-medianas que se suman a SMALL
TIERS["SMALL"] += [
    "ELEMENT", "ELEMENTO", "CARTRIDGE", "CARTUCHO", "MODULE", "MODULO",
    "HARNESS", "ARNES", "LINER", "PLUNGER", "EMBOLO", "HOOK", "GANCHO",
    "LATCH", "TUBE", "FERRULE", "SLEEVE", "CAMISA",
]
# Prioridad si un tier "gana" ante empate de posición: el más grande.
_RANK = {"FULL": 5, "STRUCT": 4, "MEDIO": 3, "SMALL": 2, "TINY": 1}
# Diccionarios derivados
WORD2TIER, PHRASE2TIER = {}, {}
for _tier, _kws in TIERS.items():
    for _kw in _kws:
        (PHRASE2TIER if " " in _kw else WORD2TIER).setdefault(_kw, _tier)
# Palabras que se ignoran al buscar el sustantivo principal: relleno y
# CALIFICADORES de posición/tamaño (no son la pieza; 'Housing Left' -> Housing)
_RELLENO = {
    "ASSY", "ASSEMBLY", "ASSEMBLIES", "ASM", "SET", "KIT", "PARTS", "PART",
    "COMP", "COMPONENT", "UNIT", "AND", "OR", "FOR", "OF", "THE", "A", "DE",
    "PARA", "CON", "WITH",
    # calificadores de posición / tamaño / lado
    "LEFT", "RIGHT", "UPPER", "LOWER", "TOP", "BOTTOM", "FRONT", "REAR",
    "INNER", "OUTER", "INTERIOR", "EXTERIOR", "INTERNAL", "EXTERNAL",
    "MAIN", "SUB", "BIG", "SMALL", "LONG", "SHORT", "NEW", "OLD", "STD",
    "STANDARD", "IZQUIERDO", "IZQUIERDA", "DERECHO", "DERECHA", "SUPERIOR",
    "INFERIOR", "DELANTERO", "DELANTERA", "TRASERO", "TRASERA", "INTERNO",
    "EXTERNO", "GRANDE", "CHICO", "COMPLETA", "SIDE",
}


def _norm(texto):
    """Mayúsculas sin acentos."""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return t.upper()


def _tokens(desc):
    """Parte la descripción en palabras, separando también las pegadas en
    camelCase: 'EngineSwitch' -> ['ENGINE','SWITCH']."""
    t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(desc))   # camelCase
    t = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", t)  # letra|nº
    t = _norm(t)
    t = re.sub(r"[^A-Z0-9 ]+", " ", t)
    return [w for w in t.split() if w and w not in _RELLENO]


def _tier_substring(token):
    """Para palabras pegadas en MAYÚSCULA ('CARBONBRUSH') que no se separaron:
    busca la keyword de una sola palabra que termina MÁS a la derecha
    (= sustantivo principal). Devuelve tier o None."""
    mejor_fin, mejor_tier = -1, None
    for kw, tier in WORD2TIER.items():
        i = token.rfind(kw)
        if i >= 0:
            fin = i + len(kw)
            if fin > mejor_fin or (fin == mejor_fin and _RANK[tier] > _RANK[mejor_tier]):
                mejor_fin, mejor_tier = fin, tier
    return mejor_tier


def _clasificar_segmento(toks):
    """Tier del sustantivo principal de una lista de tokens (derecha a
    izquierda). Una pieza GRANDE (estructural/enorme) solo cuenta si es la
    ÚLTIMA palabra: si está en el medio es un modificador ('Fuel TANK Cap')."""
    n = len(toks)
    for i in range(n - 1, -1, -1):
        es_ultima = (i == n - 1)
        if i > 0:  # frase de dos palabras terminando acá (p.ej. CONNECTING ROD)
            t = PHRASE2TIER.get(toks[i - 1] + " " + toks[i])
            if t and not (t in _TIER_GRANDE and not es_ultima):
                return t
        t = WORD2TIER.get(toks[i]) or _tier_substring(toks[i])
        if t and not (t in _TIER_GRANDE and not es_ultima):
            return t
    return None


def tipo_pieza(nom_repuesto):
    """Tier de la pieza. Lee la descripción COMPLETA: si hay coma, la pieza
    principal suele ir antes ('Joint Assy, Fuel Tank' = un joint). None si no
    reconoce nada."""
    partes = str(nom_repuesto).split(",")
    t = _clasificar_segmento(_tokens(partes[0]))
    if t is not None:
        return t
    if len(partes) > 1:  # fallback: toda la descripción junta
        return _clasificar_segmento(_tokens(str(nom_repuesto).replace(",", " ")))
    return None


# ---------------------------------------------------------------------------
# 3) MATRIZ tipo_pieza × clase_maquina → tamaño (¿en qué cubeta entra?)
# ---------------------------------------------------------------------------
CH, ME, GR, EX = "Chico", "Mediano", "Grande", "Extradimensional"
MATRIZ = {
    #            MICRO HAND  MEDIUM LARGE  XLARGE
    "TINY":   {"MICRO": CH, "HAND": CH, "MEDIUM": CH, "LARGE": CH, "XLARGE": CH},
    "SMALL":  {"MICRO": CH, "HAND": CH, "MEDIUM": CH, "LARGE": CH, "XLARGE": ME},
    "MEDIO":  {"MICRO": CH, "HAND": CH, "MEDIUM": ME, "LARGE": GR, "XLARGE": GR},
    "STRUCT": {"MICRO": CH, "HAND": ME, "MEDIUM": GR, "LARGE": GR, "XLARGE": EX},
    "FULL":   {"MICRO": ME, "HAND": GR, "MEDIUM": EX, "LARGE": EX, "XLARGE": EX},
}


def _padre_de_repuesto(cod):
    """AAC2508-SP-53/55/56 -> AAC2508 ; ING-MGT1601-SP-C -> MGT1601."""
    c = re.sub(r"^[A-Z]+[-_]", "", str(cod))
    return c.split("-SP-")[0]


def clasificar(nom_repuesto, nom_maquina):
    """Devuelve (tamano, clase_maquina, tier_pieza, confianza, motivo)."""
    cm = clase_maquina(nom_maquina)
    tier = tipo_pieza(nom_repuesto)

    motivos = []
    if tier is None:
        tier = "SMALL"  # default prudente: la mayoría de piezas son chicas
        motivos.append("pieza no reconocida")
    if cm is None:
        cm = CLASE_DEFAULT
        motivos.append("maquina no encontrada")

    tamano = MATRIZ[tier][cm]
    if motivos:
        confianza = "baja"
    elif tamano in (GR, EX):
        confianza = "media"        # alto impacto: conviene un vistazo
    else:
        confianza = "alta"
    return tamano, cm, tier, confianza, "; ".join(motivos)


def run():
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
    print("  Tamaño (% del catálogo, para dimensionar las cubetas):")
    dist = salida["tamano"].value_counts()
    for k in [CH, ME, GR, EX]:
        n = int(dist.get(k, 0))
        print(f"    {k:16s} {n:5d}  ({100 * n / len(salida):.1f}%)")
    print("  Confianza:", salida["confianza"].value_counts().to_dict())
    return salida


if __name__ == "__main__":
    run()
