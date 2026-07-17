"""
tamano_ia.py — Clasificación de tamaño con IA (Claude).

A diferencia del clasificador por reglas (src/tamano.py), este LEE la
descripción COMPLETA del repuesto Y de su máquina y decide con criterio: sabe
que el disco de un allanador es enorme y el de una amoladora es chico, aunque
ambos digan "disc". Es la opción de máxima exactitud para la cola larga.

Requisitos:
  - Una API key de Anthropic en la variable de entorno ANTHROPIC_API_KEY.
  - pip install anthropic  (ya incluido en requirements.txt)

Cómo funciona:
  - Usa la Batch API (50% más barata) porque es una tarea de una sola vez sin
    apuro. Manda un pedido por repuesto con la regla de cubetas cacheada.
  - Es reanudable: guarda lo ya clasificado en cache_tamano_ia.json, así una
    segunda corrida solo pide lo que falta.

Uso:  python -m src.tamano_ia
"""
import json
import re
import time

import pandas as pd

from . import config, tamano

# Modelo. claude-opus-4-8 es el más capaz; para abaratar ~5x esta tarea de
# clasificación se puede cambiar a "claude-haiku-4-5" sin gran pérdida.
MODELO = "claude-opus-4-8"
POR_LOTE = 100          # cada cuántos pedidos se informa avance
CACHE = config.RUTA_PROCESSED / "cache_tamano_ia.json"
SALIDA = config.RUTA_PROCESSED / "clasificacion_tamano_ia.csv"

TAMANOS = ["Chico", "Mediano", "Grande", "Extradimensional"]

SYSTEM = """\
Sos un experto en logística de un almacén de repuestos. Clasificás el tamaño \
FÍSICO de cada repuesto para saber en qué cubeta de estantería entra.

Referencia (cubeta estándar 600 x 400 x 300 mm = 72 litros):
- "Chico": entra en 1/8 de cubeta (aprox. 150x200x300 mm / 9 L). Tornillos, \
sellos, switches, escobillas, rodamientos, tapas chicas, juntas, cables.
- "Mediano": entra en 1/2 cubeta (aprox. 300x400x300 mm / 36 L). Rotores, \
carburadores, bombas chicas, tapas medianas, manijas.
- "Grande": entra en una cubeta entera (600x400x300 mm / 72 L). Cabezales, \
carcasas de máquinas medianas, motores de máquinas medianas.
- "Extradimensional": NO entra en ninguna cubeta (excede 600x400x300 mm). \
Tanques de nafta, chasis, bastidores, tambores de hormigonera, platos de \
allanador, motores de generadores grandes.

Reglas de criterio:
1) Leé la descripción COMPLETA del repuesto. La pieza es el sustantivo \
principal: "Engine Switch" es un SWITCH (chico), no un motor. "Fuel Tank Cap" \
es una TAPA (chica), no un tanque.
2) El tamaño depende de la MÁQUINA: el mismo repuesto es distinto según el \
equipo. Un "motor" de atornillador es Chico; el de un generador, \
Extradimensional. Un "disc"/"pan" de amoladora es Chico; el plato de un \
allanador (helicóptero alisador), Extradimensional.
3) Ante la duda entre dos, elegí el más chico, salvo que la máquina sea \
claramente grande.

Respondé SOLO con la categoría, sin explicaciones."""

ESQUEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"tamano": {"type": "string", "enum": TAMANOS}},
        "required": ["tamano"],
        "additionalProperties": False,
    },
}


def construir_dataset() -> pd.DataFrame:
    """Repuestos unidos a la descripción de su artículo padre (sin API)."""
    art = pd.read_excel(config.RUTA_RAW / "Planilla_articulos2.xlsx", sheet_name="Grilla")
    art["cod_articulo"] = art["cod_articulo"].astype(str).str.strip()
    art["padre"] = art["cod_articulo"].apply(lambda c: re.sub(r"^[A-Z]+_", "", c))
    maestro = art.drop_duplicates("padre").set_index("padre")["nom_articulo"]

    comp = pd.read_excel(config.RUTA_RAW / "planilla_compras.xls",
                         engine="xlrd", skiprows=4)
    comp["cod_articulo"] = comp["cod_articulo"].astype(str).str.strip()
    comp = comp[["cod_articulo", "nom_articulo"]].drop_duplicates("cod_articulo")
    comp["padre"] = comp["cod_articulo"].apply(tamano._padre_de_repuesto)
    comp["articulo_padre"] = comp["padre"].map(maestro).fillna("(máquina desconocida)")
    return comp.reset_index(drop=True)


def _mensaje_usuario(nom_repuesto, nom_maquina) -> str:
    return (f"Repuesto: {nom_repuesto}\n"
            f"Pertenece a la máquina: {nom_maquina}\n"
            f"¿En qué categoría de tamaño entra?")


def _pedido(cid: str, nom_repuesto, nom_maquina):
    """Un Request de la Batch API. Import perezoso para no exigir el SDK
    cuando solo se arma el dataset."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    return Request(
        custom_id=cid,
        params=MessageCreateParamsNonStreaming(
            model=MODELO,
            max_tokens=20,
            system=[{"type": "text", "text": SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            output_config={"format": ESQUEMA},
            messages=[{"role": "user",
                       "content": _mensaje_usuario(nom_repuesto, nom_maquina)}],
        ),
    )


def _cargar_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def run() -> pd.DataFrame:
    import anthropic

    df = construir_dataset()
    df["cid"] = ["i" + str(i) for i in range(len(df))]
    cid2cod = dict(zip(df["cid"], df["cod_articulo"]))

    cache = _cargar_cache()                       # cod_articulo -> tamano
    faltan = df[~df["cod_articulo"].isin(cache)]
    print(f"[TAMAÑO IA] {len(df)} repuestos | ya en cache: {len(cache)} | "
          f"a clasificar: {len(faltan)}")

    if len(faltan):
        client = anthropic.Anthropic()
        pedidos = [_pedido(r.cid, r.nom_articulo, r.articulo_padre)
                   for r in faltan.itertuples()]
        lote = client.messages.batches.create(requests=pedidos)
        print(f"  batch {lote.id} enviado; esperando (suele tardar minutos)...")

        while True:
            lote = client.messages.batches.retrieve(lote.id)
            if lote.processing_status == "ended":
                break
            time.sleep(30)

        ok = err = 0
        for res in client.messages.batches.results(lote.id):
            cod = cid2cod.get(res.custom_id)
            if res.result.type == "succeeded":
                txt = next((b.text for b in res.result.message.content
                            if b.type == "text"), "")
                try:
                    cache[cod] = json.loads(txt)["tamano"]
                    ok += 1
                except (json.JSONDecodeError, KeyError):
                    err += 1
            else:
                err += 1
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
        print(f"  clasificados: {ok} | con error (reintentar re-corriendo): {err}")

    df["tamano"] = df["cod_articulo"].map(cache)
    cols = ["cod_articulo", "nom_articulo", "articulo_padre", "tamano"]
    df[cols].to_csv(SALIDA, index=False)
    print(f"  ✔ Exportado a {SALIDA}")
    print(df["tamano"].value_counts(dropna=False).to_string())
    return df


if __name__ == "__main__":
    run()
