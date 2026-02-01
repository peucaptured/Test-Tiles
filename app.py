import io
import json
import random
import zipfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image
st.set_page_config(page_title="Tileset -> Catalog JSON", layout="wide")

# ============================================================
#  Tileset -> Catalog JSON (1x1 tiles + multi-tile objects)
#  - Focado em: separar tileset rápido + exportar JSON p/ outro app
#  - Suporta objetos >1 tile (ex.: árvores grandes, rochas grandes)
#  - Sem foco em casas/estruturas
# ============================================================

# -------------------------
# Core: sheet + helpers
# -------------------------


BIOMES = {
    "Floresta": {"biome_tag": "biome:forest", "terrain": ["grass", "dirt", "rock", "water", "mud"],
                 "details": ["moss", "leaf_litter", "roots", "flowers", "stones"]},
    "Caverna": {"biome_tag": "biome:cave", "terrain": ["rock", "gravel", "mud", "water", "lava"],
                "details": ["moss", "stalagmite", "stalactite", "cracks", "bones"]},
    "Rio": {"biome_tag": "biome:river", "terrain": ["water", "rock", "sand", "dirt", "mud"],
            "details": ["bank", "foam", "reeds", "pebbles"], "depth": ["depth:shallow", "depth:deep"]},
    "Lago": {"biome_tag": "biome:lake", "terrain": ["water", "sand", "dirt", "mud", "rock"],
             "details": ["shore", "algae", "lily", "foam", "reeds"], "depth": ["depth:shallow", "depth:deep"]},
    "Costa do Mar": {"biome_tag": "biome:coast", "terrain": ["sea", "sand", "rock", "water"],
                     "details": ["wave", "foam", "wet_sand", "tidepool", "shells"], "depth": ["depth:shallow", "depth:deep"]},
    "Deserto": {"biome_tag": "biome:desert", "terrain": ["sand", "rock", "dirt"],
                "details": ["dune", "cracks", "pebbles", "dry_grass", "bones"]},
    "Montanha": {"biome_tag": "biome:mountain", "terrain": ["rock", "dirt", "grass", "snow"],
                 "details": ["cliff", "scree", "moss", "cracks", "pebbles"]},
    "Terra Batida": {"biome_tag": "biome:pathland", "terrain": ["dirt", "grass", "rock", "sand"],
                     "details": ["path", "track", "ruts", "pebbles", "dry_grass"]},
}

ROLES = ["role:base", "role:variant", "role:edge", "role:corner", "role:detail"]

def _slug(s: str) -> str:
    s = (s or "").strip().lower().replace(" ", "_").replace("-", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch in "_:")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")

def build_tags(biome_name: str, terrain: str, role: str, details: list[str], extra: list[str]) -> list[str]:
    tags = []
    if biome_name in BIOMES:
        tags.append(BIOMES[biome_name]["biome_tag"])
    if terrain: tags.append(_slug(terrain))
    if role: tags.append(role)
    for d in (details or []): tags.append(_slug(d))
    for t in (extra or []):
        t = t.strip()
        if t: tags.append(_slug(t))
    # dedupe preservando ordem
    out, seen = [], set()
    for t in tags:
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out

def suggest_name(kind: str, biome_name: str, terrain: str, role: str, variant_n: int) -> str:
    prefix = "obj" if kind == "object" else "tile"
    b = _slug(biome_name)
    t = _slug(terrain)
    r = _slug(role).replace("role:", "")
    return f"{prefix}_{b}_{t}_{r}_{variant_n:02d}"

@dataclass(frozen=True)
class TileCoord:
    c: int
    r: int

@dataclass
class CatalogEntry:
    id: str                 # string estável (ex.: "t_12_5_1x1")
    kind: str               # "tile" | "object"
    name: str               # nome humano
    tags: List[str]         # ex.: ["ground","grass"] ou ["object","tree"]
    c: int                  # coluna top-left
    r: int                  # linha top-left
    w: int                  # largura em tiles
    h: int                  # altura em tiles
    weight: float = 1.0     # peso p/ sorteio (no outro app)

@dataclass
class TileSheet:
    img: Image.Image
    tile: int
    cols: int
    rows: int

    @classmethod
    def from_image(cls, img: Image.Image, tile: int) -> "TileSheet":
        img = img.convert("RGBA")
        w, h = img.size
        cols = w // tile
        rows = h // tile
        if cols <= 0 or rows <= 0:
            raise ValueError("Tile size inválido para o tamanho do spritesheet.")
        return cls(img=img, tile=tile, cols=cols, rows=rows)

    def crop_tile(self, c: int, r: int, w: int = 1, h: int = 1) -> Image.Image:
        x0 = c * self.tile
        y0 = r * self.tile
        return self.img.crop((x0, y0, x0 + w * self.tile, y0 + h * self.tile))

def pil_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    bio = io.BytesIO()
    img.save(bio, format=fmt)
    return bio.getvalue()

@st.cache_resource(show_spinner=False)
def load_sheet(png_bytes: bytes, tile_size: int) -> TileSheet:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    return TileSheet.from_image(img, tile=tile_size)

@st.cache_data(show_spinner=False)
def compute_empty_mask(png_bytes: bytes, tile_size: int) -> Tuple[int, int, List[List[bool]]]:
    """
    Retorna mask [rows][cols] True se tile é 'vazio' (alpha==0).
    Faz isso em lote (bem mais rápido do que crop por tile).
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    a = np.array(img, dtype=np.uint8)[:, :, 3]  # alpha
    h_px, w_px = a.shape
    cols = w_px // tile_size
    rows = h_px // tile_size
    a = a[:rows * tile_size, :cols * tile_size]

    # reshape: (rows, tile, cols, tile) -> max por tile
    a4 = a.reshape(rows, tile_size, cols, tile_size)
    max_alpha = a4.max(axis=(1, 3))
    empty = (max_alpha == 0)

    # to list-of-lists (st.cache_data-friendly)
    return rows, cols, empty.tolist()

@st.cache_data(show_spinner=False)
def contact_sheet_bytes(png_bytes: bytes, tile_size: int, row0: int, n_rows: int, zoom: int) -> bytes:
    """
    Renderiza uma "janela" do tileset (várias linhas) para navegação rápida.
    Cacheado por página -> navegação instantânea.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    w_px, h_px = img.size
    cols = w_px // tile_size
    rows = h_px // tile_size

    row0 = max(0, min(row0, rows - 1))
    row1 = min(rows, row0 + n_rows)
    crop = img.crop((0, row0 * tile_size, cols * tile_size, row1 * tile_size))

    if zoom != 1:
        crop = crop.resize((crop.size[0] * zoom, crop.size[1] * zoom), Image.NEAREST)

    # desenha grade leve (opcional) – custo baixo
    # (usamos pixels pretos finos)
    arr = np.array(crop, dtype=np.uint8)
    step = tile_size * zoom
    # linhas horizontais
    for y in range(0, arr.shape[0], step):
        arr[y:y+1, :, :3] = 0
        arr[y:y+1, :, 3] = np.maximum(arr[y:y+1, :, 3], 180)
    # linhas verticais
    for x in range(0, arr.shape[1], step):
        arr[:, x:x+1, :3] = 0
        arr[:, x:x+1, 3] = np.maximum(arr[:, x:x+1, 3], 180)

    out = Image.fromarray(arr, "RGBA")
    return pil_bytes(out, "PNG")

def ensure_state():
    st.session_state.setdefault("catalog", {})  # id -> CatalogEntry dict
    st.session_state.setdefault("ignored", set())  # set[(c,r)]
    st.session_state.setdefault("tile_c", 0)
    st.session_state.setdefault("tile_r", 0)
    st.session_state.setdefault("row_page", 0)


def catalog_to_json_dict(tile_size: int, cols: int, rows: int) -> Dict:
    entries = list(st.session_state["catalog"].values())
    # normaliza/ordena por (r,c,w,h)
    entries = sorted(entries, key=lambda e: (e["r"], e["c"], e["h"], e["w"], e["id"]))
    return {
        "tile_size": tile_size,
        "cols": cols,
        "rows": rows,
        "entries": entries,
    }

def make_entry_id(c: int, r: int, w: int, h: int, kind: str) -> str:
    return f"{'o' if kind=='object' else 't'}_{c}_{r}_{w}x{h}"

def add_entry(entry: CatalogEntry):
    st.session_state["catalog"][entry.id] = asdict(entry)

def remove_entry(entry_id: str):
    st.session_state["catalog"].pop(entry_id, None)

def region_overlaps(a: CatalogEntry, b: CatalogEntry) -> bool:
    ax0, ay0 = a.c, a.r
    ax1, ay1 = a.c + a.w - 1, a.r + a.h - 1
    bx0, by0 = b.c, b.r
    bx1, by1 = b.c + b.w - 1, b.r + b.h - 1
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

def region_in_bounds(cols: int, rows: int, c: int, r: int, w: int, h: int) -> bool:
    return 0 <= c < cols and 0 <= r < rows and (c + w) <= cols and (r + h) <= rows

def pack_zip_of_entries(sheet: TileSheet, entries: List[Dict]) -> bytes:
    """
    Cria um ZIP com PNGs recortados das entries.
    - tiles 1x1: tile_<id>.png
    - objects: obj_<id>.png
    """
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for e in entries:
            kind = e["kind"]
            c, r, w, h = int(e["c"]), int(e["r"]), int(e["w"]), int(e["h"])
            img = sheet.crop_tile(c, r, w=w, h=h)
            fname = ("obj_" if kind == "object" else "tile_") + e["id"] + ".png"
            z.writestr(fname, pil_bytes(img, "PNG"))
        # adiciona o JSON junto
        z.writestr("tiles_catalog.json", json.dumps({"entries": entries}, ensure_ascii=False, indent=2).encode("utf-8"))
    return bio.getvalue()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Tileset -> Catalog JSON", layout="wide")
st.title("Tileset ➜ Catálogo JSON (Pokémon maps)")

ensure_state()

with st.sidebar:
    st.subheader("Tileset")
    up = st.file_uploader("PNG do tileset", type=["png"])
    tile_size = st.number_input("Tamanho do tile (px)", min_value=8, max_value=128, value=32, step=1)
    zoom = st.selectbox("Zoom (visualização)", [1, 2, 3, 4], index=1)

    st.divider()
    st.subheader("Navegação")
    rows_per_page = st.selectbox("Linhas por página", [6, 8, 10, 12, 16], index=2)

    st.caption("Dica: use as setas pra navegar rápido e ir catalogando.")

if not up:
    st.info("Envie um tileset PNG.")
    st.stop()

png_bytes = up.getvalue()
sheet = load_sheet(png_bytes, int(tile_size))
rows, cols, empty_mask = compute_empty_mask(png_bytes, int(tile_size))
empty_mask = np.array(empty_mask, dtype=bool)

# corrige se image tinha sobra
sheet = TileSheet.from_image(sheet.img, tile=int(tile_size))

# ajustes state range
st.session_state["tile_c"] = int(np.clip(st.session_state["tile_c"], 0, sheet.cols - 1))
st.session_state["tile_r"] = int(np.clip(st.session_state["tile_r"], 0, sheet.rows - 1))
st.session_state["row_page"] = int(np.clip(st.session_state["row_page"], 0, max(0, sheet.rows - 1)))

tabs = st.tabs(["1) Catalogar tiles/objetos", "2) Exportar", "3) Preview (opcional)"])

# ============================================================
# TAB 1: Catalogar
# ============================================================
with tabs[0]:
    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        st.subheader("Navegador do tileset")

        # paginação por linhas (muito mais leve que desenhar o sheet todo)
        max_page0 = max(0, sheet.rows - rows_per_page)
        page_row0 = st.slider("Linha inicial da página", 0, max_page0, int(st.session_state["row_page"]), 1)
        st.session_state["row_page"] = int(page_row0)

        contact = contact_sheet_bytes(png_bytes, int(tile_size), row0=page_row0, n_rows=rows_per_page, zoom=int(zoom))
        st.image(contact, caption=f"Cols={sheet.cols}  Rows={sheet.rows}  | Página: linhas {page_row0}..{min(sheet.rows-1, page_row0+rows_per_page-1)}", use_container_width=True)

        st.divider()
        st.markdown("### Selecionar tile (col, row)")

        nav_a, nav_b = st.columns([1.15, 2.85], gap="large")

        def _wrap(v: int, lo: int, hi: int) -> int:
            n = hi - lo + 1
            return lo + ((v - lo) % n)

        def move(dc: int = 0, dr: int = 0):
            st.session_state["tile_c"] = _wrap(int(st.session_state["tile_c"]) + dc, 0, sheet.cols - 1)
            st.session_state["tile_r"] = _wrap(int(st.session_state["tile_r"]) + dr, 0, sheet.rows - 1)

        with nav_a:
            up_b = st.button("▲", use_container_width=True)
            lcol, rcol = st.columns(2, gap="small")
            with lcol:
                left_b = st.button("◀", use_container_width=True)
            with rcol:
                right_b = st.button("▶", use_container_width=True)
            down_b = st.button("▼", use_container_width=True)

            jump = st.selectbox("Pulo vertical", [1, 5, 10, 25, 50, 100], index=2)
            j1, j2 = st.columns(2, gap="small")
            with j1:
                jump_up = st.button(f"-{jump}", use_container_width=True)
            with j2:
                jump_dn = st.button(f"+{jump}", use_container_width=True)

            if up_b:
                move(dr=-1)
                st.rerun()
            if down_b:
                move(dr=+1)
                st.rerun()
            if left_b:
                move(dc=-1)
                st.rerun()
            if right_b:
                move(dc=+1)
                st.rerun()
            if jump_up:
                move(dr=-int(jump))
                st.rerun()
            if jump_dn:
                move(dr=+int(jump))
                st.rerun()

        with nav_b:
            c_in = st.number_input("Coluna (c)", 0, sheet.cols - 1, int(st.session_state["tile_c"]), 1)
            r_in = st.number_input("Linha (r)", 0, sheet.rows - 1, int(st.session_state["tile_r"]), 1)
            if c_in != st.session_state["tile_c"] or r_in != st.session_state["tile_r"]:
                st.session_state["tile_c"] = int(c_in)
                st.session_state["tile_r"] = int(r_in)
                st.rerun()

        c = int(st.session_state["tile_c"])
        r = int(st.session_state["tile_r"])

        tile_img = sheet.crop_tile(c, r, 1, 1)
        is_empty = bool(empty_mask[r, c])
        st.image(tile_img.resize((160, 160), Image.NEAREST), caption=f"Tile (c={c}, r={r})" + (" — vazio" if is_empty else ""))

        col1, col2 = st.columns(2, gap="small")
        with col1:
            if st.button("Ignorar / Des-ignorar (tile vazio/ruído)", use_container_width=True):
                key = (c, r)
                if key in st.session_state["ignored"]:
                    st.session_state["ignored"].remove(key)
                else:
                    st.session_state["ignored"].add(key)
                st.rerun()

        with col2:
            if st.button("Auto-ignorar TODOS os vazios", use_container_width=True):
                st.session_state["ignored"] = set(map(tuple, np.argwhere(empty_mask).tolist()))
                st.rerun()

        st.caption(f"Ignorados: {len(st.session_state['ignored'])} tiles")
        with right:
            st.subheader("Adicionar ao catálogo")
        
            # ----------------------------
            # Tipo
            # ----------------------------
            kind_ui = st.radio(
                "Tipo de item",
                ["Tile (1x1)", "Objeto (multi-tile)"],
                horizontal=True,
                help="Tile: chão/água/bordas. Objeto: árvore/rocha/estrutura maior."
            )
            entry_kind = "object" if kind_ui.startswith("Objeto") else "tile"
        
            st.divider()
        
            # ----------------------------
            # Seletor: Bioma / Terreno / Função / Detalhes
            # ----------------------------
            biome_keys = list(BIOMES.keys())
            last_biome = st.session_state.get("last_biome")
            biome_idx = biome_keys.index(last_biome) if last_biome in BIOMES else 0
        
            biome_name = st.selectbox(
                "Bioma",
                biome_keys,
                index=biome_idx,
                help="Define automaticamente biome:* e filtra opções de terreno/detalhes."
            )
        
            terrain_opts = BIOMES[biome_name]["terrain"]
            terrain = st.selectbox(
                "Terreno principal",
                terrain_opts,
                index=0,
                help="Material dominante. Ex.: grass, rock, sand, water..."
            )
        
            last_role = st.session_state.get("last_role", "role:base")
            role_idx = ROLES.index(last_role) if last_role in ROLES else 0
            role = st.selectbox(
                "Função da tile",
                ROLES,
                index=role_idx,
                help="role:base (chão base), role:variant (variação), role:edge/corner (transição), role:detail (overlay)."
            )
        
            details_opts = BIOMES[biome_name].get("details", [])
            last_details = st.session_state.get("last_details", [])
            # mantém apenas os que ainda existem no bioma escolhido
            last_details = [d for d in last_details if d in details_opts]
        
            details = st.multiselect(
                "Detalhes (opcional)",
                options=details_opts,
                default=last_details,
                help="Microtags (moss, foam, cracks...) para diferenciar variações."
            )
        
            # Profundidade (se aplicável)
            depth_opts = BIOMES[biome_name].get("depth", [])
            if depth_opts:
                depth = st.selectbox("Profundidade (água)", ["(nenhuma)"] + depth_opts, index=0)
                if depth != "(nenhuma)":
                    details = list(details) + [depth]
        
            # Tags extras (custom)
            custom_tags_raw = st.text_input(
                "Tags extras (opcional, separadas por vírgula)",
                value=st.session_state.get("last_custom_tags", ""),
                help="Para casos especiais. Ex.: bridge, cliff, overlay, walkway..."
            )
            custom_tags = [t.strip() for t in custom_tags_raw.split(",") if t.strip()]
        
            # Tags finais
            tags = build_tags(biome_name, terrain, role, list(details), custom_tags)
        
            st.caption("Tags finais (geradas automaticamente):")
            st.code(", ".join(tags) if tags else "(nenhuma)")
        
            st.divider()
        
            # ----------------------------
            # Nome (sugestão automática)
            # ----------------------------
            variant_n = st.number_input(
                "Variação #",
                min_value=1, max_value=99,
                value=int(st.session_state.get("last_variant_n", 1)),
                step=1,
                help="Gera nomes consistentes: ..._01, ..._02, ..."
            )
            auto_name = suggest_name(entry_kind, biome_name, terrain, role, int(variant_n))
        
            use_auto = st.toggle(
                "Gerar nome automaticamente",
                value=True,
                help="Recomendado: evita nomes fora do padrão."
            )
            name = st.text_input(
                "Nome",
                value=(auto_name if use_auto else st.session_state.get("last_name", "")),
                help="Ex.: tile_floresta_grass_base_01 | obj_caverna_rock_detail_01"
            )
        
            # ----------------------------
            # Peso
            # ----------------------------
            weight = st.number_input(
                "Peso (sorteio)",
                min_value=0.01, max_value=100.0,
                value=float(st.session_state.get("last_weight", 1.0)),
                step=0.25,
                help="Quanto maior, mais comum o item no sorteio."
            )
        
            # ----------------------------
            # Tamanho (objeto) / fixo (tile)
            # ----------------------------
            if entry_kind == "tile":
                w_obj, h_obj = 1, 1
            else:
                st.markdown("#### Tamanho do objeto (em tiles)")
                w_obj = st.number_input(
                    "Largura", 2, 8,
                    int(st.session_state.get("last_w", 2)),
                    1,
                    help="Quantos tiles na horizontal."
                )
                h_obj = st.number_input(
                    "Altura", 2, 8,
                    int(st.session_state.get("last_h", 2)),
                    1,
                    help="Quantos tiles na vertical."
                )
        
            # ----------------------------
            # Preview do recorte
            # ----------------------------
            c = int(st.session_state["tile_c"])
            r = int(st.session_state["tile_r"])
            ok_bounds = region_in_bounds(sheet.cols, sheet.rows, c, r, int(w_obj), int(h_obj))
        
            if not ok_bounds:
                st.error("Esse recorte sai do tileset. Ajuste (c,r) ou (w,h).")
            else:
                prev = sheet.crop_tile(c, r, int(w_obj), int(h_obj))
                st.image(
                    prev.resize((min(360, prev.size[0] * 2), min(360, prev.size[1] * 2)), Image.NEAREST),
                    caption=f"Preview recorte: (c={c}, r={r}, w={w_obj}, h={h_obj})"
                )
        
            # ----------------------------
            # Conflitos (sobreposição)
            # ----------------------------
            existing_entries = [CatalogEntry(**e) for e in st.session_state["catalog"].values()]
            conflict = False
            if ok_bounds:
                tmp = CatalogEntry(
                    id="__tmp__",
                    kind=entry_kind,
                    name=(name.strip() if name else "unnamed"),
                    tags=tags or [],
                    c=c, r=r, w=int(w_obj), h=int(h_obj),
                    weight=float(weight),
                )
                for e in existing_entries:
                    if region_overlaps(tmp, e):
                        conflict = True
                        break
        
            if conflict:
                st.warning("⚠️ Essa área sobrepõe uma entry já cadastrada. Se for intencional, remova a antiga antes.")
        
            # ----------------------------
            # Add
            # ----------------------------
            add_btn = st.button("Adicionar ao catálogo", type="primary", use_container_width=True, disabled=not ok_bounds)
            if add_btn:
                if (c, r) in st.session_state["ignored"]:
                    st.warning("Esse tile está marcado como 'ignorado'. Remova de ignorados se você quer catalogar.")
                else:
                    entry_id = make_entry_id(c, r, int(w_obj), int(h_obj), entry_kind)
                    entry = CatalogEntry(
                        id=entry_id,
                        kind=entry_kind,
                        name=name.strip() if name and name.strip() else entry_id,
                        tags=tags,
                        c=c, r=r, w=int(w_obj), h=int(h_obj),
                        weight=float(weight),
                    )
                    add_entry(entry)
        
                    # lembra inputs
                    st.session_state["last_name"] = entry.name
                    st.session_state["last_weight"] = float(weight)
                    st.session_state["last_w"] = int(w_obj)
                    st.session_state["last_h"] = int(h_obj)
        
                    st.session_state["last_biome"] = biome_name
                    st.session_state["last_role"] = role
                    st.session_state["last_details"] = list(details)
                    st.session_state["last_custom_tags"] = custom_tags_raw
                    st.session_state["last_variant_n"] = int(variant_n)
        
                    st.success(f"Adicionado: {entry.id}")
                    st.rerun()
        
            # ----------------------------
            # Catálogo atual
            # ----------------------------
            st.divider()
            st.markdown("### Catálogo atual")
            st.caption("Dica: use filtros por tag para achar rápido.")
        
            all_entries = list(st.session_state["catalog"].values())
        
            all_tags = sorted({t for e in all_entries for t in (e.get("tags") or [])})
            tag_filter = st.multiselect("Filtrar por tags", options=all_tags, default=[])
            kind_filter = st.multiselect("Filtrar por tipo", options=["tile", "object"], default=["tile", "object"])
        
            def _match(e: Dict) -> bool:
                if e.get("kind") not in kind_filter:
                    return False
                if tag_filter:
                    et = set(e.get("tags") or [])
                    if not set(tag_filter).issubset(et):
                        return False
                return True
        
            filtered = [e for e in all_entries if _match(e)]
            st.write(f"Items: **{len(filtered)}** (de {len(all_entries)})")
        
            for e in filtered[:250]:
                cols_row = st.columns([3.2, 1.2, 0.9], gap="small")
                with cols_row[0]:
                    st.markdown(f"**{e['id']}** — `{e['kind']}` — {e.get('name','')}")
                    st.caption(", ".join(e.get("tags") or []))
                with cols_row[1]:
                    st.caption(f"(c={e['c']}, r={e['r']})  {e['w']}x{e['h']}")
                with cols_row[2]:
                    if st.button("Remover", key=f"rm_{e['id']}", use_container_width=True):
                        remove_entry(e["id"])
                        st.rerun()
        
            if len(filtered) > 250:
                st.info("Mostrando só os primeiros 250 (use filtros).")

# ============================================================
# TAB 2: Exportar
# ============================================================
with tabs[1]:
    st.subheader("Exportar catálogo")

    entries = list(st.session_state["catalog"].values())
    payload = catalog_to_json_dict(int(tile_size), sheet.cols, sheet.rows)

    st.download_button(
        "Baixar tiles_catalog.json",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="tiles_catalog.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Export opcional: ZIP com PNGs recortados (tiles/objetos)")
    st.caption("Útil se o outro app preferir carregar PNGs individuais em vez do spritesheet inteiro.")
    if not entries:
        st.info("Adicione pelo menos 1 item no catálogo para gerar o ZIP.")
    else:
        zip_bytes = pack_zip_of_entries(sheet, entries)
        st.download_button(
            "Baixar tiles_catalog.zip",
            data=zip_bytes,
            file_name="tiles_catalog.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Ignorados")
    st.caption("Exportamos ignorados para você reaplicar no futuro.")
    ignored_list = sorted([{"c": c, "r": r} for (c, r) in st.session_state["ignored"]], key=lambda x: (x["r"], x["c"]))
    st.download_button(
        "Baixar ignored_tiles.json",
        data=json.dumps({"ignored": ignored_list}, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="ignored_tiles.json",
        mime="application/json",
        use_container_width=True,
    )

# ============================================================
# TAB 3: Preview (opcional)
# ============================================================
with tabs[2]:
    st.subheader("Preview rápido (só para validar visualmente)")
    st.caption("Este preview é simples e rápido. O objetivo principal do app é exportar o catálogo JSON.")

    entries = list(st.session_state["catalog"].values())
    if not entries:
        st.info("Cadastre alguns tiles de chão (tags tipo: ground, grass, water, sand...) para pré-visualizar.")
        st.stop()

    # pega pools por tag
    def pool(tag: str) -> List[Dict]:
        return [e for e in entries if tag in (e.get("tags") or []) and e.get("kind") == "tile"]

    grass = pool("grass") or pool("ground")
    water = pool("water")
    sand  = pool("sand")

    if not grass:
        st.warning("Não achei tiles com tag 'grass' nem 'ground'. Coloque tags nos tiles 1x1 para o preview.")
        st.stop()

    pw = st.slider("Preview largura (tiles)", 20, 120, 60, 5)
    ph = st.slider("Preview altura (tiles)", 15, 90, 40, 5)
    seed = st.number_input("Seed preview", value=1337, step=1)
    density = st.slider("Densidade de objetos (multi-tile)", 0.0, 0.20, 0.06, 0.01)

    rng = random.Random(int(seed))
    out = Image.new("RGBA", (pw * sheet.tile, ph * sheet.tile), (0, 0, 0, 0))

    # base ground (grass)
    g_choices = grass
    for y in range(ph):
        for x in range(pw):
            e = rng.choice(g_choices)
            img = sheet.crop_tile(int(e["c"]), int(e["r"]), 1, 1)
            out.alpha_composite(img, (x * sheet.tile, y * sheet.tile))

    # algumas manchas de água/areia (bem leve)
    def sprinkle(tag_pool: List[Dict], p: float):
        if not tag_pool:
            return
        for y in range(ph):
            for x in range(pw):
                if rng.random() < p:
                    e = rng.choice(tag_pool)
                    img = sheet.crop_tile(int(e["c"]), int(e["r"]), 1, 1)
                    out.alpha_composite(img, (x * sheet.tile, y * sheet.tile))

    sprinkle(water, p=0.06 if water else 0.0)
    sprinkle(sand,  p=0.04 if sand else 0.0)

    # objetos multi-tile
    objs = [e for e in entries if e.get("kind") == "object"]
    occ = np.zeros((ph, pw), dtype=bool)

    def can_place(x: int, y: int, w_: int, h_: int) -> bool:
        if x + w_ > pw or y + h_ > ph:
            return False
        return not occ[y:y+h_, x:x+w_].any()

    if objs:
        for y in range(ph):
            for x in range(pw):
                if rng.random() < float(density):
                    o = rng.choice(objs)
                    w_, h_ = int(o["w"]), int(o["h"])
                    if can_place(x, y, w_, h_):
                        img = sheet.crop_tile(int(o["c"]), int(o["r"]), w_, h_)
                        out.alpha_composite(img, (x * sheet.tile, y * sheet.tile))
                        occ[y:y+h_, x:x+w_] = True

    zoom2 = st.selectbox("Zoom preview", [1, 2, 3, 4], index=1)
    if zoom2 != 1:
        out2 = out.resize((out.size[0] * zoom2, out.size[1] * zoom2), Image.NEAREST)
    else:
        out2 = out

    st.image(out2, use_container_width=True)
    st.download_button(
        "Baixar preview PNG",
        data=pil_bytes(out, "PNG"),
        file_name=f"preview_{int(seed)}.png",
        mime="image/png",
        use_container_width=True,
    )
