import io
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image


# =========================
# Tileset
# =========================
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

    def tile_at(self, c: int, r: int) -> Image.Image:
        if not (0 <= c < self.cols and 0 <= r < self.rows):
            raise IndexError("Tile fora do range.")
        x0 = c * self.tile
        y0 = r * self.tile
        return self.img.crop((x0, y0, x0 + self.tile, y0 + self.tile))


def pil_bytes(img: Image.Image, fmt="PNG") -> bytes:
    bio = io.BytesIO()
    img.save(bio, format=fmt)
    return bio.getvalue()


# =========================
# IDs de terreno
# =========================
T_OCEAN   = 0
T_FRESH   = 1
T_BEACH   = 2
T_GRASS   = 3
T_FORESTF = 4
T_DESERT  = 5
T_CAVEF   = 6

# Overlay IDs
O_NONE    = 0
O_TREES   = 1
O_ROCKS   = 2
O_CACTUS  = 3
O_CAVEDEC = 4
O_FOAM    = 5


# =========================
# Noise / smoothing helpers
# =========================
def box_blur(a: np.ndarray, passes: int = 2) -> np.ndarray:
    """Blur simples (sem dependências)."""
    a = a.astype(np.float32)
    for _ in range(passes):
        s = np.zeros_like(a)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                s += np.roll(np.roll(a, dy, axis=0), dx, axis=1)
        a = s / 9.0
    return a


def near_mask(mask: np.ndarray, dist: int = 1) -> np.ndarray:
    """Retorna True onde está perto de 'mask'."""
    m = mask.astype(bool)
    out = np.zeros_like(m)
    for dy in range(-dist, dist + 1):
        for dx in range(-dist, dist + 1):
            out |= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    return out


# =========================
# Rivers / lakes
# =========================
def carve_rivers(height: np.ndarray, ocean: np.ndarray, seed: int, n_rivers: int = 3, max_len: int = 600) -> np.ndarray:
    """
    Faz rios seguindo a descida de height até atingir oceano.
    Retorna mask de água doce.
    """
    rng = random.Random(seed)
    h, w = height.shape
    fresh = np.zeros((h, w), dtype=np.uint8)

    # candidatos de nascente: locais altos e não oceano
    candidates = np.argwhere((height > np.quantile(height, 0.80)) & (~ocean))
    if len(candidates) == 0:
        return fresh

    for i in range(n_rivers):
        y, x = candidates[rng.randrange(len(candidates))]
        for _ in range(max_len):
            if ocean[y, x]:
                break
            fresh[y, x] = 1

            # escolhe próximo passo: vizinho com menor height (com ruído)
            best = None
            best_score = 1e9
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny = (y + dy) % h
                nx = (x + dx) % w
                if ocean[ny, nx]:
                    best = (ny, nx)
                    best_score = -1e9
                    break
                score = float(height[ny, nx]) + rng.random() * 0.02
                if score < best_score:
                    best_score = score
                    best = (ny, nx)

            if best is None:
                break
            y, x = best

    return fresh


def add_lakes(fresh: np.ndarray, land: np.ndarray, seed: int, n_lakes: int = 4, r_min: int = 2, r_max: int = 6) -> np.ndarray:
    """Adiciona alguns lagos circulares em terra."""
    rng = random.Random(seed + 999)
    h, w = fresh.shape
    out = fresh.copy()

    land_cells = np.argwhere(land & (out == 0))
    if len(land_cells) == 0:
        return out

    for _ in range(n_lakes):
        cy, cx = land_cells[rng.randrange(len(land_cells))]
        r = rng.randint(r_min, r_max)
        for y in range(cy - r, cy + r + 1):
            for x in range(cx - r, cx + r + 1):
                yy = y % h
                xx = x % w
                if (y - cy)**2 + (x - cx)**2 <= r*r:
                    out[yy, xx] = 1

    return out


# =========================
# Cave generation (cellular automata)
# =========================
def gen_cave_domain(h: int, w: int, seed: int, fill: float = 0.48, steps: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera máscara de caverna em dois grids:
    - cave_solid: parede (1) / chão (0)
    - cave_floor: chão (True)
    """
    rng = np.random.default_rng(seed + 12345)
    a = (rng.random((h, w)) < fill).astype(np.uint8)  # 1=parede

    for _ in range(steps):
        neigh = np.zeros_like(a, dtype=np.int32)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy == 0 and dx == 0:
                    continue
                neigh += np.roll(np.roll(a, dy, axis=0), dx, axis=1)
        # regra típica
        a = np.where(neigh >= 5, 1, 0).astype(np.uint8)

    cave_floor = (a == 0)
    return a, cave_floor


def cave_lake(cave_floor: np.ndarray, seed: int, level: float = 0.10) -> np.ndarray:
    """Água em pontos baixos dentro da caverna."""
    rng = np.random.default_rng(seed + 555)
    noise = rng.random(cave_floor.shape)
    noise = box_blur(noise, passes=3)
    lake = (noise < np.quantile(noise, level)) & cave_floor
    return lake.astype(np.uint8)


# =========================
# Biome classify + 2 layers
# =========================
from typing import Tuple, List

def generate_world(
    w: int,
    h: int,
    seed: int,
    sea_level: float,
    forest_moist: float,
    desert_dry: float,
    n_rivers: int,
    n_lakes: int,
    cave_ratio: float,
    cave_lake_level: float,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str,int,int]]]:
    """
    Retorna:
    - ground[y,x] (uint8): IDs T_*
    - overlay[y,x] (uint8): IDs O_*  (SEM árvores grandes)
    - objects: lista de ("tree_2x2", x, y)
    """
    rng = np.random.default_rng(seed)

    # height / moisture
    height = rng.random((h, w))
    height = box_blur(height, passes=3)

    moisture = rng.random((h, w))
    moisture = box_blur(moisture, passes=2)

    ocean = height < sea_level
    land = ~ocean

    # costa: terra perto do mar
    coast = land & near_mask(ocean.astype(np.uint8), dist=1)

    # rios/lagos (água doce)
    fresh = carve_rivers(height, ocean, seed=seed, n_rivers=n_rivers)
    fresh = add_lakes(fresh, land, seed=seed, n_lakes=n_lakes)

    # domínio de caverna
    cave_solid, cave_floor = gen_cave_domain(h, w, seed=seed)
    cave_noise = rng.random((h, w))
    cave_domain = (cave_noise < cave_ratio) & land
    cave_floor = cave_floor & cave_domain

    cave_water = cave_lake(cave_floor, seed=seed, level=cave_lake_level).astype(bool)

    # ground
    ground = np.full((h, w), T_GRASS, dtype=np.uint8)
    ground[ocean] = T_OCEAN
    ground[(fresh == 1) & land] = T_FRESH
    ground[coast & (ground == T_GRASS)] = T_BEACH

    base_land = land & (ground != T_FRESH) & (~cave_domain)

    ground[base_land & (moisture < desert_dry) & (~coast)] = T_DESERT
    ground[base_land & (moisture > forest_moist) & (~coast) & (ground != T_DESERT)] = T_FORESTF

    ground[cave_floor] = T_CAVEF
    ground[cave_water] = T_FRESH

    # overlay (aqui ficam só coisas 1x1)
    overlay = np.zeros((h, w), dtype=np.uint8)
    deco = rng.random((h, w))

    near_fresh = near_mask((ground == T_FRESH).astype(np.uint8), dist=1)
    can_tree = ((ground == T_FORESTF) | (ground == T_GRASS)) & (~near_fresh) & (~coast)

    # >>>>>>> AQUI: em vez de O_TREES, vamos gerar objects 2x2
    objects = []
    occ = np.zeros((h, w), dtype=bool)

    def can_place_2x2(x: int, y: int) -> bool:
        if x < 0 or y < 0 or x + 1 >= w or y + 1 >= h:
            return False
        if occ[y:y+2, x:x+2].any():
            return False
        g = ground[y:y+2, x:x+2]
        if not np.isin(g, [T_FORESTF, T_GRASS]).all():
            return False
        # opcional: garante que não está muito perto de água doce/mar
        if near_mask((ground == T_FRESH).astype(np.uint8), dist=1)[y:y+2, x:x+2].any():
            return False
        return True

    # densidade das árvores (ajuste 0.06~0.12)
    for y in range(h - 1):
        for x in range(w - 1):
            if can_tree[y, x] and deco[y, x] < 0.06 and can_place_2x2(x, y):
                objects.append(("tree_2x2", x, y))
                occ[y:y+2, x:x+2] = True

    # rochas/cactos etc continuam no overlay 1x1
    can_rocks = ((ground == T_DESERT) | (ground == T_GRASS)) & (~near_fresh)
    overlay[can_rocks & (deco > 0.92)] = O_ROCKS
    overlay[(ground == T_DESERT) & (deco < 0.06)] = O_CACTUS
    overlay[(ground == T_CAVEF) & (deco < 0.10)] = O_CAVEDEC

    ocean_near_coast = (ground == T_OCEAN) & near_mask((ground == T_BEACH).astype(np.uint8), dist=1)
    overlay[ocean_near_coast & (deco < 0.20)] = O_FOAM

    return ground, overlay, objects


# =========================
# Rendering using mapping
# =========================
def pick(rng: random.Random, arr: List[Tuple[int,int]]) -> Tuple[int,int]:
    return arr[rng.randrange(len(arr))]


def render_layers(sheet: TileSheet, ground: np.ndarray, overlay: np.ndarray, mapping: Dict, seed: int, scale: int) -> Image.Image:
    """
    mapping esperado:
    {
      "ground": { "ocean": [[c,r],...], "fresh":..., "beach":..., "grass":..., "forest":..., "desert":..., "cave":... },
      "overlay": { "trees":..., "rocks":..., "cactus":..., "cavedec":..., "foam":... }
    }
    """
    h, w = ground.shape
    out = Image.new("RGBA", (w*sheet.tile, h*sheet.tile), (0,0,0,0))
    rng = random.Random(seed)
    cache: Dict[Tuple[int,int], Image.Image] = {}

    # helpers de tiles
    def get_tile(coords: Tuple[int,int]) -> Image.Image:
        if coords not in cache:
            cache[coords] = sheet.tile_at(coords[0], coords[1])
        return cache[coords]

    # mapeia ID->lista
    gmap = {
        T_OCEAN:  mapping["ground"].get("ocean", []),
        T_FRESH:  mapping["ground"].get("fresh", []),
        T_BEACH:  mapping["ground"].get("beach", []),
        T_GRASS:  mapping["ground"].get("grass", []),
        T_FORESTF:mapping["ground"].get("forest", []),
        T_DESERT: mapping["ground"].get("desert", []),
        T_CAVEF:  mapping["ground"].get("cave", []),
    }
    omap = {
        O_TREES:  mapping["overlay"].get("trees", []),
        O_ROCKS:  mapping["overlay"].get("rocks", []),
        O_CACTUS: mapping["overlay"].get("cactus", []),
        O_CAVEDEC:mapping["overlay"].get("cavedec", []),
        O_FOAM:   mapping["overlay"].get("foam", []),
    }

    for y in range(h):
        for x in range(w):
            gid = int(ground[y, x])
            gtiles = gmap.get(gid, [])
            if gtiles:
                c, r = pick(rng, gtiles)
                out.alpha_composite(get_tile((c,r)), (x*sheet.tile, y*sheet.tile))

            oid = int(overlay[y, x])
            if oid != O_NONE:
                otiles = omap.get(oid, [])
                if otiles:
                    c, r = pick(rng, otiles)
                    out.alpha_composite(get_tile((c,r)), (x*sheet.tile, y*sheet.tile))

    if scale != 1:
        out = out.resize((out.size[0]*scale, out.size[1]*scale), Image.NEAREST)
    return out


# =========================
# UI / Tile Mapper
# =========================
DEFAULT_MAPPING = {
    "ground": {
        "ocean": [],
        "fresh": [],
        "beach": [],
        "grass": [],
        "forest": [],
        "desert": [],
        "cave": [],
    },
    "overlay": {
        "trees": [],
        "rocks": [],
        "cactus": [],
        "cavedec": [],
        "foam": [],
    },
}

def add_coord(mapping: Dict, layer: str, key: str, coord: Tuple[int,int]):
    if coord not in mapping[layer][key]:
        mapping[layer][key].append(list(coord))

def remove_coord(mapping: Dict, layer: str, key: str, coord: Tuple[int,int]):
    arr = mapping[layer][key]
    mapping[layer][key] = [v for v in arr if not (v[0] == coord[0] and v[1] == coord[1])]


# =========================
# Streamlit
# =========================
st.set_page_config(page_title="Biome + River + Cave (2 layers)", layout="wide")
st.title("Mapas Procedurais (Biomas + Rios + Cavernas) com 2 Camadas — Tiles 32")

with st.sidebar:
    st.subheader("Tileset")
    up = st.file_uploader("PNG do tileset", type=["png"])
    tile_size = 32  # fixo como você pediu
    scale = st.selectbox("Zoom (pixel art)", [1,2,3,4], index=1)

    st.subheader("Tamanho do mapa")
    map_w = st.slider("Largura (tiles)", 40, 240, 120, 10)
    map_h = st.slider("Altura (tiles)", 30, 200, 80, 10)

    st.subheader("Seed")
    seed = st.number_input("Seed", value=1337, step=1)

    st.subheader("Biomas")
    sea_level = st.slider("Nível do mar", 0.05, 0.85, 0.35, 0.01)
    forest_moist = st.slider("Umidade p/ floresta", 0.20, 0.95, 0.70, 0.01)
    desert_dry = st.slider("Secura p/ deserto", 0.05, 0.80, 0.25, 0.01)

    st.subheader("Rios e Lagos")
    n_rivers = st.slider("Qtd rios", 0, 10, 3, 1)
    n_lakes = st.slider("Qtd lagos", 0, 12, 4, 1)

    st.subheader("Caverna")
    cave_ratio = st.slider("Área de caverna (fração)", 0.00, 0.60, 0.15, 0.01)
    cave_lake_level = st.slider("Chance de lago na caverna", 0.01, 0.40, 0.10, 0.01)

    st.divider()
    do_generate = st.button("Gerar mapa", type="primary", use_container_width=True)


if not up:
    st.info("Envie um tileset PNG (32px por tile).")
    st.stop()

tileset = Image.open(up)
sheet = TileSheet.from_image(tileset, tile=32)

# mapping em session
if "mapping" not in st.session_state:
    st.session_state["mapping"] = json.loads(json.dumps(DEFAULT_MAPPING))  # deep copy

mapping = st.session_state["mapping"]

tab1, tab2 = st.tabs(["1) Tile Mapper (sem precisar me passar coords)", "2) Gerador"])

with tab1:
    st.subheader("Tile Mapper")
    st.write(f"Tileset detectado: **{sheet.cols} cols × {sheet.rows} rows** (tile=32)")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### Selecionar tile (col, row)")
                # --- estado do seletor ---
        if "tile_c" not in st.session_state:
            st.session_state["tile_c"] = 0
        if "tile_r" not in st.session_state:
            st.session_state["tile_r"] = 0
        
        # garante dentro do range
        st.session_state["tile_c"] = int(np.clip(st.session_state["tile_c"], 0, sheet.cols - 1))
        st.session_state["tile_r"] = int(np.clip(st.session_state["tile_r"], 0, sheet.rows - 1))
        
        def _wrap(v: int, lo: int, hi: int) -> int:
            # wrap inclusivo [lo..hi]
            n = hi - lo + 1
            return lo + ((v - lo) % n)
        
        def move(dc: int = 0, dr: int = 0):
            st.session_state["tile_c"] = _wrap(st.session_state["tile_c"] + dc, 0, sheet.cols - 1)
            st.session_state["tile_r"] = _wrap(st.session_state["tile_r"] + dr, 0, sheet.rows - 1)
            st.rerun()
        
        st.markdown("### Selecionar tile (col, row)")
        
        # step de pulo vertical
        step = st.selectbox("Pulo vertical", [1, 5, 10, 25, 50, 100], index=2)
        
        # layout: botões + inputs
        nav1, nav2 = st.columns([1.2, 2.8], gap="large")
        
        with nav1:
            # Controles direcionais
            up = st.button("▲", use_container_width=True)
            left, right = st.columns(2, gap="small")
            with left:
                b_left = st.button("◀", use_container_width=True)
            with right:
                b_right = st.button("▶", use_container_width=True)
            down = st.button("▼", use_container_width=True)
        
            # Pulos rápidos
            st.markdown("**Pulos**")
            j1, j2 = st.columns(2, gap="small")
            with j1:
                if st.button(f"-{step}", use_container_width=True):
                    move(dr=-step)
            with j2:
                if st.button(f"+{step}", use_container_width=True):
                    move(dr=+step)
        
            # aplica cliques direcionais
            if up:
                move(dr=-1)
            if down:
                move(dr=+1)
            if b_left:
                move(dc=-1)
            if b_right:
                move(dc=+1)
        
        with nav2:
            # Inputs diretos (sem arrastar)
            c_in = st.number_input("Coluna (c)", min_value=0, max_value=sheet.cols - 1,
                                   value=int(st.session_state["tile_c"]), step=1)
            r_in = st.number_input("Linha (r)", min_value=0, max_value=sheet.rows - 1,
                                   value=int(st.session_state["tile_r"]), step=1)
        
            # se o usuário digitou, atualiza
            if c_in != st.session_state["tile_c"] or r_in != st.session_state["tile_r"]:
                st.session_state["tile_c"] = int(c_in)
                st.session_state["tile_r"] = int(r_in)
                st.rerun()
        
        # usa o estado para preview
        col = st.session_state["tile_c"]
        row = st.session_state["tile_r"]
        tile_preview = sheet.tile_at(col, row)
        st.image(tile_preview.resize((128, 128), Image.NEAREST), caption=f"Tile (c={col}, r={row})")

        st.image(tile_preview.resize((128,128), Image.NEAREST), caption=f"Tile (c={col}, r={row})")

        layer = st.selectbox("Layer", ["ground", "overlay"])
        key_options = list(mapping[layer].keys())
        key = st.selectbox("Categoria", key_options)

        a, b, d = st.columns([1,1,1])
        with a:
            if st.button("Adicionar", use_container_width=True):
                add_coord(mapping, layer, key, (col, row))
        with b:
            if st.button("Remover", use_container_width=True):
                remove_coord(mapping, layer, key, (col, row))
        with d:
            if st.button("Limpar categoria", use_container_width=True):
                mapping[layer][key] = []

        st.divider()
        st.markdown("### Export/Import mapping")
        st.download_button(
            "Baixar mapping.json",
            data=json.dumps(mapping, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="mapping.json",
            mime="application/json",
            use_container_width=True,
        )
        imp = st.file_uploader("Importar mapping.json", type=["json"], key="imp_map")
        if imp:
            try:
                st.session_state["mapping"] = json.load(imp)
                st.success("Mapping importado.")
                st.rerun()
            except Exception as e:
                st.error(f"Falha ao importar: {e}")

    with c2:
        st.markdown("### Amostra do tileset (para você navegar)")
        sample_cols = min(sheet.cols, 10)
        sample_rows = min(sheet.rows, 8)
        sample = Image.new("RGBA", (sample_cols*sheet.tile, sample_rows*sheet.tile), (0,0,0,0))
        for r in range(sample_rows):
            for c in range(sample_cols):
                sample.alpha_composite(sheet.tile_at(c, r), (c*sheet.tile, r*sheet.tile))
        st.image(sample.resize((sample.size[0]*2, sample.size[1]*2), Image.NEAREST),
                 caption="Amostra topo-esquerda (2x)")

        st.divider()
        st.markdown("### O que mapear (mínimo recomendado)")
        st.write("""
**Ground (obrigatório p/ ficar legal):**
- ocean (mar)
- fresh (água doce)
- beach (areia/costa)
- grass (grama/terra)
- desert (areia do deserto)
- forest (chão de floresta ou grama mais escura)
- cave (chão de caverna/pedra)

**Overlay (opcional, mas dá vida):**
- trees (árvores)
- rocks (rochas)
- cactus (cactos)
- cavedec (estalactites/pedrinhas)
- foam (espuma do mar)
""")

        st.markdown("### Estado atual do mapping")
        st.json(mapping)


with tab2:
    st.subheader("Gerador")

    # valida mapping mínimo
    required = ["ocean","fresh","beach","grass","desert","forest","cave"]
    missing = [k for k in required if len(mapping["ground"].get(k, [])) == 0]
    if missing:
        st.warning(f"Mapeie pelo menos 1 tile para: {', '.join(missing)} (aba Tile Mapper).")
        st.stop()

    if not do_generate:
        st.info("Clique em **Gerar mapa** na barra lateral.")
        st.stop()

    ground, overlay, objects = generate_world(
        w=map_w, h=map_h, seed=int(seed),
        sea_level=float(sea_level),
        forest_moist=float(forest_moist),
        desert_dry=float(desert_dry),
        n_rivers=int(n_rivers),
        n_lakes=int(n_lakes),
        cave_ratio=float(cave_ratio),
        cave_lake_level=float(cave_lake_level),
    )

    img = render_layers(sheet, ground, overlay, objects, mapping, seed=seed, scale=scale)

    left, right = st.columns([2, 1])

    with left:
        st.image(img, use_container_width=True, caption="Mapa final (ground + overlay)")

    with right:
        st.markdown("### Downloads")
        st.download_button(
            "Baixar PNG",
            data=pil_bytes(img, "PNG"),
            file_name=f"map_seed_{int(seed)}.png",
            mime="image/png",
            use_container_width=True,
        )

        # export simples dos grids
        payload = {
            "seed": int(seed),
            "w": int(map_w),
            "h": int(map_h),
            "ground": ground.tolist(),
            "overlay": overlay.tolist(),
        }
        st.download_button(
            "Baixar grids (JSON)",
            data=json.dumps(payload).encode("utf-8"),
            file_name=f"grids_seed_{int(seed)}.json",
            mime="application/json",
            use_container_width=True,
        )

        st.markdown("### Debug rápido")
        st.caption("IDs de ground: 0=ocean 1=fresh 2=beach 3=grass 4=forest 5=desert 6=cave")
        st.caption("IDs overlay: 0=none 1=trees 2=rocks 3=cactus 4=cavedec 5=foam")
