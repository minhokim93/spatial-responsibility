"""
One-time data-export script for the interactive neighborhood visualization.
Regenerates gdf_results / G_pre / G_pre_OR from this repo's own src/ modules
and data/ shapefiles, then writes a single self-contained data.json consumed
by the HTML/JS app in this folder. Not part of the visualization app itself.
"""
import os
import sys
import json
import math
import random as pyrandom

import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.ops import unary_union

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))

from utils import SQM_TO_SQFT, group_bldgs_by_parcel  # noqa: E402
from compute_sr import bldg_to_parcel, compute_responsibility  # noqa: E402
from network import build_network_v2  # noqa: E402

DATA_DIR = os.path.join(REPO, 'data')
OUT_PATH = os.path.join(REPO, 'visualization', 'data.json')

RNG_SEED = 7

# ---------------------------------------------------------------------------
# 1. Load, match buildings to parcels, buffer to DS, group multi-building
#    parcels -- reusing the exact pipeline from notebooks/main.ipynb cells
#    3, 4, 7 so this matches the paper's own data exactly.
# ---------------------------------------------------------------------------
buffer_distance = 30  # DS buffer distance in feet

bldgs = gpd.read_file(os.path.join(DATA_DIR, 'bldgs.shp'))
parcels = gpd.read_file(os.path.join(DATA_DIR, 'parcels.shp'))

all_bldgs_tax = bldg_to_parcel(bldgs, parcels)
all_bldgs_tax['DS_30'] = all_bldgs_tax.geometry.buffer(buffer_distance / 3.281)  # ft -> m
all_bldgs_tax.rename(columns={'geometry': 'bldg_geometry', 'DS_30': 'geometry'}, inplace=True)
all_bldgs_tax = all_bldgs_tax.set_geometry('geometry')

all_bldgs_tax_grouped = group_bldgs_by_parcel(all_bldgs_tax).reset_index(drop=True)

print(f"Raw buildings: {len(all_bldgs_tax)}  ->  network nodes (one per parcel): {len(all_bldgs_tax_grouped)}")

# group_bldgs_by_parcel collapses parcels with >1 structure (house + ADU/
# garage/etc.) into a single node, since PR/SR/OR and the network model one
# property as one node -- that's correct and matches the paper. But it means
# a handful of individual building footprints (5 parcels x 2 buildings here)
# would otherwise never get drawn on the map at all. Recover the mapping
# from each RAW building to the node it belongs to (same groupby order
# group_bldgs_by_parcel itself uses, so the ids line up with gdf_results'
# row order) so the app can render all raw footprints individually while
# still routing clicks/selection to the shared underlying node.
parcel_idx_to_node_id = {parcel_idx: node_id for node_id, (parcel_idx, _) in enumerate(all_bldgs_tax.groupby('parcel_idx'))}
assert len(parcel_idx_to_node_id) == len(all_bldgs_tax_grouped)
all_bldgs_tax['node_id'] = all_bldgs_tax['parcel_idx'].map(parcel_idx_to_node_id)

# ---------------------------------------------------------------------------
# 2. Responsibility metrics, using the same ROS raster as the paper.
# ---------------------------------------------------------------------------
risk_path = os.path.join(DATA_DIR, 'Outputs', 'ROS_nomit.tif')
gdf_results = compute_responsibility(all_bldgs_tax_grouped, risk=risk_path, plot=False)
gdf_results = gdf_results.reset_index(drop=True)


def clean(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return [float(x) for x in v]
    if isinstance(v, float) and math.isnan(v):
        return []
    return [float(v)]


def safe_sum(v):
    lst = clean(v)
    return float(np.nansum(lst)) if lst else 0.0


# ---------------------------------------------------------------------------
# 4. Per-node AREA-only PR/SR/OR (ft^2), independent of hazard weighting --
#    matches Area_SR / Area_OR used elsewhere in the paper's supplementary
#    figures. Recomputed directly from geometry rather than compute_SR's
#    risk-weighted intermediate values.
# ---------------------------------------------------------------------------
n = len(gdf_results)
sindex = gdf_results.sindex
pr_area = [0.0] * n
sr_area = [0.0] * n
or_area = [0.0] * n
pr_geom = [None] * n
sr_geom = [None] * n
or_geom = [None] * n

for i in range(n):
    D_i = gdf_results.iloc[i].geometry
    t_i = gdf_results.iloc[i]['tax_geometry']
    candidates = [j for j in sindex.intersection(D_i.bounds) if j != i]
    neighbor_ds = [gdf_results.iloc[j].geometry for j in candidates if D_i.intersects(gdf_results.iloc[j].geometry)]

    # PR region: (D_i n t_i) \ union(overlapping D_j)
    if t_i is not None:
        d_own = D_i.intersection(t_i)
        pr_region = d_own.difference(unary_union(neighbor_ds)) if neighbor_ds else d_own
        if not pr_region.is_empty:
            pr_area[i] = pr_region.area * SQM_TO_SQFT
            pr_geom[i] = pr_region

    # SR region: union over neighbors of (D_i n D_j n t_i)
    sr_pieces = []
    for j in candidates:
        D_j = gdf_results.iloc[j].geometry
        if D_i.intersects(D_j) and t_i is not None:
            ov = D_i.intersection(D_j).intersection(t_i)
            if not ov.is_empty:
                sr_pieces.append(ov)
    if sr_pieces:
        sr_region = unary_union(sr_pieces)
        sr_area[i] = sr_region.area * SQM_TO_SQFT
        sr_geom[i] = sr_region

    # OR region (IN direction, "what i owes"): union over all j of (D_j n t_i \ D_i)
    or_pieces = []
    if t_i is not None:
        for j in gdf_results.index:
            if j == i:
                continue
            D_j = gdf_results.iloc[j].geometry
            ov = D_j.intersection(t_i)
            if not ov.is_empty:
                ov = ov.difference(D_i)
                if not ov.is_empty:
                    or_pieces.append(ov)
    if or_pieces:
        or_region = unary_union(or_pieces)
        or_area[i] = or_region.area * SQM_TO_SQFT
        or_geom[i] = or_region

gdf_results['PR_area'] = pr_area
gdf_results['SR_area_total'] = sr_area
gdf_results['OR_area_total'] = or_area
gdf_results['SR_owner_total'] = gdf_results['SR_owner'].apply(safe_sum)
gdf_results['OR_total'] = gdf_results['OR'].apply(safe_sum)

# ---------------------------------------------------------------------------
# 5. Directed networks (SR, OR) -- same edge-direction convention fixed
#    earlier this session: SR edge (i, j) carries SR_owner_ij (i's own-parcel
#    share); OR edge (k, i) carries what i owes, i.e. arrow points TO whoever
#    owes because the hazard sits on their own land.
# ---------------------------------------------------------------------------
G_sr = build_network_v2(gdf_results, node_variable='PR', edge_variable='SR',
                         crs1='EPSG:3857', crs2='EPSG:4326', directed=True)
G_or = build_network_v2(gdf_results, node_variable='PR', edge_variable='OR',
                         crs1='EPSG:3857', crs2='EPSG:4326', directed=True)

sr_edges = [{'source': int(u), 'target': int(v), 'weight': float(d['SR'])}
            for u, v, d in G_sr.edges(data=True) if 'SR' in d]
or_edges = [{'source': int(u), 'target': int(v), 'weight': float(d['OR'])}
            for u, v, d in G_or.edges(data=True) if 'OR' in d]


# ---------------------------------------------------------------------------
# 6. Precompute deterministic removal ORDER for 3 strategies x 2 networks,
#    so the browser-side slider just replays a fixed sequence (no simulation
#    logic needed client-side beyond "remove the first k edges").
# ---------------------------------------------------------------------------
def targeted_order(edges):
    return [e for e in sorted(edges, key=lambda e: -e['weight'])]


def random_order(edges, seed):
    rng = pyrandom.Random(seed)
    shuffled = edges[:]
    rng.shuffle(shuffled)
    return shuffled


def localized_order(edges, seed):
    rng = pyrandom.Random(seed)
    remaining = edges[:]
    order = []
    frontier = set()
    while remaining:
        candidate_idxs = [i for i, e in enumerate(remaining) if e['source'] in frontier or e['target'] in frontier]
        if not candidate_idxs:
            nodes = list({e['source'] for e in remaining} | {e['target'] for e in remaining})
            seed_node = rng.choice(nodes)
            frontier = {seed_node}
            candidate_idxs = [i for i, e in enumerate(remaining) if e['source'] == seed_node or e['target'] == seed_node]
        pick_i = rng.choice(candidate_idxs)
        e = remaining.pop(pick_i)
        order.append(e)
        frontier.add(e['source'])
        frontier.add(e['target'])
    return order


networks_order = {}
for label, edges in [('SR', sr_edges), ('OR', or_edges)]:
    networks_order[label] = {
        'random': [{'source': e['source'], 'target': e['target']} for e in random_order(edges, RNG_SEED)],
        'localized': [{'source': e['source'], 'target': e['target']} for e in localized_order(edges, RNG_SEED)],
        'targeted': [{'source': e['source'], 'target': e['target']} for e in targeted_order(edges)],
    }

# ---------------------------------------------------------------------------
# 7. Geometry -> plain coordinate arrays (EPSG:3857 meters; app.js normalizes
#    to screen space). MultiPolygons -> list of rings.
# ---------------------------------------------------------------------------
def polygon_rings(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == 'MultiPolygon':
        geoms = list(geom.geoms)
    elif geom.geom_type == 'GeometryCollection':
        geoms = [g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
        expanded = []
        for g in geoms:
            expanded.extend(list(g.geoms) if g.geom_type == 'MultiPolygon' else [g])
        geoms = expanded
    elif geom.geom_type == 'Polygon':
        geoms = [geom]
    else:
        geoms = []
    rings = []
    for g in geoms:
        if g.is_empty:
            continue
        rings.append([[round(c[0], 2), round(c[1], 2)] for c in g.exterior.coords])
    return rings


nodes = []
for i in range(n):
    row = gdf_results.iloc[i]
    c = row['bldg_geometry'].centroid
    nodes.append({
        'id': int(i),
        'centroid': [round(c.x, 2), round(c.y, 2)],
        'building': polygon_rings(row['bldg_geometry']),
        'parcel': polygon_rings(row['tax_geometry']),
        'ds': polygon_rings(row.geometry),
        'pr_region': polygon_rings(pr_geom[i]),
        'sr_region': polygon_rings(sr_geom[i]),
        'or_region': polygon_rings(or_geom[i]),
        'PR_resp': round(float(row['PR']) if row['PR'] else 0.0, 2),
        'SR_resp': round(float(row['SR_owner_total']), 2),
        'OR_resp': round(float(row['OR_total']), 2),
        'PR_area': round(float(row['PR_area']), 2),
        'SR_area': round(float(row['SR_area_total']), 2),
        'OR_area': round(float(row['OR_area_total']), 2),
    })

# Every raw (pre-grouping) building footprint, individually, tagged with
# the network node it belongs to -- so all 70 structures are drawn on the
# map even though 5 pairs share a single node (and therefore a single set
# of PR/SR/OR values and click behavior). building_id (0..69) is a distinct
# per-structure number, deliberately separate from node_id (0..64, one per
# parcel) -- the two only coincide for the 60 parcels with a single
# structure; for the 5 merged parcels, two different building_ids map to
# the same node_id.
raw_buildings = []
for building_id, (_, row) in enumerate(all_bldgs_tax.iterrows()):
    c = row['bldg_geometry'].centroid
    raw_buildings.append({
        'building_id': int(building_id),
        'node_id': int(row['node_id']),
        'centroid': [round(c.x, 2), round(c.y, 2)],
        'building': polygon_rings(row['bldg_geometry']),
    })

out = {
    'nodes': nodes,
    'raw_buildings': raw_buildings,
    'edges': {'SR': sr_edges, 'OR': or_edges},
    'removal_order': networks_order,
}

with open(OUT_PATH, 'w') as f:
    json.dump(out, f)

# Also emit as a plain <script>-loadable JS file so index.html works when
# opened directly via file:// (fetch() of a local .json is blocked by CORS
# in most browsers without a local server; a <script src="data.js"> is not).
JS_OUT_PATH = os.path.join(REPO, 'visualization', 'data.js')
with open(JS_OUT_PATH, 'w') as f:
    f.write('const RESPONSIBILITY_DATA = ')
    json.dump(out, f)
    f.write(';\n')

print(f"Wrote {OUT_PATH} and {JS_OUT_PATH}  ({len(nodes)} nodes, {len(sr_edges)} SR edges, {len(or_edges)} OR edges)")
