"""
Spatial responsibility metrics: Personal (PR), Shared (SR), and Owed (OR) Responsibility

- compute_SR: shared responsibility
- compute_OR: owed responsibility for both directions
- compute_responsibility: Computes PR, SR, OR, and (total) TR for every parcel
- bldg_to_parcel: Many-to-one mapping of building footprints to tax parcels polygons
- preprocess: load buildings/parcels and build defensible space buffers
"""

import os
import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import geopandas as gpd
import rasterio
from shapely.geometry import MultiPolygon

from utils import *


def compute_SR(gdf, index, polygon, tax_polygon, plot, risk_src=None):
    """Shared and personal responsibility for parcel index."""
    count = 0
    overlap_ids = []
    overlap_proportion = []
    SR_owner = []
    SR_neighbors = []
    PR = []
    indiv_overlap_areas = []
    indiv_overlap_polygons = []
    overlaps_tax_owner = []
    overlaps_tax_neighbor = []
    overlaps_tax_owner_polygons = []
    overlaps_tax_neighbor_polygons = []
    other_polygons = []
    other_tax_polygons = []

    for other_index in gdf.sindex.intersection(polygon.bounds):
        if index == other_index:
            continue

        other_row = gdf.iloc[other_index]
        other_polygon = other_row['geometry']          # DS30
        other_tax_polygon = other_row['tax_geometry']   # tax parcel

        if polygon.intersects(other_polygon):
            overlap = polygon.intersection(other_polygon)  # total SR area

            if not overlap.is_empty and overlap.geom_type not in ('LineString', 'Point'):
                overlap_tax_owner = overlap.intersection(tax_polygon)           # owner's SR area
                overlap_tax_neighbor = overlap.intersection(other_tax_polygon)  # neighbor's SR area

                if risk_src:
                    if overlap_tax_neighbor is not None and not overlap_tax_neighbor.is_empty:
                        mean_risk_overlap_tax_neighbor = np.mean(clip_raster(overlap_tax_neighbor, risk_src))
                    else:
                        mean_risk_overlap_tax_neighbor = 0

                    if overlap_tax_owner is not None and not overlap_tax_owner.is_empty:
                        mean_risk_overlap_tax_owner = np.mean(clip_raster(overlap_tax_owner, risk_src))
                    else:
                        mean_risk_overlap_tax_owner = 0

                    other_polygons.append(other_polygon)
                    other_tax_polygons.append(other_tax_polygon)
                else:
                    mean_risk_overlap_tax_owner = 1
                    mean_risk_overlap_tax_neighbor = 1

                indiv_overlap_areas.append(overlap.area)
                overlaps_tax_owner.append(overlap_tax_owner.area * SQM_TO_SQFT * mean_risk_overlap_tax_owner)
                overlaps_tax_neighbor.append(overlap_tax_neighbor.area * SQM_TO_SQFT * mean_risk_overlap_tax_neighbor)
                overlap_ids.append(other_index)
                indiv_overlap_polygons.append(overlap)
                overlaps_tax_owner_polygons.append(overlap_tax_owner)
                overlaps_tax_neighbor_polygons.append(overlap_tax_neighbor)

                count += 1

    # PR_area is the raw geometric area of the PR region [ft^2], with no risk
    # weighting -- independent of whether a risk raster is supplied, unlike
    # PR itself (Area x mean_risk, [ft^3/min]). Used for the area-only
    # network figures (weight_by='area'), where edges are already unweighted
    # by risk and the node color should stay in the same units [ft^2].
    PR_area = 0

    if count > 0:
        overlapping_polygons_gdf = gpd.GeoDataFrame(geometry=indiv_overlap_polygons, columns=['geometry']).unary_union
        overlap_proportion = 0
        SR_owner = list(overlaps_tax_owner)
        SR_neighbors = list(overlaps_tax_neighbor)

        # PR must stay on the owner's own parcel
        overlap_PR = polygon.intersection(tax_polygon).difference(overlapping_polygons_gdf)
        PR_area = overlap_PR.area * SQM_TO_SQFT if not overlap_PR.is_empty else 0

        if risk_src:
            risk_overlap_PR = clip_raster(overlap_PR, risk_src)
            PR = np.mean(risk_overlap_PR) * overlap_PR.area * SQM_TO_SQFT if risk_overlap_PR is not None else 0
        else:
            PR = overlap_PR.area * SQM_TO_SQFT

        if overlap_PR is None or overlap_PR == []:
            overlap_PR = 0
            PR = 0

    elif count == 0:
        overlap_proportion = 0
        SR_owner = 0
        SR_neighbors = 0

        # No SR-overlapping neighbors
        overlap_PR = polygon.intersection(tax_polygon)
        if not overlap_PR.is_empty:
            PR_area = overlap_PR.area * SQM_TO_SQFT
            if risk_src:
                risk_overlap_PR = clip_raster(overlap_PR, risk_src)
                PR = np.mean(risk_overlap_PR) * overlap_PR.area * SQM_TO_SQFT if risk_overlap_PR is not None else 0
            else:
                PR = overlap_PR.area * SQM_TO_SQFT
        else:
            PR = 0

    if PR == []:
        PR = None

    return count, overlap_proportion, overlap_ids, indiv_overlap_areas, SR_owner, SR_neighbors, PR, PR_area, overlaps_tax_owner_polygons, overlaps_tax_neighbor_polygons, other_polygons, other_tax_polygons


def compute_OR(gdf, index, polygon, tax_polygon, risk_src=None):
    """
    Both directed owed responsibility quantities for parcel index
    """
    owed_count_in = 0
    owed_ids_in = []
    owed_overlap_areas_in = []
    owed_overlap_proportion_in = []  # OR_{i<-j}: this owner owes each of these

    owed_count_out = 0
    owed_ids_out = []
    owed_overlap_areas_out = []
    owed_overlap_proportion_out = []  # OR_{i->j}: each neighbor owes this one

    def _mean_risk(geom):
        return np.mean(clip_raster(geom, risk_src)) if risk_src else 1

    for other_index in gdf.index:
        if other_index == index:
            continue

        other_row = gdf.loc[other_index]
        other_polygon = other_row['geometry']          # D_j
        other_tax_polygon = other_row['tax_geometry']  # t_j

        # IN: (D_j n t_i) \ D_i
        intersection_in = other_polygon.intersection(tax_polygon)
        if not intersection_in.is_empty:
            overlap_in = intersection_in.difference(polygon)
            if not overlap_in.is_empty:
                mean_risk_in = _mean_risk(overlap_in)
                owed_overlap_areas_in.append(overlap_in.area)
                owed_overlap_proportion_in.append(overlap_in.area * SQM_TO_SQFT * mean_risk_in)
                owed_ids_in.append(other_index)
                owed_count_in += 1

        # OUT: (D_i n t_j) \ D_j
        if other_tax_polygon is not None:
            intersection_out = polygon.intersection(other_tax_polygon)
            if not intersection_out.is_empty:
                overlap_out = intersection_out.difference(other_polygon)
                if not overlap_out.is_empty:
                    mean_risk_out = _mean_risk(overlap_out)
                    owed_overlap_areas_out.append(overlap_out.area)
                    owed_overlap_proportion_out.append(overlap_out.area * SQM_TO_SQFT * mean_risk_out)
                    owed_ids_out.append(other_index)
                    owed_count_out += 1

    if owed_count_in > 0:
        owed_polygons_area_indiv_in = gdf.loc[owed_ids_in, 'geometry'].unary_union.area
    else:
        owed_polygons_area_indiv_in = 0
        owed_overlap_proportion_in = 0

    if owed_count_out > 0:
        owed_polygons_area_indiv_out = gdf.loc[owed_ids_out, 'geometry'].unary_union.area
    else:
        owed_polygons_area_indiv_out = 0
        owed_overlap_proportion_out = 0

    return (
        owed_count_in, owed_ids_in, owed_overlap_proportion_in, owed_overlap_areas_in, owed_polygons_area_indiv_in,
        owed_count_out, owed_ids_out, owed_overlap_proportion_out, owed_overlap_areas_out, owed_polygons_area_indiv_out,
    )


def compute_responsibility(gdf, risk=None, plot=False):
    """Computes PR, SR, OR (both directions), and TR = PR + SR_owner + OR for every row of input gdf."""
    gdf.sindex

    results = {
        'count': [], 'owed_count': [], 'owed_ID': [],
        'SR_indiv': [], 'SR_avg': [], 'SR_ID': [], 'SR_area': [], 'SR_area_avg': [],
        'SR_owner': [], 'SR_owner_avg': [], 'SR_neighbors': [], 'SR_neighbors_avg': [],
        'PR': [], 'PR_area': [],
        'OR': [], 'OR_avg': [],
        'owed_overlap_area': [], 'owed_overlap_area_indiv': [],
        'owed_count_out': [], 'owed_ID_out': [],
        'OR_out': [], 'OR_out_avg': [],
        'owed_overlap_area_out': [], 'owed_overlap_area_indiv_out': [],
    }

    risk_src = rasterio.open(risk) if risk is not None else None

    for index, row in tqdm.tqdm(gdf.iterrows(), total=len(gdf), desc='Processing ...'):
        polygon = row['geometry']          # DS30
        tax_polygon = row['tax_geometry']  # tax parcel

        count, overlap_proportion, overlap_ids, indiv_overlap_areas, SR_owner, SR_neighbors, PR, PR_area, owner_polys, neighbor_polys, other_polygons, other_tax_polygons = compute_SR(
            gdf, index, polygon, tax_polygon, plot, risk_src)
        (owed_count, owed_ids, owed_overlap_proportion, owed_overlap_areas, owed_polygons_area_indiv,
         owed_count_out, owed_ids_out, owed_overlap_proportion_out, owed_overlap_areas_out, owed_polygons_area_indiv_out) = compute_OR(
            gdf, index, polygon, tax_polygon, risk_src)

        results['count'].append(count)
        results['SR_indiv'].append(overlap_proportion)
        results['SR_avg'].append(np.nanmean(overlap_proportion))
        results['SR_ID'].append(overlap_ids)
        results['SR_area'].append(indiv_overlap_areas)
        results['SR_area_avg'].append(np.nanmean(indiv_overlap_areas))
        results['SR_owner'].append(SR_owner)
        results['SR_owner_avg'].append(np.nanmean(SR_owner))
        results['SR_neighbors'].append(SR_neighbors)
        results['SR_neighbors_avg'].append(np.nanmean(SR_neighbors))

        results['PR'].append(PR)
        results['PR_area'].append(PR_area)

        results['owed_count'].append(owed_count)
        results['owed_ID'].append(owed_ids)
        results['OR'].append(owed_overlap_proportion)
        results['OR_avg'].append(np.nanmean(np.array(owed_overlap_proportion)))
        results['owed_overlap_area'].append(owed_overlap_areas)
        results['owed_overlap_area_indiv'].append(owed_polygons_area_indiv)

        results['owed_count_out'].append(owed_count_out)
        results['owed_ID_out'].append(owed_ids_out)
        results['OR_out'].append(owed_overlap_proportion_out)
        results['OR_out_avg'].append(np.nanmean(np.array(owed_overlap_proportion_out)))
        results['owed_overlap_area_out'].append(owed_overlap_areas_out)
        results['owed_overlap_area_indiv_out'].append(owed_polygons_area_indiv_out)


    new_gdf = gdf.copy()
    for key, value in results.items():
        new_gdf[key] = value

    new_gdf['TR'] = [
        np.sum([np.sum(new_gdf.iloc[i]['PR']), np.sum(new_gdf.iloc[i]['SR_owner']), np.sum(new_gdf.iloc[i]['OR'])])
        for i in range(len(new_gdf))
    ]
    return new_gdf


def bldg_to_parcel(gdf1, gdf2):
    """
    Maps building footprints (gdf1) to tax parcels (gdf2), many-to-one, by largest overlap area. 
    Each building is tagged with 'parcel_idx' and 'building_type' ('house' for the largest building on a parcel, 'others' for the rest).
    """

    gdf2_sindex = gdf2.sindex
    parcel_buildings = {}
    unmatched_buildings = []

    for _, row in tqdm.tqdm(gdf1.iterrows(), total=len(gdf1)):
        building_polygon = row['geometry']
        best_parcel = None
        max_overlap_area = 0

        for idx in gdf2_sindex.intersection(building_polygon.bounds):
            intersection_area = building_polygon.intersection(gdf2.iloc[idx]['geometry']).area
            if intersection_area > max_overlap_area:
                max_overlap_area = intersection_area
                best_parcel = idx

        building_data = row.to_dict()
        building_data['building_area'] = building_polygon.area

        if best_parcel is not None:
            building_data['parcel_idx'] = best_parcel
            parcel_buildings.setdefault(best_parcel, []).append(building_data)
        else:
            building_data['building_type'] = 'unmatched'
            unmatched_buildings.append(building_data)

    results = []
    for parcel_idx, buildings in parcel_buildings.items():
        parcel_data = gdf2.iloc[parcel_idx].to_dict()
        parcel_data['tax_geometry'] = parcel_data.pop('geometry')

        buildings.sort(key=lambda x: x['building_area'], reverse=True)
        for i, building in enumerate(buildings):
            building.pop('building_area', None)
            building['building_type'] = 'house' if i == 0 else 'others'
            results.append({**building, **parcel_data})

    for building in unmatched_buildings:
        building.pop('building_area', None)
        results.append(building)

    if results:
        return gpd.GeoDataFrame(results, geometry='geometry')
    
    return gpd.GeoDataFrame(columns=list(gdf1.columns) + ['building_type'], geometry='geometry', crs=gdf1.crs)


def preprocess(base_path, buffer_distance=30, check=False, base_crs=BASE_CRS):
    """Loads bldgs.shp/parcels.shp from `base_path`, maps buildings to parcels, and buffers each to its defensible-space (DS) zone."""
    
    bldgs = gpd.read_file(os.path.join(base_path, 'bldgs.shp')).to_crs(base_crs)
    parcels = gpd.read_file(os.path.join(base_path, 'parcels.shp')).to_crs(base_crs)

    # check since some buildings or parcels are combined multipolygons that causes errors
    if any(isinstance(g['geometry'], MultiPolygon) for _, g in parcels.iterrows()):
        parcels = parcels.explode()

    # Map buildings to parcels
    all_bldgs_tax = bldg_to_parcel(bldgs, parcels)
    all_bldgs_tax['DS_30'] = all_bldgs_tax.geometry.buffer(buffer_distance / 3.281)  #TODO: Hardcoded conv
    all_bldgs_tax.rename(columns={'geometry': 'bldg_geometry', 'DS_30': 'geometry'}, inplace=True)
    all_bldgs_tax.geometry = all_bldgs_tax['geometry']

    if all_bldgs_tax.crs is None:
        all_bldgs_tax = all_bldgs_tax.set_crs(base_crs)

    if check:
        print("Missing tax parcels:", all_bldgs_tax.tax_geometry.isna().sum())
        print("CRS:", all_bldgs_tax.crs)

    return all_bldgs_tax