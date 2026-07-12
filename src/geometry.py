"""
Neighborhood region geometry for PR, SR, and OR
"""

import geopandas as gpd
from shapely.ops import unary_union


def compute_component_regions(gdf_results, component='PR'):
    r"""
    For the entire neighborhood, computes individual polygon regions that count toward PR, SR, or OR     
    Returns a GeoDataFrame of regions in gdf_results
    """
    assert component in ('PR', 'SR', 'OR'), "component must be 'PR', 'SR', or 'OR'"

    sindex = gdf_results.sindex
    regions = []

    for i in gdf_results.index:

        row_i = gdf_results.iloc[i]
        D_i = row_i.geometry
        t_i = row_i['tax_geometry']

        ## PR
        if component == 'PR':
            
            # DS candidate search
            candidates = [j for j in sindex.intersection(D_i.bounds) if j != i]
            neighbor_ds = [gdf_results.iloc[j].geometry for j in candidates
                           if D_i.intersects(gdf_results.iloc[j].geometry)]
            
            # PR must stay on the owner's own parcel (matches compute_SR's overlap_PR) 
            D_i_own_parcel = D_i.intersection(t_i) if t_i is not None else D_i
            pr = D_i_own_parcel.difference(unary_union(neighbor_ds)) if neighbor_ds else D_i_own_parcel
            if not pr.is_empty:
                regions.append(pr)

        ## SR
        elif component == 'SR':
            candidates = [j for j in sindex.intersection(D_i.bounds) if j != i]
            for j in candidates:
                if j <= i:  # only add once per pair for SR link
                    continue
                D_j = gdf_results.iloc[j].geometry
                if D_i.intersects(D_j):
                    overlap = D_i.intersection(D_j)
                    if not overlap.is_empty:
                        regions.append(overlap)

        ## OR
        elif component == 'OR':
            for j in gdf_results.index:
                if j == i:
                    continue
                t_j = gdf_results.iloc[j]['tax_geometry']
                if t_j is None:
                    continue
                D_j = gdf_results.iloc[j].geometry
                intersection = D_i.intersection(t_j)
                if not intersection.is_empty:
                    or_region = intersection.difference(D_j)
                    if not or_region.is_empty:
                        regions.append(or_region)

    return gpd.GeoDataFrame(geometry=regions, crs=gdf_results.crs)
