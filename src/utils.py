import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.ops import unary_union

BASE_CRS = 'EPSG:3857'
GEO_CRS = 'EPSG:4326'


def open_asc(file_path):
    with rasterio.open(file_path, 'r') as src:
        data = src.read(1)
        return np.ma.masked_array(data, data == -9999)


def replace_empty_list(pr):
    return pr if str(pr) != '[]' else None


def replace_none_nan(attr_dict):
    """Replace None or NaN values in a dictionary with 0."""
    return {k: (0 if v is None or (isinstance(v, float) and np.isnan(v)) else v) for k, v in attr_dict.items()}


def clip_raster(geom, raster_src):
    try:
        if not geom.is_empty and geom.is_valid:
            mask = geometry_mask([geom], out_shape=raster_src.shape,
                                  transform=raster_src.meta.copy()['transform'], invert=False)
            masked_data = np.ma.masked_array(raster_src.read(1), mask)
            return np.nan if np.all(masked_data.mask) else masked_data
        return np.nan
    except Exception as e:
        print(f"Error in clip_raster: {e}")
        return np.nan


def normalize(variable):
    return (variable - np.nanmin(variable)) / (np.nanmin(variable) + np.nanmax(variable) + 1e-10)


def minmax_values_sr(gdf):
    """Returns (sr_min, sr_max, or_min, or_max, pr_min, pr_max, tr_min, tr_max) across gdf."""
    for column in gdf.columns:
        gdf[column] = gdf[column].apply(lambda x: 0 if x is None or x == [] else x)

    # SR_owner/SR_neighbors are per-neighbor lists -- combine and flatten before min/max
    combined_sr = [
        [sum(x) for x in zip(owner, neighbors)] if isinstance(owner, list) else owner + neighbors
        for owner, neighbors in zip(gdf['SR_owner'], gdf['SR_neighbors'])
    ]
    sr_flat = [v for sub in combined_sr if isinstance(sub, list) for v in sub] + [v for v in combined_sr if not isinstance(v, list)]
    or_flat = [v for sub in gdf['OR'] if isinstance(sub, list) for v in sub] + [v for v in gdf['OR'] if not isinstance(v, list)]

    sr_min, sr_max = np.nanmin(sr_flat), np.nanmax(sr_flat)
    or_min, or_max = np.nanmin(or_flat), np.nanmax(or_flat)
    pr_min, pr_max = np.nanmin(gdf['PR']), np.nanmax(gdf['PR'])
    tr_min, tr_max = np.nanmin(gdf['TR']), np.nanmax(gdf['TR'])

    return sr_min, sr_max, or_min, or_max, pr_min, pr_max, tr_min, tr_max


def calculate_distance(poly1, poly2):
    return poly1.distance(poly2)


def compute_avg_ssd(bldgs):
    """Average separation distance between each building and its nearest neighbor."""
    average_distances = []
    for _, row in bldgs.iterrows():
        geom = row['geometry']
        closest = bldgs[bldgs.geometry != geom]['geometry'].distance(geom).idxmin()
        average_distances.append(calculate_distance(geom, bldgs.loc[closest]['geometry']))

    avg_ssd = sum(average_distances) / len(average_distances)
    print("Average SSD = {:.3f}m for {} buildings.".format(avg_ssd, len(bldgs)))
    return avg_ssd


def compute_avg_parcel_size(parcels):
    avg_parcel_area = sum(parcels.area) / len(parcels)
    print("Average parcel area = {:.3f} m^2 for {} parcels.".format(avg_parcel_area, len(parcels)))
    return avg_parcel_area


def group_bldgs_by_parcel(all_bldgs_tax):
    """
    Collapses multiple buildings sharing the same parcel (e.g. House + ADU) into a single row per parcel to treat one property as one node. 
    **bldg_to_parcel tags every row with 'parcel_idx' and 'building_type' 
    ***Unions the DS geometries within each group, keeping the 'house' row as the base.
    """
    results = []
    for _, group in all_bldgs_tax.groupby('parcel_idx'):
        house_rows = group[group['building_type'] == 'house']
        base_row = house_rows.iloc[0].copy() if len(house_rows) > 0 else group.iloc[0].copy()

        geometries = group['geometry'].tolist()
        if len(geometries) > 1:
            base_row['geometry'] = unary_union(geometries)
            base_row['building_count'] = len(geometries)
            if len(house_rows) > 0:
                other_count = len(geometries) - len(house_rows)
                base_row['building_type'] = f'house_and_{other_count}_others' if other_count > 0 else 'house'
            else:
                base_row['building_type'] = f'combined_{len(geometries)}_buildings'
        else:
            base_row['building_count'] = 1

        results.append(base_row)

    return gpd.GeoDataFrame(results, crs=all_bldgs_tax.crs).reset_index(drop=True)
