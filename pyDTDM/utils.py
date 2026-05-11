"""
Utility Functions for pyDTDM
=============================

This module provides a comprehensive collection of utility functions for geospatial 
data processing, plate tectonic analysis, and geological data manipulation. It supports
operations on raster and vector data, geometric transformations, nearest neighbor 
analyses, and file format conversions.

Core Functionality:
-------------------
- **Geospatial Processing**: Grid interpolation, KNN filling, geodesic distance calculations
- **Geometric Operations**: Polygon creation, point generation, subduction teeth geometry
- **File I/O**: NetCDF to GeoTIFF conversion, CPT colormap reading, raster operations
- **Plate Tectonics**: Topology processing, overriding plate identification, profile creation
- **Nearest Neighbor**: Geodesic joins, spatial indexing with BallTree
- **Plotting**: Reconstructed GeoDataFrame visualization, colorbar generation
- **Data Gridding**: Convert point data to regular grids with various statistics

Key Functions:
--------------
Grid Processing:
    - post_process_grid: Fill NaN gaps using KNN interpolation
    - df_to_NetCDF: Convert scattered points to gridded NetCDF
    - nan_gaussian_filter: Gaussian smoothing handling NaN values

Geometric:
    - get_subduction_teeth: Create triangular subduction teeth polygons
    - sjoin_nearest_geodesic_points: Geodesic nearest neighbor join
    - generate_mesh: Create icosahedral mesh for global sampling
    - multipoints_from_polygon: Generate point cloud from polygon

Plate Tectonics:
    - create_geodataframe_topologies: Convert pygplates topologies to GeoDataFrame
    - get_overriding_pid: Identify overriding plate IDs at subduction zones
    - poly_around_sub: Create polygons around subduction zones

File Operations:
    - nc_to_tiff: Convert NetCDF to GeoTIFF
    - readcpt: Load GMT/PyGMT color palette files
    - interpolate_and_save_as_geotiff: Temporal interpolation of rasters

Distance & Proximity:
    - haversine_distance: Great circle distance calculation
    - minimum_distance: Find nearest point in GeoDataFrame
    - calc_dist: Simple Euclidean distance

Statistics:
    - calculate_wma: Weighted moving average for GeoDataFrames
    - mean_gdfs: Average multiple GeoDataFrames

Visualization:
    - plotgdf: Plot reconstructed GeoDataFrames with plate features
    - plot_only_colorbar: Create standalone colorbars

Dependencies:
-------------
- pygplates: Plate reconstruction and topology operations
- geopandas/shapely: Vector geospatial operations
- rasterio/xarray: Raster data processing
- scikit-learn: Spatial indexing and interpolation
- geopy: Geodesic distance calculations

Author: Satyam Pratap Singh
Email: singhsatyampratap@gmail.com
License: GNU General Public License v3.0

Examples:
---------
>>> import pyDTDM.utils as utils
>>> import geopandas as gpd
>>> 
>>> # Geodesic nearest neighbor join
>>> result = utils.sjoin_nearest_geodesic_points(
...     gdf1=points_gdf,
...     gdf2=trenches_gdf,
...     k=1,
...     distance_col='distance_to_trench_m'
... )
>>> 
>>> # Generate global mesh
>>> lons, lats = utils.generate_mesh(refinement_levels=8)
>>> 
>>> # Convert NetCDF to GeoTIFF
>>> utils.nc_to_tiff('input.nc', 'output.tif')
"""

import re
import glob
import pygplates
import pandas as pd
import geopandas as gpd
import gplately
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from stripy.spherical_meshes import icosahedral_mesh
import time
from shapely.geometry import Point,Polygon
from shapely.prepared import prep
from joblib import Parallel, delayed
import shapely
import math
from scipy.stats import binned_statistic_2d
import xarray as xr
import rioxarray
import rasterio

# cptpath = 'cmaps'
import matplotlib.cm as cm
import os, sys
# sys.path.append(cptpath)
from .get_cpt import *
import cartopy.crs as ccrs
import warnings
Etopo_REED = get_cmap('ETOPO1-Reed.cpt')
from rasterio.transform import from_origin
import ptt

from geopy import Point as GeopyPoint
from geopy.distance import geodesic

from scipy.interpolate import interp1d

from matplotlib.colors import LinearSegmentedColormap
from sklearn.neighbors import BallTree
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter






def post_process_grid(data, n_neighbors=3, threshold_distance=5):
    """
    Fill NaN gaps in gridded data using K-Nearest Neighbors interpolation.
    
    This function intelligently fills missing values (NaN) in raster data by finding
    nearby valid points and interpolating values. Only fills gaps within a specified
    distance threshold to avoid unrealistic extrapolation.
    
    Parameters
    ----------
    data : xarray.DataArray
        Input grid with potential NaN values to fill
    n_neighbors : int, default=3
        Number of nearest neighbors to use for interpolation
    threshold_distance : int or float, default=5
        Maximum distance (in grid cells) to search for neighbors.
        NaN values farther than this from valid data remain unfilled
    
    Returns
    -------
    xarray.DataArray
        Grid with NaN values filled where possible, maintaining original
        coordinates and dimensions
    
    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> # Create sample grid with NaN values
    >>> data = xr.DataArray(
    ...     np.random.rand(100, 100),
    ...     dims=['y', 'x']
    ... )
    >>> data.values[40:50, 40:50] = np.nan
    >>> 
    >>> # Fill gaps
    >>> filled = post_process_grid(data, n_neighbors=5, threshold_distance=10)
    >>> print(f"Remaining NaN: {np.isnan(filled.values).sum()}")
    
    Notes
    -----
    - Uses cKDTree for efficient spatial indexing
    - Only fills NaN values close to valid data (within threshold_distance)
    - Preserves original xarray structure and metadata
    - Prints number of interpolated points for monitoring
    
    See Also
    --------
    scipy.spatial.cKDTree : Spatial indexing
    sklearn.neighbors.KNeighborsRegressor : K-NN interpolation
    """

    elevation = data

    # Get the coordinates of the valid data points and their values
    valid_points = np.column_stack(np.where(~np.isnan(elevation)))
    valid_values = elevation.values[~np.isnan(elevation)]

    # Get the coordinates of the NaN data points
    nan_points = np.column_stack(np.where(np.isnan(elevation)))

    # Create a KDTree for the valid points
    tree = cKDTree(valid_points)

    # Find the nearest neighbors for NaN points
    distances, _ = tree.query(nan_points)

    # Filter NaN points within the threshold distance
    close_nan_points = nan_points[distances < threshold_distance]


    # Perform KNN interpolation only for close NaN points
    if len(close_nan_points) > 0:
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)

        # Train the model with all available data
        knn.fit(valid_points, valid_values)

        # Predict values at the close_nan_points
        interpolated_values = knn.predict(close_nan_points)
        
        # Assign interpolated values to the elevation raster
        elevation.values[tuple(close_nan_points.T)] = interpolated_values

        print(f'Interpolated {len(close_nan_points)} points')

    # Convert the numpy array back to an xarray DataArray
    elevation_interp = xr.DataArray(elevation, dims=elevation.dims, coords=elevation.coords)
    return elevation_interp

def create_directory_if_not_exists(directory_path):
    """
    Create directory and all parent directories if they don't exist.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory to create
    
    Notes
    -----
    Uses os.makedirs with exist_ok=True to avoid errors if directory exists
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path,exist_ok=True)
        print(f"Created directory: {directory_path}")
  
        # print(f"Directory already exists: {directory_path}")
        
        
        
        
def get_subduction_teeth(row, size=1):
    """
    Create a triangular polygon representing a subduction tooth symbol.
    
    Generates a triangle pointing in the direction of subduction (perpendicular
    to the trench) based on the subduction normal angle. Used for visualizing
    subduction polarity on maps.
    
    Parameters
    ----------
    row : pd.Series or dict
        Row containing subduction zone information with keys:
        - 'Trench Longitude': Longitude of trench point
        - 'Trench Latitude': Latitude of trench point
        - 'Subduction Normal Angle': Angle perpendicular to trench (degrees)
    size : float, default=1
        Size of the triangle in degrees (approximately 111 km per degree)
    
    Returns
    -------
    shapely.geometry.Polygon
        Triangular polygon representing subduction tooth
    
    Examples
    --------
    >>> import pandas as pd
    >>> from shapely.geometry import Polygon
    >>> 
    >>> # Single subduction point
    >>> sz_data = pd.DataFrame({
    ...     'Trench Longitude': [140.0],
    ...     'Trench Latitude': [35.0],
    ...     'Subduction Normal Angle': [45.0]
    ... })
    >>> 
    >>> # Create tooth geometry
    >>> tooth = get_subduction_teeth(sz_data.iloc[0], size=1)
    >>> print(tooth.area)
    
    Notes
    -----
    - Triangle base is perpendicular to subduction normal
    - Triangle points in direction of plate subduction
    - Commonly used with GeoDataFrame.apply() to create tooth column
    
    See Also
    --------
    PlateKinematicsParameters.get_subductiondf : For obtaining subduction data
    """
    angle_rad = np.deg2rad(row['Subduction Normal Angle'])
    half_size = size / 2
    center_base = (row['Trench Longitude'], row['Trench Latitude'])
    
    # Calculate the base vertices
    base_vertex1 = (center_base[0] - half_size * np.sin(angle_rad),
                    center_base[1] + half_size * np.cos(angle_rad))
    base_vertex2 = (center_base[0] + half_size * np.sin(angle_rad),
                    center_base[1] - half_size * np.cos(angle_rad))
    
    # Calculate the third vertex
    top_vertex = (center_base[0] + size * np.cos(angle_rad),
                  center_base[1] + size * np.sin(angle_rad))
    
    vertices = [base_vertex1, base_vertex2, top_vertex]
    return Polygon(vertices)
    
    

    
def find_filename_with_number(folder, target_number):
    files = glob.glob(f"{folder}/*")
    pattern=re.compile(r"(\d+)")
    for file_name in files:
        matches=pattern.findall(file_name)
        number = int(matches[-1])  # Convert the matched number to an integer
        # print(int(number))
        if number == target_number:
            return file_name
    
    return None



def sjoin_nearest_geodesic_points(gdf1, gdf2, k=1,distance_col='dist_m'):
    """
    Perform geodesic nearest neighbor spatial join between point GeoDataFrames.
    
    Finds the k-nearest neighbors from gdf2 for each point in gdf1 using geodesic
    (great circle) distances on the sphere. This is crucial for global analyses where
    Euclidean distances are inaccurate.
    
    Parameters
    ----------
    gdf1 : gpd.GeoDataFrame
        Source points GeoDataFrame (EPSG:4326). Geometry is preserved from gdf1
    gdf2 : gpd.GeoDataFrame  
        Target points GeoDataFrame (EPSG:4326) to search for nearest neighbors
    k : int, default=1
        Number of nearest neighbors to find for each point in gdf1
    distance_col : str, default='dist_m'
        Name of column to store distances (in meters)
    
    Returns
    -------
    gpd.GeoDataFrame
        gdf1 merged with attributes from nearest point(s) in gdf2, including
        geodesic distance in meters. Overlapping columns from gdf2 are prefixed
        with 'nearest_'
    
    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> 
    >>> # Create sample point datasets
    >>> points1 = gpd.GeoDataFrame(
    ...     {'id': [1, 2, 3]},
    ...     geometry=[Point(120, 30), Point(130, 35), Point(140, 40)],
    ...     crs='EPSG:4326'
    ... )
    >>> 
    >>> trenches = gpd.GeoDataFrame(
    ...     {'trench_name': ['Japan', 'Mariana', 'Kurile']},
    ...     geometry=[Point(142, 38), Point(145, 15), Point(152, 48)],
    ...     crs='EPSG:4326'
    ... )
    >>> 
    >>> # Find nearest trench to each point
    >>> result = sjoin_nearest_geodesic_points(
    ...     points1, trenches, k=1, distance_col='trench_dist_m'
    ... )
    >>> print(result[['id', 'trench_name', 'trench_dist_m']])
    
    Notes
    -----
    - Uses sklearn's BallTree with haversine metric for efficiency
    - Automatically converts both GeoDataFrames to EPSG:4326
    - Distance is calculated along great circles (geodesic)
    - Much more accurate than Euclidean distance for global datasets
    - Prefixes overlapping column names with 'nearest_' to avoid conflicts
    - Earth radius used: 6,371,000 meters
    
    See Also
    --------
    gpd.sjoin_nearest : Euclidean nearest neighbor (inappropriate for global data)
    sklearn.neighbors.BallTree : Efficient spatial indexing
    """
    # Ensure both GeoDataFrames are in EPSG:4326
    gdf1 = gdf1.to_crs("EPSG:4326").copy()
    gdf2 = gdf2.to_crs("EPSG:4326").copy()

    # Convert to radians for BallTree
    gdf1_coords = np.deg2rad(np.column_stack([gdf1.geometry.y, gdf1.geometry.x]))
    gdf2_coords = np.deg2rad(np.column_stack([gdf2.geometry.y, gdf2.geometry.x]))

    # Build BallTree on gdf2 points
    tree = BallTree(gdf2_coords, metric="haversine")

    # Query nearest neighbors
    dist, ind = tree.query(gdf1_coords, k=k)

    # Convert distance from radians to meters
    dist_m = dist * 6371000  # Earth radius in meters

    # Build nearest point DataFrame
    nearest_df = gdf2.iloc[ind.flatten()].reset_index(drop=True)

    # Drop geometry from nearest_df to avoid conflict
    nearest_df = nearest_df.drop(columns="geometry")

    # Identify overlapping columns
    overlap_cols = [col for col in nearest_df.columns if col in gdf1.columns]
    if overlap_cols:
        nearest_df = nearest_df.rename(columns={col: f"nearest_{col}" for col in overlap_cols})

    # Attach results to gdf1
    result = gdf1.reset_index(drop=True).copy()
    result[distance_col] = dist_m.flatten()
    result = gpd.GeoDataFrame(
        pd.concat([result, nearest_df], axis=1), crs="EPSG:4326"
    )

    return result





def find_mantle_file(filenames, time, depth):
    # Ensure time and depth are within the specified range
    if not (0 <= time <= 1000) or not (0 <= depth <= 3000):
        return None

    # Create the pattern to match the file name
    pattern = re.compile(f".*_t{time}_{depth}(\D?.*).nc$")

    for filename in filenames:
        if pattern.search(filename):
            return filename
    
    return None


def calc_dist(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)



def flatten_list(lis,absolute=False):
    if absolute:
        flattened_list = [abs(item) for sublist in lis for item in sublist]
    else:
        flattened_list = [item for sublist in lis for item in sublist]
    
    return flattened_list


def generate_mesh(refinement_levels=8, *args, **kwargs):
    """
    Generate evenly distributed points on Earth's surface using icosahedral mesh.
    
    Creates a quasi-uniform global point distribution by subdividing an icosahedron.
    Essential for spatially unbiased sampling of geological data across the globe.
    
    Parameters
    ----------
    refinement_levels : int, default=8
        Mesh refinement level controlling point density:
        - Level 0: ~20° spacing (~2200 km)
        - Level 1: ~10° spacing (~1100 km)
        - Level 2: ~5° spacing (~550 km)
        - Level 3: ~2.5° spacing (~275 km)
        - Level 4: ~1.25° spacing (~140 km)
        - Level 5: ~0.6° spacing (~70 km)
        - Level 6: ~0.3° spacing (~35 km)
        - Level 7: ~0.15° spacing (~17 km)
        - Level 8: ~0.08° spacing (~9 km)
        Each level halves the spacing between points
    *args, **kwargs
        Additional arguments passed to stripy.icosahedral_mesh()
    
    Returns
    -------
    lons : np.ndarray
        Longitude values in degrees (-180 to 180)
    lats : np.ndarray
        Latitude values in degrees (-90 to 90)
    
    Examples
    --------
    >>> # Create medium-resolution global mesh (~17 km spacing)
    >>> lons, lats = generate_mesh(refinement_levels=7)
    >>> print(f"Generated {len(lons)} points")
    Generated 163842 points
    >>> 
    >>> # Use with plate reconstruction
    >>> import pygplates
    >>> points = pygplates.MultiPointOnSphere(zip(lats, lons))
    
    Notes
    -----
    - Higher refinement levels exponentially increase point count and computation time
    - Point count ≈ 10 × 4^refinement_levels + 2
    - Level 8 generates ~655,000 points (may be slow for large analyses)
    - Level 5-7 recommended for most geological applications
    - Points are quasi-uniform but not perfectly regular
    - Inherits from stripy's icosahedral_mesh implementation
    
    Warnings
    --------
    Refinement levels > 9 may cause memory issues and very long computation times.
    
    See Also
    --------
    stripy.spherical_meshes.icosahedral_mesh : Underlying mesh generator
    PlateKinematicsParameters.get_mean_subduction : Uses mesh for sampling
    """
    
    degrees = bool(kwargs.pop("degrees", True))

    mesh = icosahedral_mesh(refinement_levels, *args, **kwargs)
    lons = np.array(mesh.lons)
    lats = np.array(mesh.lats)
    del mesh
    if degrees:
        lons = np.rad2deg(lons)
        lats = np.rad2deg(lats)
    return lons, lats


def multipoints_from_polygon(polygon,resolution=0.1):
    
    'Input a single shape file to return discrete lat and lon point '


    # determine maximum edges
    # polygon = gpd_file.geometry
    latmin, lonmin, latmax, lonmax = polygon.bounds

    # create prepared polygon
    prep_polygon = prep(polygon)

    # construct a rectangular mesh
    points = []
    valid_points=[]
    for lat in np.arange(latmin, latmax, resolution):
        for lon in np.arange(lonmin, lonmax, resolution):
            points.append(Point((round(lat,4), round(lon,4))))

    # validate if each point falls inside shape using
    # the prepared polygon
    # valid_points.extend(filter(prep_polygon.contains, points))
    valid_points.extend(filter(prep_polygon.covers, points))
    lat=[]
    lon=[]
    for valid_point in valid_points:
        lat.append(valid_point.y)
        lon.append(valid_point.x)
    Multipoints=pygplates.MultiPointOnSphere(zip(lat,lon))
    
    return Multipoints,lat,lon

def poly_around_sub(i, subduction_df, n_steps=20,resolution=0.1):
    results = {
        'point_lats':[],
        'point_lons':[],
        'dist':[],
        # 'trench_lats':[],
#         'trench_lons':[]
    }
   
    y1 = subduction_df.iloc[i]['Trench Latitude']
    y2 = subduction_df.iloc[i + 1]['Trench Latitude']
    x1 = subduction_df.iloc[i]['Trench Longitude']
    x2 = subduction_df.iloc[i + 1]['Trench Longitude']

    dist = calc_dist(x1, y1, x2, y2)
    results['dist'].append(dist)

    if dist <= 2.0:
        try:
            
           
            dlon1 = n_steps * np.sin(np.radians(subduction_df.iloc[i]['Subduction Normal Angle']))
            dlat1 = n_steps * np.cos(np.radians(subduction_df.iloc[i]['Subduction Normal Angle']))
        
            ilon1 = subduction_df.iloc[i]['Trench Longitude'] + dlon1
            ilat1 = subduction_df.iloc[i]['Trench Latitude'] + dlat1
        
            dlon2 = n_steps * np.sin(np.radians(subduction_df.iloc[i + 1]['Subduction Normal Angle']))
            dlat2 = n_steps * np.cos(np.radians(subduction_df.iloc[i + 1]['Subduction Normal Angle']))
        
            ilon2 = subduction_df.iloc[i + 1]['Trench Longitude'] + dlon2
            ilat2 = subduction_df.iloc[i + 1]['Trench Latitude'] + dlat2
    
            y1 = subduction_df.iloc[i]['Trench Latitude']
            y2 = subduction_df.iloc[i + 1]['Trench Latitude']
            x1 = subduction_df.iloc[i]['Trench Longitude']
            x2 = subduction_df.iloc[i + 1]['Trench Longitude']
            
            coords = ((x1, y1), (x2, y2), (ilon2, ilat2), (ilon1, ilat1), (x1, y1))
            polygon = Polygon(coords)
            _, lats, lons = multipoints_from_polygon(polygon, resolution=(resolution-0.1*resolution))
            results['point_lats']=lats
            results['point_lons']=lons
            # results['trench_lats']=(y1+y2)/2
 #            results['trench_lons']=(x1+x2)/2
            
            
        except:
            pass        
        
        return results
       
            

    
def pointinpoly(points_gdf, polygons_gdf):
    '''
    Return the filtered polygons dataframe with polygons that overlap with the points dataframe
    '''
    # warnings.filterwarnings('ignore')
    
    # List to store the filtered polygons
    filtered_polygons = []

    # Iterate over each polygon and check for overlaps with points
    for index, polygon in polygons_gdf.iterrows():
        # Check if any points intersect the current polygon
        overlapping_points = points_gdf[points_gdf.intersects(polygon.geometry)]

        # If there are overlapping points, add the polygon to the list
        if not overlapping_points.empty:
            filtered_polygons.append(polygon)

    # Convert the list of filtered polygons to a GeoDataFrame
    filtered_polygons_gdf = gpd.GeoDataFrame(filtered_polygons, columns=polygons_gdf.columns)

    return filtered_polygons_gdf
    

def df_to_NetCDF(x,y,z, statistic='mean',  
                 grid_resolution=0.1, 
                 clip=(None,None),
                 lon_bin_edges=None,
                 lat_bin_edges=None):
    """
    Convert scattered point data to a regular gridded NetCDF DataArray.
    
    Bins irregular point data onto a regular lat/lon grid using various statistical
    aggregations. Essential for creating continuous surfaces from discrete samples.
    
    Parameters
    ----------
    x : array-like
        Longitude values of data points
    y : array-like
        Latitude values of data points
    z : array-like
        Values to grid (elevation, temperature, etc.)
    statistic : str, default='mean'
        Statistic to compute in each bin:
        - 'mean': Average of points in bin
        - 'median': Median of points
        - 'std': Standard deviation
        - 'count': Number of points
        - 'sum', 'min', 'max': Other aggregations
    grid_resolution : float, default=0.1
        Grid spacing in degrees (ignored if bin_edges provided)
    clip : tuple of (float, float), default=(None, None)
        (min_value, max_value) to clip output. None means no clipping
    lon_bin_edges : array-like, optional
        Custom longitude bin edges. If None, auto-generated from data extent
    lat_bin_edges : array-like, optional
        Custom latitude bin edges. If None, auto-generated from data extent
    
    Returns
    -------
    xarray.DataArray
        Gridded data with dimensions ['Latitude', 'Longitude'] and coordinates
        at bin midpoints. NaN where no data exists
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> 
    >>> # Scattered elevation data
    >>> df = pd.DataFrame({
    ...     'lon': np.random.uniform(-180, 180, 10000),
    ...     'lat': np.random.uniform(-90, 90, 10000),
    ...     'elevation': np.random.normal(0, 1000, 10000)
    ... })
    >>> 
    >>> # Create 0.5° grid
    >>> grid = df_to_NetCDF(
    ...     x=df['lon'],
    ...     y=df['lat'],
    ...     z=df['elevation'],
    ...     statistic='mean',
    ...     grid_resolution=0.5,
    ...     clip=(-2000, 5000)
    ... )
    >>> 
    >>> # Save to NetCDF
    >>> grid.to_netcdf('gridded_elevation.nc')
    
    Notes
    -----
    - Uses scipy.stats.binned_statistic_2d for efficient gridding
    - Bins with no data are set to NaN
    - Coordinates represent bin midpoints
    - Clipping is applied after gridding
    - For large datasets, increase grid_resolution for faster processing
    
    See Also
    --------
    scipy.stats.binned_statistic_2d : Underlying gridding function
    xarray.DataArray : Output data structure
    """

    if lon_bin_edges is None:
    # Define bin edges (lat and lon) based on your data range and desired bin sizes
        lon_bin_edges = np.arange(x.min(), x.max() + grid_resolution, grid_resolution)
    if lat_bin_edges is None:
        lat_bin_edges = np.arange(y.min(), y.max()+ grid_resolution, grid_resolution)

    # Calculate binned statistics (mean, median, etc.)
    arr, _, _, _ = binned_statistic_2d(
        x,
        y,
        values=z,
        statistic=statistic,
        bins=[lon_bin_edges, lat_bin_edges],
    )
    
    arr = arr.T
    if clip[0] !=None:
        arr[arr<clip[0]]=np.nan
    if clip[1] !=None:
        arr[arr>clip[1]]=np.nan
    
    # Replace NaN values with the chosen nan_replacement value
    # arr[np.isnan(arr)] = nan_replacement
    
    # Calculate midpoint of latitude bins
    lat_midpoints = lat_bin_edges[:-1] + grid_resolution / 2
    lon_midpoints = lon_bin_edges[:-1] + grid_resolution / 2
    
    da = xr.DataArray(
        data=arr,
        coords={'Latitude': lat_midpoints, 'Longitude': lon_midpoints},
        dims=['Latitude', 'Longitude']
    )
    return da

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great circle distance between two points using Haversine formula.
    
    Computes the shortest distance between two points on Earth's surface,
    accounting for Earth's curvature. More accurate than Euclidean distance
    for geographical coordinates.
    
    Parameters
    ----------
    lat1 : float or array-like
        Latitude of first point(s) in degrees
    lon1 : float or array-like
        Longitude of first point(s) in degrees
    lat2 : float or array-like
        Latitude of second point(s) in degrees
    lon2 : float or array-like
        Longitude of second point(s) in degrees
    
    Returns
    -------
    float or np.ndarray
        Great circle distance in kilometers
    
    Examples
    --------
    >>> # Distance between New York and London
    >>> dist = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
    >>> print(f"Distance: {dist:.1f} km")
    Distance: 5570.2 km
    >>> 
    >>> # Vectorized calculation
    >>> lats1 = np.array([35, 40, 45])
    >>> lons1 = np.array([135, 140, 145])
    >>> lats2 = np.array([36, 41, 46])
    >>> lons2 = np.array([136, 141, 146])
    >>> distances = haversine_distance(lats1, lons1, lats2, lons2)
    
    Notes
    -----
    - Assumes spherical Earth with radius 6,371 km
    - Accurate for most geological applications
    - For higher precision, consider geopy.distance.geodesic (uses WGS84 ellipsoid)
    - Vectorized for efficient array operations
    
    References
    ----------
    https://en.wikipedia.org/wiki/Haversine_formula
    
    See Also
    --------
    geopy.distance.geodesic : More accurate ellipsoidal distance
    sjoin_nearest_geodesic_points : Uses similar distance calculation
    """
    R = 6371  # Radius of the Earth in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c

    return distance

def minimum_distance(gdf,lat_ref,lon_ref):
        '''
        Find the closest lat and lon point along with minimum distance in dataframe for a reference lat and lon
    
    
        ** Useful in calculating closest points and distance to trench 

        '''
    
        geometry=gdf.geometry
        min_dist=haversine_distance(geometry.iloc[0].y, geometry.iloc[0].x,lat_ref,lon_ref)
        lon=None
        lat=None
        for geo in geometry:
            dist=haversine_distance(geo.y, geo.x,lat_ref,lon_ref)
            if dist<min_dist:
                min_dist=dist
                lon=geo.x
                lat=geo.y
            
            
        return min_dist,lat,lon   

# def calculate_wma(gdfs, exclude_cols=None):
#     print("Using weighted mean")
#     # Number of GeoDataFrames
#     n = len(gdfs)
    
#     if n == 0:
#         raise ValueError("The list of GeoDataFrames is empty.")
    
#     if exclude_cols is None:
#         exclude_cols = []

#     # Calculate weights in descending order
#     weights = np.arange(n, 0, -1)  # Weights from n to 1
#     weight_sum = weights.sum()

#     # Initialize with the first GeoDataFrame
#     wma_df = gdfs[0].copy()

#     # Apply WMA only on numeric columns not in exclude_cols
#     for column in wma_df.columns:
#         if column not in exclude_cols and wma_df[column].dtype.kind in 'bifc':
#             weighted_sum = np.zeros(len(wma_df))
#             for i, gdf in enumerate(gdfs):
#                 weighted_sum += gdf[column] * weights[i]
#             wma_df[column] = weighted_sum / weight_sum

#     return wma_df

def calculate_wma(gdfs, exclude_cols=None):
    """
    Calculate weighted moving average across multiple GeoDataFrames.
    
    Computes time-weighted average where recent time steps have higher weights.
    Useful for time-averaged subduction parameters or other temporal analyses.
    Handles NaN values by adjusting weights accordingly.
    
    Parameters
    ----------
    gdfs : list of gpd.GeoDataFrame
        List of GeoDataFrames to average, ordered from oldest to newest.
        All must have same structure and index
    exclude_cols : list of str, optional
        Column names to exclude from averaging (e.g., 'geometry', 'Latitude').
        These columns are taken from the first GeoDataFrame
    
    Returns
    -------
    gpd.GeoDataFrame
        Weighted average GeoDataFrame with same structure as inputs
    
    Examples
    --------
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> 
    >>> # Create sample time series of subduction data
    >>> gdfs = []
    >>> for t in range(5):
    ...     gdf = gpd.GeoDataFrame({
    ...         'convergence_rate': np.random.rand(100) * 10,
    ...         'time': t
    ...     }, geometry=gpd.points_from_xy(
    ...         np.random.rand(100)*360-180,
    ...         np.random.rand(100)*180-90
    ...     ))
    ...     gdfs.append(gdf)
    >>> 
    >>> # Calculate weighted average (recent times weighted higher)
    >>> wma_gdf = calculate_wma(gdfs, exclude_cols=['time', 'geometry'])
    
    Notes
    -----
    - Weights decrease linearly from n to 1 (n = number of GeoDataFrames)
    - Most recent GeoDataFrame (last in list) gets highest weight
    - Only numeric columns are averaged (excluding those in exclude_cols)
    - NaN values are handled intelligently: only valid values contribute to weights
    - Geometry and specified columns are preserved from first GeoDataFrame
    
    Warnings
    --------
    All GeoDataFrames must have matching indices and column structure
    
    See Also
    --------
    mean_gdfs : Simple unweighted average of GeoDataFrames
    PlateKinematicsParameters.get_mean_subduction : Uses this for time averaging
    """
    print("Using weighted mean")
    n = len(gdfs)
    if n == 0:
        raise ValueError("The list of GeoDataFrames is empty.")

    if exclude_cols is None:
        exclude_cols = []

    weights = np.arange(n, 0, -1)  # Weights from n to 1

    wma_df = gdfs[0].copy()

    for column in wma_df.columns:
        if column not in exclude_cols and wma_df[column].dtype.kind in 'bifc':
            weighted_sum = np.zeros(len(wma_df))
            weight_tracker = np.zeros(len(wma_df))  # Track total valid weights

            for i, gdf in enumerate(gdfs):
                col_values = gdf[column].to_numpy()
                mask = ~np.isnan(col_values)  # valid values
                weighted_sum[mask] += col_values[mask] * weights[i]
                weight_tracker[mask] += weights[i]

            # Avoid divide-by-zero where all are NaN
            with np.errstate(invalid='ignore'):
                wma_df[column] = weighted_sum / weight_tracker

    return wma_df



# def mean_gdfs(gdfs, exclude_cols=None):

#     print("Using mean")
#     if len(gdfs) == 0:
#         raise ValueError("The list of GeoDataFrames is empty.")
    
#     if exclude_cols is None:
#         exclude_cols = []
    
#     # Start with a copy of the first GeoDataFrame
#     mean_df = gdfs[0].copy()

#     # Iterate over columns
#     for column in mean_df.columns:
#         if column not in exclude_cols and mean_df[column].dtype.kind in 'bifc':
#             stacked = np.vstack([gdf[column].values for gdf in gdfs])
#             mean_df[column] = stacked.mean(axis=0)
#         # else: keep the column from the first gdf unchanged
    
#     return mean_df

def mean_gdfs(gdfs, exclude_cols=None):
    """
    Calculate element-wise mean across multiple GeoDataFrames.
    
    Computes simple average of corresponding values across GeoDataFrames,
    handling NaN values appropriately.
    
    Parameters
    ----------
    gdfs : list of gpd.GeoDataFrame
        List of GeoDataFrames to average. All must have same structure
    exclude_cols : list of str, optional
        Column names to exclude from averaging (e.g., 'geometry', 'Latitude')
    
    Returns
    -------
    gpd.GeoDataFrame
        Mean GeoDataFrame with same structure as inputs
    
    Notes
    -----
    - Uses np.nanmean to ignore NaN values when computing average
    - Non-numeric columns and excluded columns preserved from first GeoDataFrame
    - All GeoDataFrames must have matching structure
    
    See Also
    --------
    calculate_wma : Weighted average alternative
    """

    print("Using mean")
    if len(gdfs) == 0:
        raise ValueError("The list of GeoDataFrames is empty.")
    
    if exclude_cols is None:
        exclude_cols = []
    
    # Start with a copy of the first GeoDataFrame
    mean_df = gdfs[0].copy()

    # Iterate over columns
    for column in mean_df.columns:
        if column not in exclude_cols and mean_df[column].dtype.kind in 'bifc':
            stacked = np.vstack([gdf[column].values for gdf in gdfs])
            # Ignore NaNs when computing the mean
            mean_df[column] = np.nanmean(stacked, axis=0)
        # else: keep the column from the first gdf unchanged
    
    return mean_df

def interpolate_and_save_as_geotiff(folder, param_type, start_time, end_time, depths, required_time_step=1):
    """
    Temporally interpolate raster data between two time steps and save as GeoTIFFs.
    
    Creates intermediate time steps between existing rasters using linear interpolation.
    Essential for creating continuous time series from sparse temporal sampling.
    
    Parameters
    ----------
    folder : str
        Base folder containing time-organized rasters:
        {folder}/{time}/{param_type}_{depth}.tif
    param_type : str
        Parameter type (e.g., 'temperature', 'vz', 'viscosity')
    start_time : int or float
        Starting time (Ma) with existing raster
    end_time : int or float
        Ending time (Ma) with existing raster
    depths : list of int or float
        Depth levels (km) to interpolate
    required_time_step : int or float, default=1
        Interval (Myr) for creating interpolated time steps
    
    Examples
    --------
    >>> # Interpolate between 0 Ma and 10 Ma at 1 Myr intervals
    >>> interpolate_and_save_as_geotiff(
    ...     folder='mantle_data/temperature',
    ...     param_type='temperature',
    ...     start_time=0,
    ...     end_time=10,
    ...     depths=[100, 300, 600],
    ...     required_time_step=1
    ... )
    
    Notes
    -----
    - Uses linear interpolation: value(t) = v1 + (v2-v1) * (t-t1)/(t2-t1)
    - Creates directories automatically for new time steps
    - Skips if source rasters don't exist
    - Output rasters use LZW compression
    - Preserves georeferencing from source rasters
    
    See Also
    --------
    MantleParameters.interpolate_mantle_data : Higher-level interpolation interface
    """
    
    initial_time_step=int(end_time-start_time)
    # Loop through each depth
    for d in depths:
        
        try:
            with rasterio.open(os.path.join(folder, str(start_time), f"{param_type}_{d}.tif")) as src_start, \
                rasterio.open(os.path.join(folder, str(end_time), f"{param_type}_{d}.tif")) as src_end:

                # Define the interpolation time steps you want (1, 2, 3, ..., 9).
                time_steps = range(1, initial_time_step,required_time_step)

                    # Loop through each time step and perform interpolation
                for time in time_steps:
                        # Calculate the interpolated data using your formula
                    time_fraction = time / float(initial_time_step)
                    interpolated_data = src_start.read(1) + (src_end.read(1) - src_start.read(1)) * time_fraction

                        # Prepare GeoTIFF parameters
                    create_directory_if_not_exists(os.path.join(folder, f"{start_time + time}"))
                    output_tiff = os.path.join(folder, f"{start_time + time}", f"{param_type}_{d}.tif")
                    height, width = src_start.shape
                    transform = src_start.transform
                    dtype = interpolated_data.dtype

                    # Write data to GeoTIFF
                    with rasterio.open(output_tiff, 'w', driver='GTiff', height=height, width=width, count=1, 
                                       dtype=dtype, transform=transform,compress='lzw') as dst:
                        dst.write(interpolated_data, 1)
        
            # Open the source GeoTIFF files using rasterio
           

        except Exception as e:
            print(f"No raster file for interpolating {d} from {start_time} Ma to {end_time} Ma")
            continue
    print(f"Interpolation completed from {start_time} Ma to {end_time} Ma completed.")
            
        
       
def open_dataset_with_fallback(file_path):
    engines = ['netcdf4', 'h5netcdf', 'scipy']
    for engine in engines:
        try:
            ds = xr.open_dataset(file_path, engine=engine)
            return ds
        except ValueError:
            continue
    raise ValueError(f"Could not read {file_path} with any known engine.")    
    

def process_mantle_data(new_folder, time, depth, param_type, data_folder):
    create_directory_if_not_exists(f"{new_folder}/{time}")

    folder = f"{data_folder}/{time}/*"
    filenames = glob.glob(folder)
    file_path = find_mantle_file(filenames, time, depth)
    
    if file_path is None:
        print(f"No file found for time: {time}, depth: {depth}. Skipping...")
        return
    
    # print(file_path)
    
    try:
        ds = open_dataset_with_fallback(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return

    try:
        lon_data = ds['lon']

        # Convert longitudes from 0-360 to -180-180
        lon_data = np.where(lon_data > 180, lon_data - 360, lon_data)
        ds = ds.assign_coords(lon=lon_data)
        ds = ds.sortby('lon')
        lat_data = ds['lat'][::-1]

        # Prepare GeoTIFF parameters
        output_tiff = os.path.join(f"{new_folder}", str(time), f"{param_type}_{depth}.tif")
        height, width = ds['lon'].shape[0], ds['lat'].shape[0]

        lon_step = (lon_data.max() - lon_data.min()) / (width - 1)
        lat_step = (lat_data.max() - lat_data.min()) / (height - 1)
        transform = from_origin(float(ds['lon'].min()), float(ds['lat'].min()), lon_step, -lat_step)
        dtype = ds['z'].dtype
    except KeyError:
        try:
            # Access the longitude data
            lon_data = ds['x']

            # Convert longitudes from 0-360 to -180-180
            lon_data = np.where(lon_data > 180, lon_data - 360, lon_data)
            ds = ds.assign_coords(x=lon_data)
            ds = ds.sortby('x')

            lat_data = ds['y'][::-1]

            # Prepare GeoTIFF parameters
            output_tiff = os.path.join(f"{new_folder}", str(time), f"{param_type}_{depth}.tif")
            height, width = ds['x'].shape[0], ds['y'].shape[0]

            lon_step = (lon_data.max() - lon_data.min()) / (width - 1)
            lat_step = (lat_data.max() - lat_data.min()) / (height - 1)
            transform = from_origin(float(ds['x'].min()), float(ds['y'].min()), lon_step, -lat_step)
            dtype = ds['z'].dtype
        except KeyError as e:
            print(f"Data has no column lon/lat or x/y: {e}")
            return
        except Exception as e:
            print(f"Unexpected error processing file {file_path}: {e}")
            return
    except Exception as e:
        print(f"Unexpected error processing file {file_path}: {e}")
        return

    # Write data to GeoTIFF with LZW compression
    with rasterio.open(output_tiff, 'w', driver='GTiff', height=height, width=width, count=1, dtype=dtype, transform=transform, compress='lzw') as dst:
        dst.write(ds['z'].values, 1)

def delete_empty_folders(folder_path):
    """
    Recursively delete empty folders.

    Args:
        folder_path (str): The path of the folder to check.
    """
    # Iterate over all the directories and subdirectories
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # Check if the directory is empty
        if not os.listdir(root):
            # If empty, delete the directory
            print(f"Deleting empty folder: {root}")
            os.rmdir(root)
            
        
                    

    
def create_geodataframe_topologies(topologies, reconstruction_time):
    """ This is a function to convert topologies from pygplates into a GeoDataFrame
    This helps select the closed topological plates ('gpml:TopologicalClosedPlateBoundary',
    and also helps resolve plotting artefacts from crossing the dateline. 
    This function does NOT incorporate various plate boundary types into the geodataframe!
    
    Input: 
        - pygplates.Feature. This is designed for `topologies`, which comes from:
              resolved_topologies = ptt.resolve_topologies.resolve_topologies_into_features(
                                        rotation_model, topology_features, reconstruction_time)
              topologies, ridge_transforms, ridges, transforms, trenches, trench_left, trench_right, other = resolved_topologies
        - recontruction time - this is just for safekeeping in the geodataframe!
    Output: 
        - gpd.GeoDataFrame of the feature"""
    
    # function for getting closed topologies only
    # i.e., the plates themselves, NOT all the features for plotting!
    
    # # set up the empty geodataframe
    # recon_gpd = gpd.GeoDataFrame()
    # recon_gpd['NAME'] = None
    # recon_gpd['PLATEID1'] = None
    # recon_gpd['PLATEID2'] = None
    # recon_gpd['FROMAGE'] = None
    # recon_gpd['TOAGE'] = None
    # # recon_gpd['geometry'] = None
    # recon_gpd['reconstruction_time'] = None
    # recon_gpd['gpml_type'] = None
    

    # some empty things to write stuff to
    names                = []
    plateid1s            = []
    plateid2s            = []
    fromages             = []
    toages               = []
    geometrys            = []
    reconstruction_times = []
    gpml_types           = []
    
    # a dateline wrapper! so that they plot nicely and do nice things in geopandas
    date_line_wrapper = pygplates.DateLineWrapper()
    
    for i, seg in enumerate(topologies):
        gpmltype = seg.get_feature_type()
        
        # polygon and wrap
        polygon = seg.get_geometry()
        wrapped_polygons = date_line_wrapper.wrap(polygon)
        for poly in wrapped_polygons:
            ring = np.array([(p.get_longitude(), p.get_latitude()) for p in poly.get_exterior_points()])
            ring[:,1] = np.clip(ring[:,1], -89, 89) # anything approaching the poles creates artefacts
            for wrapped_point in poly.get_exterior_points():
                wrapped_point_lat_lon = wrapped_point.get_latitude(), wrapped_point.get_longitude()
            
            # might result in two polys - append to loop here (otherwise we will be missing half the pacific etc)
            name = seg.get_name()
            plateid = seg.get_reconstruction_plate_id()
            conjid = seg.get_conjugate_plate_id()
            from_age, to_age = seg.get_valid_time()
            
            names.append(name)
            plateid1s.append(plateid)
            plateid2s.append(conjid)
            fromages.append(from_age)
            toages.append(to_age)
            geometrys.append(shapely.geometry.Polygon(ring)) 
            reconstruction_times.append(reconstruction_time)
            gpml_types.append(str(gpmltype))
    
    # write to geodataframe
    recon_gpd=gpd.GeoDataFrame(geometry=geometrys)
    recon_gpd['NAME'] = names
    recon_gpd['PLATEID1'] = plateid1s
    recon_gpd['PLATEID2'] = plateid2s
    recon_gpd['FROMAGE'] = fromages
    recon_gpd['TOAGE'] = toages
    
    recon_gpd['reconstruction_time'] = reconstruction_times
    recon_gpd['gpml_type'] = gpml_types
    # recon_gpd=recon_gpd.set_geometry(geometrys)
    recon_gpd = recon_gpd.set_crs(epsg=4326)
    
    return recon_gpd


# Function to determine majority PlateID1
def get_majority_plate_id(points, topologies_gdf):
    plate_ids = []
    for point in points:
        for _, row in topologies_gdf.iterrows():
            if row['geometry'].contains(point):
                plate_ids.append(row['PLATEID1'])
                break
    if plate_ids:
        return max(set(plate_ids), key=plate_ids.count)
    else:
        return None
       


def generate_points(lat, lon, angle, num_points=5, distance=20):
    """
    Generates points at a given distance and angle from a starting point on the Earth's surface.

    Parameters:
    - lat (float): Latitude of the starting point.
    - lon (float): Longitude of the starting point.
    - angle (float): Angle (bearing) in degrees from the north, in the clockwise direction.
    - num_points (int): Number of points to generate.
    - distance (float): Distance between each point in kilometers.

    Returns:
    - list of shapely.geometry.Point: List of Shapely Point objects with the new points' coordinates.
    """
    start_point = GeopyPoint(lat, lon)
    points = []

    for i in range(1, num_points + 1):
        # Calculate the distance for the current point
        incremental_distance = distance * i
        
        # Calculate the destination point using geodesic method
        destination = geodesic(kilometers=incremental_distance).destination(start_point, angle)
        
        # Append the new point (longitude, latitude) as a Shapely Point geometry to the list
        points.append(Point(destination.longitude, destination.latitude))

    return points


def nan_gaussian_filter(data, sigma,radius=5):
    """
    Apply Gaussian smoothing to data with NaN values.
    
    Performs weighted Gaussian filtering that properly handles missing data
    by adjusting filter weights based on valid data availability.
    
    Parameters
    ----------
    data : np.ndarray
        2D array to smooth, may contain NaN values
    sigma : float
        Standard deviation of Gaussian kernel (controls smoothing strength)
    radius : float, default=5
        Truncation radius in standard deviations
    
    Returns
    -------
    np.ndarray
        Smoothed array with same shape as input
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create noisy data with gaps
    >>> data = np.random.rand(100, 100) + np.random.normal(0, 0.1, (100, 100))
    >>> data[30:40, 30:40] = np.nan
    >>> 
    >>> # Smooth with 2-pixel sigma
    >>> smoothed = nan_gaussian_filter(data, sigma=2, radius=5)
    
    Notes
    -----
    - Fills NaN with zero before filtering
    - Tracks valid data locations with binary weights
    - Normalizes by smoothed weights to handle edge effects
    - Sigma controls smoothing: larger = smoother
    - Radius controls filter extent: larger = slower but more accurate
    
    See Also
    --------
    scipy.ndimage.gaussian_filter : Underlying filter implementation
    """
    data_filled = np.nan_to_num(data, nan=0.0)
    weights = ~np.isnan(data)
    # smoothed_data = gaussian_filter(data_filled * weights, sigma=sigma,radius=radius)
    # weights_smooth = gaussian_filter(weights.astype(float), sigma=sigma,radius=radius)

    smoothed_data = gaussian_filter(data_filled * weights, sigma=sigma,truncate=radius)
    weights_smooth = gaussian_filter(weights.astype(float), sigma=sigma,truncate=radius)

    return smoothed_data / weights_smooth



def plotgdf(gdf,gplot,column=None,mollweide=False,time=0,cbar=False,quick=True,**kwargs):
    """
    Plot reconstructed GeoDataFrame with plate tectonic features.
    
    Creates publication-quality maps of geological data with plate boundaries,
    coastlines, and other tectonic features overlaid.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to plot with point geometry
    gplot : gplately.PlotTopologies
        GPlately plot object with plate model and features
    column : str, optional
        Column name to visualize with colors
    mollweide : bool, default=False
        Use Mollweide projection (equal-area) if True, else PlateCarree
    time : int or float, default=0
        Reconstruction time (Ma) for plotting plate features
    cbar : bool, default=False
        Whether to display colorbar
    quick : bool, default=True
        If True, grids data for faster plotting (may lose some detail).
        If False, plots individual points (slower but more accurate)
    **kwargs : dict
        Additional plotting parameters:
        - cmap : str or colormap, colormap name
        - vmin, vmax : float, color scale limits
        - label : str, colorbar label
        - title : str, plot title
        - features : bool, whether to plot plate features (default True)
        - color : str, point color if not using column
        - markersize : int, point size (default 10)
        - orientation : str, colorbar orientation ('vertical'/'horizontal')
        - shrink : float, colorbar shrink factor
        - extend : str, colorbar extension ('neither'/'both'/'min'/'max')
        - central_longitude : float, map center longitude (default 0)
        - figsize : tuple, figure size (default (12,8))
    
    Returns
    -------
    matplotlib.axes.Axes
        Axes object for further customization
    
    Examples
    --------
    >>> import geopandas as gpd
    >>> import gplately
    >>> 
    >>> # Load reconstructed data
    >>> gdf = gpd.read_file('reconstructed_50Ma.shp')
    >>> 
    >>> # Create plot object
    >>> model = gplately.PlateReconstruction(...)
    >>> gplot = gplately.PlotTopologies(model, time=50)
    >>> 
    >>> # Plot with features
    >>> ax = plotgdf(
    ...     gdf, gplot,
    ...     column='elevation',
    ...     mollweide=True,
    ...     time=50,
    ...     cbar=True,
    ...     quick=True,
    ...     cmap='terrain',
    ...     vmin=-5000,
    ...     vmax=5000,
    ...     label='Elevation (m)',
    ...     figsize=(14, 8)
    ... )
    
    Notes
    -----
    - quick=True grids data at 0.2° resolution for faster rendering
    - Mollweide projection recommended for global equal-area maps
    - PlateCarree better for regional maps or preserving angles
    - Automatically plots trenches, ridges, transforms when features=True
    - High DPI (300) suitable for publications
    
    See Also
    --------
    gplately.PlotTopologies : Plate feature plotting
    df_to_NetCDF : Used for quick gridding option
    """

    cmap = kwargs.get('cmap', None)
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)
    label = kwargs.get('label', None)
    title=kwargs.get('title', None)
    features=kwargs.get('features',True)
    color=kwargs.get('color',None)
    markersize=kwargs.get('markersize',10)
    orientation=kwargs.get('orientation','vertical')
    shrink=kwargs.get('shrink',0.5)
    extend=kwargs.get('extend',None)
    
    central_longitude=kwargs.get('central_longitude',0)
    figsize=kwargs.get('figsize',(12,8))
    

    
    fig = plt.figure(figsize=figsize, dpi=300)
    # gplot = gplately.PlotTopologies(model, coastlines=model.coastlines, continents=model.continents, time=time)

    if mollweide:
        ax = fig.add_subplot(111, projection=ccrs.Mollweide(central_longitude = central_longitude))
        ax.gridlines(color='0.7',linestyle='--', xlocs=np.arange(-180,180,30), ylocs=np.arange(-90,90,30))
    
        mollweide_proj = f"+proj=moll +lon_0={central_longitude} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        gdf=gdf.to_crs(mollweide_proj)
    else:
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude = central_longitude))
        ax.gridlines(color='0.7',linestyle='--', xlocs=np.arange(-180,180,15), ylocs=np.arange(-90,90,15))
    
        
        
    if features:
    
        # Plot shapefile features, subduction zones and MOR boundaries at time Ma
        gplot.time = time # Ma
        gplot.plot_continents(ax, facecolor='grey', alpha=0.2)
        gplot.plot_coastlines(ax, color='skyblue',alpha=0.3)
        gplot.plot_ridges_and_transforms(ax, color='k')
        gplot.plot_trenches(ax, color='k')
        gplot.plot_subduction_teeth(ax, color='k')

        gplot.plot_ridges(ax, color='k')
        gplot.plot_transforms(ax, color='k')
        gplot.plot_misc_boundaries(ax, color='k')

    
        # Plot the GeoDataFrame
    
    if quick:
        da=df_to_NetCDF(x=gdf["Longitude"],y=gdf["Latitude"],z=gdf[column],grid_resolution=0.2)
        plot=gplot.plot_grid(ax=ax, grid=da,**{'cmap':cmap,'vmax':vmax,'vmin': vmin})
    else:
        plot = gdf.plot(ax=ax, cmap=cmap, column=column,vmax=vmax,vmin=vmin,color=color,markersize=markersize)
    

                                             # 'label':f'{column}'})
    if cbar:
        # Create a ScalarMappable object
        sm = cm.ScalarMappable(cmap=cmap)
        sm.set_array(gdf[column])
        sm.set_clim(vmin, vmax)
        
        # Add colorbar using the same Axes object used for plotting
        colorbar = plt.colorbar(sm, ax=ax, orientation=orientation,shrink=shrink,extend=extend, label=label)
        colorbar.set_label(label)
    
    ax.set_global()
    
    return ax    
    
def poly_around_sub_ver2(i, subduction_df,topologies_gdf, n_steps=14,resolution=0.1):


    '''
    This function creates a polygon around the subduction zone and fill the polygon with Point at specified resolution

    i: index of the subducting point (int)
    subduction_df: Dataframe containing all the subducting points
    n_steps: length of profile in deg (1 deg= 111 Km)
    resolution: the resolution of points within the polygon


    '''

    # getting trench point and adjacents point
    y1 = subduction_df.iloc[i]['Trench Latitude']
    y2 = subduction_df.iloc[i + 1]['Trench Latitude']
    x1 = subduction_df.iloc[i]['Trench Longitude']
    x2 = subduction_df.iloc[i + 1]['Trench Longitude']

    # dist = haversine_distance(y1, x1, y2, x2)
    dist = calc_dist(x1, y1, x2, y2)


    if dist <= 2: ## checking if the point does not have significant gaps.  
        try:

                
            dlon1 = n_steps * np.sin(np.radians(subduction_df.iloc[i]['Subduction Normal Angle'])) 
            dlat1 = n_steps * np.cos(np.radians(subduction_df.iloc[i]['Subduction Normal Angle']))

            ilon1 = subduction_df.iloc[i]['Trench Longitude'] + dlon1 ## creating end point of first profile
            ilat1 = subduction_df.iloc[i]['Trench Latitude'] + dlat1

            dlon2 = n_steps * np.sin(np.radians(subduction_df.iloc[i + 1]['Subduction Normal Angle']))
            dlat2 = n_steps * np.cos(np.radians(subduction_df.iloc[i + 1]['Subduction Normal Angle']))

            ilon2 = subduction_df.iloc[i + 1]['Trench Longitude'] + dlon2
            ilat2 = subduction_df.iloc[i + 1]['Trench Latitude'] + dlat2 ## creating end point of 2nd profile

            y1 = subduction_df.iloc[i]['Trench Latitude']
            y2 = subduction_df.iloc[i + 1]['Trench Latitude']
            x1 = subduction_df.iloc[i]['Trench Longitude']
            x2 = subduction_df.iloc[i + 1]['Trench Longitude']

            coords = ((x1, y1), (x2, y2), (ilon2, ilat2), (ilon1, ilat1), (x1, y1)) ## creating a quadrilateral 
            polygon = Polygon(coords)
            _, lats, lons = multipoints_from_polygon(polygon, resolution=(resolution-0.1*resolution)) ## creating point within the polygons
            
            
            points_gdf=gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons,lats)) ## points in geodataframe
            points_gdf=points_gdf.set_crs("epsg:4326")
            
            
            ## getting polygons plate corresponding overriding plate. we will remove point that doesnot lies on the overrriding plate
            topologies_gdfc = topologies_gdf[(topologies_gdf['PLATEID1'] == subduction_df.iloc[i]["Overriding Plate ID"]) | 
                                            (topologies_gdf['PLATEID1'] == subduction_df.iloc[i+1]["Overriding Plate ID"])].copy() 

            

          
            points_within_oid = gpd.sjoin(points_gdf, topologies_gdfc[['geometry', 'PLATEID1']], how='left', predicate='within')

            ## getting overriding plate id for all the points that are generated 
            
            return points_within_oid
     

        except Exception as e:
            print(e)
            pass


    return None
       


def multipoints_from_shape(gpd_file,resolution=0.1):
    
    'Input a single shape file to return discrete lat and lon point '


    # determine maximum edges
    polygon = gpd_file.geometry
    latmin, lonmin, latmax, lonmax = polygon.bounds

    # create prepared polygon
    prep_polygon = prep(polygon)

    # construct a rectangular mesh
    points = []
    valid_points=[]
    for lat in np.arange(latmin, latmax, resolution):
        for lon in np.arange(lonmin, lonmax, resolution):
            points.append(Point((round(lat,4), round(lon,4))))

    # validate if each point falls inside shape using
    # the prepared polygon
    valid_points.extend(filter(prep_polygon.contains, points))
    lat=[]
    lon=[]
    for valid_point in valid_points:
        lat.append(valid_point.y)
        lon.append(valid_point.x)
    Multipoints=pygplates.MultiPointOnSphere(zip(lat,lon))
    
    return Multipoints,lat,lon


def create_geodataframe_topologies(topologies, reconstruction_time):
    """
    Convert pygplates resolved topologies to GeoDataFrame.
    
    Transforms pygplates topological features into a GeoDataFrame for easier
    manipulation and spatial operations. Handles dateline wrapping to avoid
    plotting artifacts.
    
    Parameters
    ----------
    topologies : list of pygplates.ResolvedTopologicalBoundary
        Resolved topological features from pygplates.resolve_topologies()
    reconstruction_time : int or float
        Reconstruction time (Ma) for metadata
    
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with columns:
        - NAME : Topology name
        - PLATEID1 : Reconstruction plate ID
        - PLATEID2 : Conjugate plate ID
        - FROMAGE : Valid from age (Ma)
        - TOAGE : Valid to age (Ma)
        - reconstruction_time : Time of reconstruction
        - gpml_type : Feature type
        - geometry : Polygon geometry (EPSG:4326)
    
    Examples
    --------
    >>> import pygplates
    >>> 
    >>> # Resolve topologies at 50 Ma
    >>> resolved_topologies = []
    >>> pygplates.resolve_topologies(
    ...     topology_features,
    ...     rotation_model,
    ...     resolved_topologies,
    ...     50.0
    ... )
    >>> 
    >>> # Convert to GeoDataFrame
    >>> topologies_gdf = create_geodataframe_topologies(
    ...     resolved_topologies,
    ...     reconstruction_time=50
    ... )
    >>> 
    >>> # Filter to specific plate
    >>> pacific_plate = topologies_gdf[topologies_gdf['PLATEID1'] == 901]
    
    Notes
    -----
    - Uses pygplates DateLineWrapper to handle Pacific dateline issues
    - Clips latitudes to ±89° to avoid polar artifacts
    - Creates multiple polygons if feature crosses dateline
    - All geometries in EPSG:4326 (WGS84)
    - Useful for spatial joins with other geological data
    
    See Also
    --------
    pygplates.resolve_topologies : Create resolved topologies
    poly_around_sub_ver2 : Uses topologies for plate ID matching
    """
    
    # some empty things to write stuff to
    names                = []
    plateid1s            = []
    plateid2s            = []
    fromages             = []
    toages               = []
    geometrys            = []
    reconstruction_times = []
    gpml_types           = []
    
    # a dateline wrapper! so that they plot nicely and do nice things in geopandas
    date_line_wrapper = pygplates.DateLineWrapper()
    
    for i, seg in enumerate(topologies):
        gpmltype = seg.get_feature_type()
        
        # polygon and wrap
        polygon = seg.get_geometry()
        wrapped_polygons = date_line_wrapper.wrap(polygon)
        for poly in wrapped_polygons:
            ring = np.array([(p.get_longitude(), p.get_latitude()) for p in poly.get_exterior_points()])
            ring[:,1] = np.clip(ring[:,1], -89, 89) # anything approaching the poles creates artefacts
            for wrapped_point in poly.get_exterior_points():
                wrapped_point_lat_lon = wrapped_point.get_latitude(), wrapped_point.get_longitude()
            
            # might result in two polys - append to loop here (otherwise we will be missing half the pacific etc)
            name = seg.get_name()
            plateid = seg.get_reconstruction_plate_id()
            conjid = seg.get_conjugate_plate_id()
            from_age, to_age = seg.get_valid_time()
            
            names.append(name)
            plateid1s.append(plateid)
            plateid2s.append(conjid)
            fromages.append(from_age)
            toages.append(to_age)
            geometrys.append(shapely.geometry.Polygon(ring)) 
            reconstruction_times.append(reconstruction_time)
            gpml_types.append(str(gpmltype))
    
    # write to geodataframe
    recon_gpd=gpd.GeoDataFrame(geometry=geometrys)
    recon_gpd['NAME'] = names
    recon_gpd['PLATEID1'] = plateid1s
    recon_gpd['PLATEID2'] = plateid2s
    recon_gpd['FROMAGE'] = fromages
    recon_gpd['TOAGE'] = toages
    
    recon_gpd['reconstruction_time'] = reconstruction_times
    recon_gpd['gpml_type'] = gpml_types
    # recon_gpd=recon_gpd.set_geometry(geometrys)
    recon_gpd = recon_gpd.set_crs("epsg:4326")
    
    return recon_gpd




def get_overriding_pid(PK,subduction_df,reconstruction_time):
    oid=[]
    k=-1
    indices=[]
    fc = [pygplates.Feature.create_reconstructable_feature(feature_type=pygplates.FeatureType.gpml_subduction_zone, geometry=pygplates.PointOnSphere(lat, lon)) for lat, lon in zip(subduction_df['Trench Latitude'].values, subduction_df['Trench Longitude'].values)]
    features=pygplates.FeatureCollection(fc)
    # Load one or more rotation files into a rotation model.
    rotation_model = PK.rotation_model
    
    topological_model = pygplates.TopologicalModel(PK.topology_features, rotation_model,anchor_plate_id=PK.anchor_plate_id)
    
    # Reconstruct the features to the current 'time'.
    reconstructed_features = []
    pygplates.reconstruct(features, rotation_model, reconstructed_features,reconstruction_time, group_with_feature=True,anchor_plate_id=PK.anchor_plate_id)
    
    # Get a snapshot of our resolved topologies at the current 'time'.
    topological_snapshot = topological_model.topological_snapshot(reconstruction_time)
    # Extract the boundary sections between our resolved topological plate polygons (and deforming networks) from the current snapshot.
    shared_boundary_sections = topological_snapshot.get_resolved_topological_sections()

    # Iterate over all reconstructed features.
    for feature, feature_reconstructed_geometries in reconstructed_features:
        k=k+1
        # Find the nearest subducting line (in the resolved topologies) to the current feature.
        # The minimum distance of the current feature (its geometries) to all subducting lines in resolved topologies.
        min_distance_to_all_subducting_lines = None
        nearest_shared_sub_segment = None
    
        # Iterate over all reconstructed geometries of the current feature.
        for feature_reconstructed_geometry in feature_reconstructed_geometries:
    
            # Iterate over the shared boundary sections of all resolved topologies.
            for shared_boundary_section in shared_boundary_sections:
    
                # Skip sections that are not subduction zones.
                # We're only interested in closeness to subducting lines.
                if shared_boundary_section.get_feature().get_feature_type() != pygplates.FeatureType.gpml_subduction_zone:
                    continue
    
                # Iterate over the shared sub-segments of the current subducting line.
                # These are the parts of the subducting line that actually contribute to topological boundaries.
                for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():
    
                    # Get the minimum distance from the current reconstructed geometry to
                    # the current subducting line.
                    min_distance_to_subducting_line = pygplates.GeometryOnSphere.distance(
                        feature_reconstructed_geometry.get_reconstructed_geometry(),
                        shared_sub_segment.get_resolved_geometry(),
                        min_distance_to_all_subducting_lines
                    )

                    # If the current subducting line is nearer than all previous ones
                    # then it's the nearest subducting line so far.
                    if min_distance_to_subducting_line is not None:
                        min_distance_to_all_subducting_lines = min_distance_to_subducting_line
                        nearest_shared_sub_segment = shared_sub_segment
    
        # We should have found the nearest subducting line.
        if nearest_shared_sub_segment is None:
            print('    Unable to find the nearest subducting line:')
            print('      either feature has no geometries or there are no subducting lines in topologies.')
            continue
    
        # Determine the overriding plate of the subducting line.
        # Get the subduction polarity of the nearest subducting line.
        subduction_polarity = nearest_shared_sub_segment.get_feature().get_enumeration(pygplates.PropertyName.gpml_subduction_polarity)
        if (not subduction_polarity or subduction_polarity == 'Unknown'):
            print(f'    Unable to find the overriding plate of the nearest subducting line "{nearest_shared_sub_segment.get_feature().get_name()}"')
            print('      subduction zone feature is missing subduction polarity property or it is set to "Unknown".')
            continue
    
        overriding_plate = None
    
        # Iterate over the topologies that are sharing the part (sub-segment) of the subducting line that is closest to the feature.
        sharing_resolved_topologies = nearest_shared_sub_segment.get_sharing_resolved_topologies()
        geometry_reversal_flags = nearest_shared_sub_segment.get_sharing_resolved_topology_geometry_reversal_flags()
        for index in range(len(sharing_resolved_topologies)):
    
            sharing_resolved_topology = sharing_resolved_topologies[index]
            geometry_reversal_flag = geometry_reversal_flags[index]

            if sharing_resolved_topology.get_resolved_boundary().get_orientation() == pygplates.PolygonOnSphere.Orientation.clockwise:
                # The current topology sharing the subducting line has clockwise orientation (when viewed from above the Earth).
                # If the overriding plate is to the 'left' of the subducting line (when following its vertices in order) and
                # the subducting line is reversed when contributing to the topology then that topology is the overriding plate.
                # A similar test applies to the 'right' but with the subducting line not reversed in the topology.
                if ((subduction_polarity == 'Left' and geometry_reversal_flag) or
                    (subduction_polarity == 'Right' and not geometry_reversal_flag)):
                    overriding_plate = sharing_resolved_topology
                    break
            else:
                # The current topology sharing the subducting line has counter-clockwise orientation (when viewed from above the Earth).
                # If the overriding plate is to the 'left' of the subducting line (when following its vertices in order) and
                # the subducting line is not reversed when contributing to the topology then that topology is the overriding plate.
                # A similar test applies to the 'right' but with the subducting line reversed in the topology.
                if ((subduction_polarity == 'Left' and not geometry_reversal_flag) or
                    (subduction_polarity == 'Right' and geometry_reversal_flag)):
                    overriding_plate = sharing_resolved_topology
                    break
    
        if not overriding_plate:
            print(f'    Unable to find the overriding plate of the nearest subducting line "{nearest_shared_sub_segment.get_feature().get_name()}"')
            print('      topology on overriding side of subducting line is missing.')
            continue
    
        # Success - we've found the overriding plate of the nearest subduction zone to the current feature.
        # So print out the overriding plate ID and the distance to nearest subducting line.
        oid.append(overriding_plate.get_feature().get_reconstruction_plate_id())
        # print(index)
        indices.append(k)
    
    
        # print(f'    overriding plate ID: {overriding_plate.get_feature().get_reconstruction_plate_id()}')
        # print(f'    distance to subducting line: {min_distance_to_all_subducting_lines * pygplates.Earth.mean_radius_in_kms:.2f} Kms')
        
    # print(len(oid))
    selected_rows = subduction_df.iloc[indices]
    selected_rows['Overriding Plate ID']=oid
    return selected_rows

def latlonlist2point(lat,lon):
    point_geometries = [Point(lon[i], lat[i]) for i in range(len(lat))]
    return gpd.GeoSeries(point_geometries)

def value_at_point(name,target_lat,target_lon):
    with rasterio.open(f'{name}') as src:

        # value=next(src.sample((target_lon,target_lat)))[0]
        sampled_values = []
        for val in src.sample([(target_lon, target_lat)]):
            sampled_values.append(val[0])
        
        # Extract the sampled value
        value = sampled_values[0]



    return value


def create_profile(start_lat, start_lon, end_lat, end_lon, interval):
    """
    Create a latitude and longitude profile with a specified interval between two points.
    """
    # distance = calculate_haversine(start_lon, start_lat, end_lon, end_lat)
    # haversine_distance
    distance =haversine_distance(start_lat, start_lon, end_lat, end_lon)
    num_points = int(distance / interval) + 1
    
    latitudes = []
    longitudes = []
    for i in range(num_points):
        fraction = i / (num_points - 1)
        lat = start_lat + fraction * (end_lat - start_lat)
        lon = start_lon + fraction * (end_lon - start_lon)
        latitudes.append(lat)
        longitudes.append(lon)
    
    return latitudes, longitudes



def interpolate_value(depth,values,interp_depth=np.arange(0, -70, -1)):

    depthC=depth.copy()
    valuesC=values.copy()


    if min(depthC)< min(interp_depth):
        depthC.append(min(interp_depth))
        valuesC.append(values[-1])
    if max(depthC)< max(interp_depth):
        depthC.append(max(interp_depth))
        valuesC.append(values[0])

    sorted_indices = np.argsort(depthC)


    sorted_depth = np.array(depthC)[sorted_indices]
    sorted_values = np.array(valuesC)[sorted_indices]
    # Create an interpolation function
    # interpolated_func = interp1d(sorted_depth, sorted_values, kind='nearest')
    interpolated_func = interp1d(sorted_depth, sorted_values, kind='slinear')
    
    
    # Interpolate t and vp
    interp_value = interpolated_func(interp_depth)

    return interp_value,interp_depth



def readcpt(filename, n_colors=256):
    """
    Read GMT/PyGMT .cpt color palette file and convert to matplotlib colormap.
    
    Parses GMT color palette files with 'r/g/b' format and creates a continuous
    matplotlib colormap for use in plotting.
    
    Parameters
    ----------
    filename : str
        Path to .cpt file
    n_colors : int, default=256
        Number of discrete colors in output colormap
    
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Continuous colormap object
    
    Examples
    --------
    >>> # Load GMT colormap
    >>> cmap = readcpt('pyDTDM/cpt/geo.cpt', n_colors=256)
    >>> 
    >>> # Use in matplotlib
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(x, y, c=values, cmap=cmap)
    
    Notes
    -----
    - Ignores comment lines starting with #
    - Skips B/F/N (background/foreground/NaN) entries
    - Expects format: value1 r1/g1/b1 value2 r2/g2/b2
    - RGB values should be in range 0-255
    - Linearly interpolates between defined colors
    
    See Also
    --------
    get_cmap : pyDTDM function for loading built-in colormaps
    """
    cpt_data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '' or line.startswith(('#','B','F','N')):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                x1 = float(parts[0])
                x2 = float(parts[2])
                r1, g1, b1 = [float(c)/255 for c in parts[1].split('/')]
                r2, g2, b2 = [float(c)/255 for c in parts[3].split('/')]
                cpt_data.append([x1, r1, g1, b1, x2, r2, g2, b2])
            except:
                # skip lines that can't be parsed
                continue
    
    if len(cpt_data) == 0:
        raise ValueError(f"No valid color data found in {filename}")
    
    # Interpolate colors
    x_min = min(row[0] for row in cpt_data)
    x_max = max(row[4] for row in cpt_data)
    xs = np.linspace(x_min, x_max, n_colors)
    color_list = []
    
    for xi in xs:
        for row in cpt_data:
            if row[0] <= xi <= row[4]:
                t = (xi - row[0]) / (row[4] - row[0] + 1e-12)
                r = row[1] + t * (row[5] - row[1])
                g = row[2] + t * (row[6] - row[2])
                b = row[3] + t * (row[7] - row[3])
                color_list.append((r, g, b))
                break
    
    return LinearSegmentedColormap.from_list("cpt_colormap", color_list, N=n_colors)


def nc_to_tiff(input_filename,outputfile):
    """
    Convert NetCDF file to GeoTIFF format.
    
    Reads a NetCDF file with spatial dimensions and converts to georeferenced
    GeoTIFF, preserving coordinate system and spatial extent.
    
    Parameters
    ----------
    input_filename : str
        Path to input NetCDF file (.nc)
    outputfile : str
        Path for output GeoTIFF file (.tif)
    
    Examples
    --------
    >>> # Convert slab depth model to GeoTIFF
    >>> nc_to_tiff(
    ...     'slab2_depth.nc',
    ...     'slab2_depth.tif'
    ... )
    Conversion Complete
    
    Notes
    -----
    - Assumes WGS84 (EPSG:4326) coordinate system
    - Expects 'x' and 'y' dimensions for longitude and latitude
    - Reverses y-coordinates if ascending (GeoTIFF requires descending)
    - Computes affine transform from coordinate arrays
    - Uses rioxarray for CRS-aware operations
    
    See Also
    --------
    xarray.open_dataarray : Read NetCDF
    rioxarray.to_raster : Write GeoTIFF
    """
    # Load data
    slab_dep = xr.open_dataarray(input_filename)

    # Assign CRS (WGS84) and spatial dimensions
    slab_dep = slab_dep.rio.write_crs("EPSG:4326", inplace=True)

    # Compute affine transform
    res_x = float(slab_dep['x'][1] - slab_dep['x'][0])
    res_y = float(slab_dep['y'][1] - slab_dep['y'][0])

    # rioxarray requires descending y-coordinates (top to bottom)
    if slab_dep.y[0] < slab_dep.y[-1]:
        slab_dep = slab_dep[::-1]

    transform = from_origin(float(slab_dep['x'][0]), float(slab_dep['y'][0]), res_x, res_y)
    slab_dep.rio.write_transform(transform, inplace=True)

    # Save as GeoTIFF
    slab_dep.rio.to_raster(outputfile)
    print ("Conversion Complete")



import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_only_colorbar(vmin, vmax, cmap, label, shrink=1.0, extend='neither', orientation='horizontal'):
    """
    Create a standalone colorbar without associated plot.
    
    Useful for creating separate colorbars for publications or presentations
    where the colorbar needs to be positioned independently.
    
    Parameters
    ----------
    vmin : float
        Minimum value of color scale
    vmax : float
        Maximum value of color scale
    cmap : str or matplotlib.colors.Colormap
        Colormap name or colormap object
    label : str
        Label for the colorbar
    shrink : float, default=1.0
        Fraction by which to shrink the colorbar (0-1)
    extend : str, default='neither'
        Whether to add extension triangles:
        - 'neither': no extensions
        - 'both': extend both ends
        - 'min': extend minimum end only
        - 'max': extend maximum end only
    orientation : str, default='horizontal'
        Colorbar orientation ('horizontal' or 'vertical')
    
    Examples
    --------
    >>> # Create horizontal colorbar for elevation
    >>> plot_only_colorbar(
    ...     vmin=-5000,
    ...     vmax=5000,
    ...     cmap='terrain',
    ...     label='Elevation (m)',
    ...     shrink=0.8,
    ...     extend='both',
    ...     orientation='horizontal'
    ... )
    
    Notes
    -----
    - Figure size automatically adjusts based on orientation
    - High DPI (300) suitable for publications
    - Colorbar positioned with padding for readability
    
    See Also
    --------
    plotgdf : Creates maps with integrated colorbars
    """
    fig, ax = plt.subplots(figsize=(6, 1.2) if orientation == 'horizontal' else (1.2, 6), dpi=300)
    ax.set_visible(False)

    # Create scalar mappable
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Colorbar
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation=orientation,
        pad=0.3,
        shrink=shrink,
        extend=extend,
    )
    cbar.set_label(label, fontsize=10)

    plt.show()
