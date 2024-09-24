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





from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter




def post_process_grid(data, output_path,data_typ, n_neighbors=3, threshold_distance=40):
    # Load the main NetCDF file
    # data = xr.open_dataset(file_path)
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
    if not os.path.exists(directory_path):
        os.makedirs(directory_path,exist_ok=True)
        print(f"Created directory: {directory_path}")
  
        # print(f"Directory already exists: {directory_path}")
        
        
        
        
def get_subduction_teeth(row, size=1):
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
    
    '''
    This function create discrete points earth surface where the parameters are calculated
    
    The initial positions of points are evenly distributed within the designated region. 
    
    INPUT:
    refinement_levels: int
    
    
    At mesh refinement level zero, the points are approximately 20 degrees apart.
    Each increase in the density level results in a halving of the spacing between points.
    Higher refinement level will take longer time to run. 
    
    
    '''
    
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
    

def df_to_NetCDF(x,y,z, statistic='mean',  grid_resolution=0.1, clip=(None,None)):
    # Define bin edges (lat and lon) based on your data range and desired bin sizes
    lon_bin_edges = np.arange(-180, 180 + grid_resolution, grid_resolution)
    lat_bin_edges = np.arange(-90, 90 + grid_resolution, grid_resolution)

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
    Calculate the distance between two points (latitude and longitude) using the Haversine formula.
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

def calculate_wma(gdfs):
    # Number of GeoDataFrames
    n = len(gdfs)
    
    # Ensure there are enough GeoDataFrames to apply the moving average
    if n == 0:
        raise ValueError("The list of GeoDataFrames is empty.")

    # Calculate weights in descending order
    weights = np.arange(n, 0, -1)  # Weights from n to 1
    weight_sum = weights.sum()

    # Initialize the WMA GeoDataFrame with the same structure as the first GeoDataFrame
    wma_df = gdfs[0].copy()

    # Apply the weights and sum the weighted values
    for column in wma_df.columns:
        if wma_df[column].dtype.kind in 'bifc':  # Only for numerical columns
            weighted_sum = np.zeros(len(wma_df))
            for i, gdf in enumerate(gdfs):
                weighted_sum += gdf[column] * weights[i]
            wma_df[column] = weighted_sum / weight_sum

    return wma_df  


def interpolate_and_save_as_geotiff(folder, param_type, start_time, end_time, depths, required_time_step=1):
    
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
    data_filled = np.nan_to_num(data, nan=0.0)
    weights = ~np.isnan(data)
    smoothed_data = gaussian_filter(data_filled * weights, sigma=sigma,radius=radius)
    weights_smooth = gaussian_filter(weights.astype(float), sigma=sigma,radius=radius)
    return smoothed_data / weights_smooth



def plotgdf(gdf,gplot,column=None,mollweide=False,time=0,cbar=False,quick=True,**kwargs):

    '''This function can be used to plot the reconstructed geodataframe at any time along with topologies and 
    features. If the data is large it will take a lot of time to plot. Turn quick to True to plot the data faster.
    However, there may be some issues with the colors.

    gdf: gpd.GeoDataFrame
    model: gplatey.PlateReconconstruction
    column: name of the colum to be plotted (str)
    time: reconstruction time (int)
    cbar: whether to display colorbar

    '''
    
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
        gplot.plot_ridges_and_transforms(ax, color='red')
        gplot.plot_trenches(ax, color='k')
        gplot.plot_subduction_teeth(ax, color='k')
    
    
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
   From Nicky's workflows 
    
    This is a function to convert topologies from pygplates into a GeoDataFrame
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
