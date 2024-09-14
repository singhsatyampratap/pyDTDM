import pygplates
import pandas as pd
import geopandas as gpd
import gplately
import ptt
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from stripy.spherical_meshes import icosahedral_mesh
import time
from shapely.geometry import Point,Polygon
from shapely.prepared import prep
from joblib import Parallel, delayed
import math
from scipy.stats import binned_statistic_2d
import xarray as xr
import rasterio

cptpath = 'cmaps'
import matplotlib.cm as cm
import os, sys

from .get_cpt import *
import cartopy.crs as ccrs
Etopo_REED = get_cmap('ETOPO1-Reed.cpt')

import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler,QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation,Input
from keras.regularizers import l2
from tensorflow import keras
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import tensorflow as tf
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show

from tensorflow.keras.models import load_model,save_model


import warnings
import pickle
from sklearn.impute import KNNImputer

from .utils import *


from rasterio.transform import from_origin



# Define a function to determine the belt value based on latitude
# Define your assign_belt function
def assign_belt(latitude, lat_band):
    return np.where(np.logical_and(latitude >= -lat_band, latitude <= lat_band), 1, 0)

    
class ClimateParameters:
    
    def __init__(self, topology_filenames=None,rotation_filenames=None, static_polygons=None,agegrid=None, coastlines=None, continents=None,anchor_plate_id=0):
    
        self.static_polygons=static_polygons
        self.agegrid=agegrid
        self.coastlines=coastlines
        self.continents=continents
        self.anchor_plate_id=anchor_plate_id
    
        try:
            self.rotation_model = pygplates.RotationModel(rotation_filenames)
        except Exception as e: 
            print("Rotation File not provided!")
            pass
        try:
            self.topology_features = pygplates.FeatureCollection()
            for topology_filename in topology_filenames:
                self.topology_features.add( pygplates.FeatureCollection(topology_filename))
        except Exception as e:
            print("Topologies not provided") 
            pass
    

        try:
            self.model = gplately.PlateReconstruction(self.rotation_model, self.topology_features, self.static_polygons,anchor_plate_id=self.anchor_plate_id)
        except Exception as e: 
            print(e)
            pass
        
    
    
    def get_time_spent_in_humid_belt(self,original_plate_model,df,reconstruction_time,window_size,lat_band,use_trench=True,drop_fraction = 0.8):
        if use_trench:
            columns=list(df.columns)
            dfc=df.copy()

            # Calculate the new Trench Longitude and Latitude values
            new_trench_longitude = dfc['Trench Longitude'] + 5 * np.sin(np.radians(dfc['Subduction Normal Angle']))
            new_trench_latitude = dfc['Trench Latitude'] + 5 * np.cos(np.radians(dfc['Subduction Normal Angle']))

            # Apply the transformations conditionally
            dfc['Trench Longitude'] = np.where(
                (new_trench_longitude >= -180) & (new_trench_longitude <= 180),
                new_trench_longitude,
                dfc['Trench Longitude']
            )

            dfc['Trench Latitude'] = np.where(
                (new_trench_latitude >= -90) & (new_trench_latitude <= 90),
                new_trench_latitude,
                dfc['Trench Latitude']
            )
            
            dfc=dfc[["Trench Longitude","Trench Latitude"]]
            dfc=dfc.drop_duplicates()
            
            drop_fraction = drop_fraction
            rows_to_drop = np.random.choice(dfc.index, size=int(drop_fraction * len(dfc)), replace=False)
            dfc = dfc.drop(rows_to_drop)
            
            clon=dfc['Trench Longitude']
            clat=dfc['Trench Latitude']
            gpts1 = gplately.Points(original_plate_model, clon, clat, time=reconstruction_time)
            rlons, rlats = gpts1.reconstruct(time=0, return_array=True)
            
            
            gpts2 = gplately.Points(self.model, rlons, rlats, time=0)
            
            # Initialize a variable to accumulate sums
            sum_result = np.zeros_like(clat, dtype=int)  # Assuming rlats has the same shape as result
            
            for time in range(reconstruction_time,reconstruction_time+window_size):
                rlons, rlats = gpts2.reconstruct(time=time, return_array=True)
                result = assign_belt(np.array(rlats), lat_band)
    
                # Accumulate the result
                sum_result += result
            
            dfc['Trench Latitude']=clat
            dfc['Trench Longitude']=clon
            dfc["Humid_Belt"]=sum_result
            columns.append("Humid_Belt")
            
            gdfc=gpd.GeoDataFrame(dfc,geometry=gpd.points_from_xy(dfc['Trench Longitude'],dfc['Trench Latitude']))
            gdf=gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'],df['Latitude']))
            
            dfk=gpd.sjoin_nearest(gdfc, gdf, how='right', distance_col='DistHB')
            dfk['Trench Longitude']=df['Trench Longitude']
            dfk['Trench Latitude']=df['Trench Latitude']
            
            dfk=dfk.drop(columns=['index_left','Trench Longitude_right','Trench Latitude_right','Trench Longitude_left','Trench Latitude_left'])
            return dfk
            
            
            
        else:
            
            clon = df['Longitude']
            clat = df['Latitude']
            gpts1 = gplately.Points(original_plate_model, clon, clat, time=reconstruction_time)
            rlons, rlats = gpts1.reconstruct(time=0, return_array=True)
           
           
            gpts2 = gplately.Points(self.model, rlons, rlats, time=0)
           
            # Initialize a variable to accumulate sums
            sum_result = np.zeros_like(clat, dtype=int)  # Assuming rlats has the same shape as result
           
            for time in range(reconstruction_time,reconstruction_time+window_size):
                rlons, rlats = gpts2.reconstruct(time=time, return_array=True)
                result = assign_belt(np.array(rlats), lat_band)
   
                # Accumulate the result
                sum_result += result
            
        
            df["Humid_Belt"]=sum_result
            
            return df
        
        
  
     
     




class MantleParameters:
    
    def __init__(self, original_folder,new_folder, parameter_type, depths, starttime,endtime,timestep,
    topology_filenames=None, 
    rotation_filenames=None,
    static_polygons=None,
    agegrid=None,
    coastlines=None,
    continents=None,
    anchor_plate_id=0):
        
        
        self.folder=original_folder
        create_directory_if_not_exists(new_folder)
        self.new_folder=new_folder
        self.param_type=parameter_type
        self.depths=depths
        self.starttime=starttime
        self.endtime=endtime
        self.timestep=timestep
        
        
        
        self.static_polygons=static_polygons
        self.agegrid=agegrid
        self.coastlines=coastlines
        self.continents=continents
        self.anchor_plate_id=anchor_plate_id
        
        try:
            self.rotation_model = pygplates.RotationModel(rotation_filenames)
        except Exception as e: 
            print("Rotation File not provided!")
            pass
        try:
            self.topology_features = pygplates.FeatureCollection()
            for topology_filename in topology_filenames:
                self.topology_features.add( pygplates.FeatureCollection(topology_filename))
        except Exception as e:
            print("Topologies not provided") 
            pass
        

        try:
            self.model = gplately.PlateReconstruction(self.rotation_model, self.topology_features, self.static_polygons,anchor_plate_id=self.anchor_plate_id)
        except Exception as e: 
            print(e)
            pass
            
        
    def get_mantle_parameters(self, df, reconstruction_time, depth_wise=True, n_jobs=-1):
            coordinates = [(x, y) for x, y in zip(df['Longitude'].values, df['Latitude'].values)]
            time=reconstruction_time
            def process_depth(new_folder, depth,param_type,time):
                raster_file = f"{new_folder}/{time}/{param_type}_{depth}.tif"
            
                if not os.path.exists(raster_file):
                    # If the raster file doesn't exist, fill the column with NaN
                    return (f"{param_type}_{depth}", np.nan * len(df))
                else:
                    mantle_data = rasterio.open(raster_file)
                    mantle = list(mantle_data.sample(coordinates))
                    mantle = [mantle[i][0] for i in range(len(mantle))]
                    return (f"{param_type}_{depth}", mantle)

            results = Parallel(n_jobs=n_jobs)(delayed(process_depth)(self.new_folder, depth,self.param_type,time) for depth in self.depths)

            for col_name, data in results:
                df[col_name] = data

            if not depth_wise:
                # Calculate the mean across all depths
                df[self.param_type] = df[[f"{self.param_type}_{depth}" for depth in self.depths]].mean(axis=1)

                # Drop the original {self.param_type}_{depth} columns
                df.drop(columns=[f"{self.param_type}_{depth}" for depth in self.depths], inplace=True)

            return df

    
    
    def get_time(self):
        times=glob.glob(f"{self.folder}/*")
        self.times=np.sort([int(time.split('/')[-1]) for time in times])
        self.times = self.times[(self.times >= self.starttime) & (self.times <= self.endtime)]
        return self.times
        
    # def get_depth(self,time):
 #        folder = f"{data_folder}/{time}/*"
 #        filenames = glob.glob(folder)
        
    
    def interpolate_mantle_data(self,n_jobs,required_timesteps):
        
        
        if not hasattr(self, 'times'):
            self.times=self.get_time()
        k=Parallel(n_jobs=n_jobs)(delayed(process_mantle_data)(self.new_folder,t, depth,self.param_type, self.folder) for t in self.times for depth in self.depths
        )
        delete_empty_folders(self.new_folder)
        
        for i in range(len(self.times)-1):
            interpolate_and_save_as_geotiff(self.new_folder,self.param_type, self.times[i], self.times[i+1], self.depths,required_timesteps)
    
    
        
    
class BhuRaster:
    def __init__(self,folder_location,Raster_Type=None):
        self.folder=folder_location
        self.Raster_Type=Raster_Type
        
    def get_parameters(self,df):
        coordinates = [(x, y) for x, y in zip(df['Longitude'].values, df['Latitude'].values)]
        data_file= rasterio.open(self.folder)
        data = list(data_file.sample(coordinates))
        data=[data[i][0] for i in range(len(data))]
        df[f"{self.Raster_Type}"]=data
        return df


def add_time_step(time, reconstructed_time_span, all_sz_df):
    print(f"{time} Ma")

    reconstructed_points = reconstructed_time_span.get_geometry_points(time, return_inactive_points=True)
    lats = []
    lons = []
    indices = []
    for i, rp in enumerate(reconstructed_points):
        if rp is not None:
            indices.append(i)
            lats.append(rp.to_lat_lon()[0])
            lons.append(rp.to_lat_lon()[1])

    reconstructed_scalars = pd.DataFrame()
    reconstructed_scalars["Latitude"] = lats
    reconstructed_scalars["Longitude"] = lons
    reconstructed_scalars["Index"] = indices

    reconstructed_gdf = gpd.GeoDataFrame(reconstructed_scalars, geometry=gpd.points_from_xy(reconstructed_scalars["Longitude"], reconstructed_scalars["Latitude"]))
    reconstructed_gdf = reconstructed_gdf.set_crs("epsg:4326").to_crs('EPSG:3857')

    subduction_df = all_sz_df[all_sz_df['Time'] == time].copy()
    subduction_gdf = gpd.GeoDataFrame(subduction_df, geometry=gpd.points_from_xy(subduction_df["Trench Longitude"], subduction_df["Trench Latitude"]))
    subduction_gdf = subduction_gdf.set_crs("epsg:4326").to_crs('EPSG:3857')
    nearest = gpd.sjoin_nearest(reconstructed_gdf, subduction_gdf, how='left', distance_col='Trench Distance')
    nearest = nearest.set_index(nearest['Index'])
    return nearest, indices        

class PlateKinematicsParameters:
    def __init__(self, topology_filenames, rotation_filenames,static_polygons,agegrid=None,coastlines=None,continents=None,anchor_plate_id=0):
        
        
        self.static_polygons=static_polygons
        self.agegrid=agegrid
        self.coastlines=coastlines
        self.continents=continents
        self.anchor_plate_id=anchor_plate_id
        
        self.rotation_model = pygplates.RotationModel(rotation_filenames,default_anchor_plate_id=anchor_plate_id)
        self.topology_features = pygplates.FeatureCollection()
        for topology_filename in topology_filenames:
                self.topology_features.add( pygplates.FeatureCollection(topology_filename))

        self.model = gplately.PlateReconstruction(self.rotation_model, self.topology_features, self.static_polygons,anchor_plate_id=self.anchor_plate_id)
        
    
    def __repr__(self):
           return f"PlateKinematicsParameters(topology_filenames={self.topology_features}, rotation_filenames={self.rotation_model}, static_polygons={self.static_polygons}, agegrid={self.agegrid}, coastlines={self.coastlines}, continents={self.continents}, anchor_plate_id={self.anchor_plate_id})"
    
   
    
    def get_mean_subduction(self,all_sz_df,reconstruction_time=0,window_size=15,weighted_mean=False,n_jobs=-1, timesteps=1,refinement_levels=9,tessellation_threshold_deg=0.1):
        projected_crs = "EPSG:8857" 
        
        print(f"Working on time={reconstruction_time} Ma")
        lons,lats=generate_mesh(refinement_levels=refinement_levels)
        lat=np.array(lats).astype(float)
        lon=np.array(lons).astype(float)
        initial_points=pygplates.MultiPointOnSphere(zip(lat,lon))
        # Elevation=np.ones(len(lat))
        initial_time = reconstruction_time+window_size
        # time_increment = timesteps
        self.topological_model = pygplates.TopologicalModel(self.topology_features, self.rotation_model,anchor_plate_id=self.anchor_plate_id)
        # no_deactivate_points = NoDeactivatePoints()

        deactivate_points = (
            pygplates.ReconstructedGeometryTimeSpan
        ).DefaultDeactivatePoints(
            deactivate_points_that_fall_outside_a_network=False
        )
        
        reconstructed_time_span = self.topological_model.reconstruct_geometry(initial_points, initial_time, 
                                                                         time_increment=1,
                                                                         deactivate_points=deactivate_points ) 


        gdfs=[]
        print("Adding time:")
        for time in range(reconstruction_time,reconstruction_time+window_size+1,timesteps):
            print(f"{time} Ma")

            reconstructed_points = reconstructed_time_span.get_geometry_points(time,return_inactive_points=True)
            lats=[]
            lons=[]
            indices=[]
            i=0
            for rp in reconstructed_points:
                if rp !=None:
                    indices.append(i)
                    lats.append(rp.to_lat_lon()[0])
                    lons.append(rp.to_lat_lon()[1])
                i+=1
            if time==reconstruction_time:
                main_indices=indices
            reconstructed_scalars=pd.DataFrame()
            reconstructed_scalars["Latitude"]=lats
            reconstructed_scalars["Longitude"]=lons
            reconstructed_scalars["Index"]=indices


            reconstructed_gdf=gpd.GeoDataFrame(reconstructed_scalars,geometry=gpd.points_from_xy(reconstructed_scalars["Longitude"],reconstructed_scalars["Latitude"]))
            reconstructed_gdf=reconstructed_gdf.set_crs("epsg:4326").to_crs(projected_crs )


            # subduction_df=self.get_subductiondf(time,  tessellation_threshold_deg=tessellation_threshold_deg,velocity_delta_time=timesteps)
            subduction_df=all_sz_df[all_sz_df['Time']==time].copy()
            subduction_gdf=gpd.GeoDataFrame(subduction_df,geometry=gpd.points_from_xy(subduction_df["Trench Longitude"],subduction_df["Trench Latitude"]))
            subduction_gdf=subduction_gdf.set_crs("epsg:4326").to_crs(projected_crs)
            nearest = gpd.sjoin_nearest(reconstructed_gdf, subduction_gdf, how='left', distance_col='Trench Distance')
            nearest=nearest.set_index(nearest['Index'])
            nearest=nearest.loc[main_indices]
            nearest=nearest.drop(columns=['geometry','Index','index_right'])
            gdfs.append(nearest)

        if weighted_mean:
            mean_df = calculate_wma(gdfs)
        else:
            sum_df = sum(gdfs)
            mean_df=sum_df/ len(gdfs)
        mean_df['Latitude']=gdfs[0]["Latitude"]
        mean_df['Longitude']=gdfs[0]["Longitude"]
        mean_gdf=gpd.GeoDataFrame(mean_df,geometry=gpd.points_from_xy(mean_df['Longitude'],mean_df['Latitude']))
        mean_gdf=mean_gdf.set_crs("epsg:4326")

        return mean_gdf

    def create_points_around_trench(self,reconstruction_time, 
                                    all_subduction_df,
                                    threshold_distance_in_kms=1400, 
                                    tessellation_threshold_deg=0.1,
                                   mesh_refinement_level=9): 
        
        projected_crs = "EPSG:8857" 
        
        # Resolve our topological plate polygons (and deforming networks) to the current 'time'.
        # We generate both the resolved topology boundaries and the boundary sections between them.
        resolved_topologies = []
        shared_boundary_sections = []
        subduction_segements=[]
        overriding_plate_ids=[]
        pygplates.resolve_topologies(self.topology_features, self.rotation_model, resolved_topologies, reconstruction_time, shared_boundary_sections) #anchor_plate_id=self.anchor_plate_id
    
        # Iterate over the shared boundary sections of all resolved topologies.
        for shared_boundary_section in shared_boundary_sections:
        
            # Skip sections that are not subduction zones.
            if shared_boundary_section.get_feature().get_feature_type() != pygplates.FeatureType.gpml_subduction_zone:
                continue
        
            # Iterate over the shared sub-segments of the current subducting line.
            # These are the parts of the subducting line that actually contribute to topological boundaries.
            for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():
        
                # Get the overriding and subducting resolved plates/networks on either side of the current shared sub-segment.
                overriding_and_subducting_plates = shared_sub_segment.get_overriding_and_subducting_plates(True)
                if overriding_and_subducting_plates:
                    overriding_plate, subducting_plate, subduction_polarity = overriding_and_subducting_plates
                    overriding_plate_ids.append(overriding_plate.get_feature().get_reconstruction_plate_id())
                    # subducting_plate_id = subducting_plate.get_feature().get_reconstruction_plate_id()
                    subduction_segements.append(shared_sub_segment.get_resolved_geometry().to_tessellated(math.radians(tessellation_threshold_deg)))
                    # print(overriding_plate_id)
        
        
        all_trench_points=[]
        oids=[]
        i=0
        for subduction_segement in subduction_segements:
            lat_lon=subduction_segement.to_lat_lon_list()
            all_trench_points.append(lat_lon)
            oids.append([overriding_plate_ids[i]]*len(lat_lon))
            i+=1
        all_trench_points=flatten_list(all_trench_points)
        
        oids=flatten_list(oids) ### overriding plate IDs at all the segmented subduction zones
        
        ### we will convert all this trench points into a GeoDataFrame that will be used to check three conditions later
        # 1. The points should lies on overridding plate IDs
        # 2. It should be closer than threshold_distance_in_kms from the trench
        ## we are using pygplates to find the overiding plate ID 

        lats=[]
        lons=[]
        for point in all_trench_points:
            lats.append(point[0])
            lons.append(point[1])
    
        sb_gdf=gpd.GeoDataFrame({'Overriding Plate ID':oids},geometry=gpd.points_from_xy(lons,lats))
        sb_gdf=sb_gdf.set_crs("epsg:4326")
        
        sb_gdf=sb_gdf.to_crs(projected_crs)
    

        ### all the other subduction parameters are calculated from all_subduction_df calculated from 
        ## self.get_subductiondf()
        subduction_df=all_subduction_df[all_subduction_df['Time']==reconstruction_time].copy()
        subduction_df=gpd.GeoDataFrame(subduction_df,geometry=gpd.points_from_xy(subduction_df['Trench Longitude'],subduction_df['Trench Latitude']))
        subduction_df=subduction_df.set_crs("epsg:4326")
        subduction_df=subduction_df.to_crs(projected_crs)
        subduction_df=gpd.sjoin_nearest(subduction_df,sb_gdf,distance_col='dist')
        subduction_df=subduction_df.to_crs("epsg:4326")


        
        ### we will now need all the tectonics plates that will be used to filter the subducting and overridding plates
        resolved_topologies = ptt.resolve_topologies.resolve_topologies_into_features(
                self.rotation_model,self.topology_features, reconstruction_time)#,anchor_plate_id=self.anchor_plate_id)
        topologies, ridge_transforms, ridges, transforms, trenches, trench_left, trench_right, other = resolved_topologies
        
        topologies_gdf=create_geodataframe_topologies(topologies, reconstruction_time)
        topologies_gdf=topologies_gdf.set_crs("epsg:4326")

        
        
        
        
        lons,lats=generate_mesh(refinement_levels=mesh_refinement_level) ### we will generate a grid in spherical coordinates
        points_gdf=gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons,lats))
        points_gdf = points_gdf.set_crs(topologies_gdf.crs)
        points_gdf=points_gdf.to_crs(projected_crs)
        
        
        
        # Reproject both GeoDataFrames to the chosen PCS
        subduction_df = subduction_df.to_crs(projected_crs)
    
    
        result=gpd.sjoin_nearest(points_gdf,subduction_df.drop(columns=['index_right']),distance_col='Trench Distance')
        result = result.to_crs("EPSG:4326")
        
        # Ensure that both GeoDataFrames are in the same coordinate reference system (CRS)
        points_gdf = points_gdf.to_crs(topologies_gdf.crs)
        
        # Perform the spatial join
        joined_gdf = gpd.sjoin(result[['geometry','Trench Distance', 'Overriding Plate ID']], topologies_gdf[['PLATEID1', 'geometry']], how='inner', predicate='within')
        # Filter the joined GeoDataFrame to keep only the matching PlateIDs
        matching_points_gdf = joined_gdf[joined_gdf['PLATEID1'] == joined_gdf['Overriding Plate ID']]
        # matching_points_gdf = result[result['PLATEID1'] == result['Overriding Plate ID']]
    
    
        matching_points_gdf=matching_points_gdf[matching_points_gdf['Trench Distance']<=threshold_distance_in_kms*1000]
        matching_points_gdf=matching_points_gdf[['geometry']]
        matching_points_gdf['Latitude']=matching_points_gdf['geometry'].y
        matching_points_gdf['Longitude']=matching_points_gdf['geometry'].x
        
        return matching_points_gdf[['Latitude','Longitude','geometry']].to_crs("EPSG:4326")
    

        
  


    def get_plate_kinematics(self,training_df,
                             all_subduction_df,
                             reconstruction_time=0,
                             window_size=15,
                             weighted_mean=False,
                             n_jobs=-1, 
                             timesteps=1,
                             refinement_levels=9,
                             tessellation_threshold_deg=0.1):
        projected_crs = "EPSG:8857" 
                             

        mean_gdf=self.get_mean_subduction(all_subduction_df,reconstruction_time,window_size=window_size,weighted_mean=weighted_mean,
                                 n_jobs=n_jobs, timesteps=timesteps,refinement_levels=refinement_levels,tessellation_threshold_deg=tessellation_threshold_deg)


    

        print("Calculating Trench Distance")
        
        
        Data_gdf=gpd.GeoDataFrame(training_df,geometry=gpd.points_from_xy(training_df['Longitude'],training_df['Latitude']))
        Data_gdf=Data_gdf.set_crs("epsg:4326").to_crs(projected_crs)
        mean_gdf=mean_gdf.to_crs(projected_crs)
        training_points_all=gpd.sjoin_nearest(Data_gdf, mean_gdf, how='left', distance_col='Dist')
        training_points_all=training_points_all.to_crs("epsg:4326")
        training_points_all["Latitude"]=training_points_all["Latitude_left"]
        training_points_all["Longitude"]=training_points_all["Longitude_left"]
        training_points_all=training_points_all.drop(columns=['Latitude_left', 'Longitude_left','geometry','Latitude_right', 'Longitude_right'])
  
        return training_points_all
        
    
    def get_subductiondf(self, reconstruction_time,  tessellation_threshold_deg=0.1,velocity_delta_time=1):
        """
        Calculate subduction zone parameters and create a GeoDataFrame with the results.

        Parameters:
        - model: The plate tectonic model to use for the reconstruction.
        - agegrid: Path to the directory containing age grid files.
        - time: The reconstruction time in millions of years.
        - tessellation_threshold_deg: The threshold for tessellation in degrees (default is 0.1).

        Returns:
        - subduction: A GeoDataFrame containing subduction zone parameters.
        """
    
        # Set tessellation threshold in radians
        tessellation_threshold_radians = np.deg2rad(tessellation_threshold_deg)

        # Calculate subduction convergence parameters
        subduction_data = self.model.tessellate_subduction_zones(
            reconstruction_time,
            tessellation_threshold_radians,
            ignore_warnings=True,
            output_distance_to_nearest_edge_of_trench=True,
            output_distance_to_start_edge_of_trench=True,
            output_convergence_velocity_components=True,
            output_trench_absolute_velocity_components=True,
            output_subducting_absolute_velocity=True,
            output_subducting_absolute_velocity_components=True,
            velocity_delta_time=velocity_delta_time
        )

        # Extract subduction data components
        subduction_lon = subduction_data[:, 0]
        subduction_lat = subduction_data[:, 1]
        subduction_vel = subduction_data[:, 2] * 1e-2
        subduction_angle = subduction_data[:, 3]
        subduction_norm = subduction_data[:, 7]
        subduction_pid_sub = subduction_data[:, 8]
        subduction_pid_over = subduction_data[:, 9]
        subduction_length = np.deg2rad(subduction_data[:, 6]) * gplately.EARTH_RADIUS * 1e3  # in metres
        subduction_convergence = np.fabs(subduction_data[:, 2]) * 1e-2 * np.cos(np.radians(subduction_data[:, 3]))
        subduction_migration = np.fabs(subduction_data[:, 4]) * 1e-2 * np.cos(np.radians(subduction_data[:, 5]))
        distance_to_nearest_edge_of_trench = np.deg2rad(subduction_data[:, 10]) * gplately.EARTH_RADIUS * 1e3  # in metres
        distance_to_start_edge_of_trench = np.deg2rad(subduction_data[:, 11]) * gplately.EARTH_RADIUS * 1e3  # in metres
        convergence_velocity_orth = subduction_data[:, 12]
        convergence_velocity_par = subduction_data[:, 13]
        trench_absolute_velocity_orth = subduction_data[:, 14]
        trench_absolute_velocity_par = subduction_data[:, 15]
        subduction_plate_vel = subduction_data[:, 16]
        subduction_plate_obliquity = subduction_data[:, 17]
        subducting_absolute_velocity_orth = subduction_data[:, 18]
        subducting_absolute_velocity_par = subduction_data[:, 19]
         
        # Clip subduction convergence to non-negative values
        subduction_convergence = np.clip(subduction_convergence, 0, 1e99)
        # Calculate velocity ratio (vratio)
        vratio = (subduction_convergence + subduction_migration) / (subduction_convergence + 1e-22)
        vratio[subduction_plate_vel < 0] *= -1
        vratio = np.clip(vratio, 0, 1)
        
        
        # Create a GeoDataFrame with subduction parameters
        subduction = gpd.GeoDataFrame({
            'Trench Latitude': subduction_lat,
            'Trench Longitude': subduction_lon,
            'Convergence Rate': subduction_convergence,
            'Migration Rate': subduction_migration,
            'Subduction Velocity': subduction_vel,
            'Obliquity Angle': subduction_angle,
            'Subduction Normal Angle': subduction_norm,
            'Subduction Length': subduction_length,
            'Subduction Plate Velocity': subduction_plate_vel,
            'Subduction Plate Obliquity': subduction_plate_obliquity,
            # 'Subduction Volume Rate': subduction_vol_rate,
            'vratio': vratio,
            # 'Plate Thickness': plate_thickness,
            'Nearest Trench Edge': distance_to_nearest_edge_of_trench,
            'Start Edge Trench': distance_to_start_edge_of_trench,
            'Convergence Velocity Orthogonal': convergence_velocity_orth,
            'Convergence Velocity Parallel': convergence_velocity_par,
            'Trench Velocity Orthogonal': trench_absolute_velocity_orth,
            'Trench Velocity Parallel': trench_absolute_velocity_par,
            'Subducting Velocity Orthogonal': subducting_absolute_velocity_orth,
            'Subducting Velocity Parallel': subducting_absolute_velocity_par,
            'Subducting Plate ID': subduction_pid_sub,
            'Trench Plate ID':subduction_pid_over
        }, geometry=gpd.points_from_xy(subduction_lon,subduction_lat))
        
        
        
        if self.agegrid!=None:
            
            # Construct age grid file path
            # try:
            try:
                age_grid_file = find_filename_with_number(self.agegrid,reconstruction_time)
                print(f"Using agegrid file : {age_grid_file}")
                # Interpolate seafloor age at subduction one locations
                age_raster = gplately.Raster(data=age_grid_file, plate_reconstruction=self.model, time=reconstruction_time)
            # except:
            #     try:
            #         age_grid_file = find_filename_with_number2(self.agegrid,reconstruction_time)
            #         print(f"Using agegrid file : {age_grid_file}")
            #         # Interpolate seafloor age at subduction one locations
            #         age_raster = gplately.Raster(data=age_grid_file, plate_reconstruction=self.model, time=reconstruction_time)
            except Exception as e:
                print("No Seafloor age grid found!")
                        
            age_raster.fill_NaNs(inplace=True)
            age_interp = age_raster.interpolate(subduction_lon, subduction_lat)

            # Calculate plate thickness from seafloor age
            plate_thickness = gplately.tools.plate_isotherm_depth(age_interp)

            # Calculate subduction volume rate (in km^3/yr)
            subduction_vol_rate = plate_thickness * subduction_length * subduction_convergence  # in m^3/yr
            subduction_vol_rate *= 1e-9  # convert to km^3/yr

            # Calculate subduction flux
            subduction_flux = plate_thickness * subduction_convergence

            subduction['Subduction Volume Rate']=subduction_vol_rate
            subduction['Plate Thickness']=plate_thickness
            subduction['Subduction Flux']=subduction_flux
            # except Exception as e:
            #     print(e)

      
    
        return subduction
        
     


    
    
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from joblib import Parallel, delayed
from geopy.distance import great_circle as haversine_distance

def create_profile_data(subduction_df):
    parameters = {col: [] for col in subduction_df.columns[:-1]}
    
    results = Parallel(n_jobs=-1)(
        delayed(process_profile)(subduction_df.iloc[i], subduction_df.iloc[i + 1], subduction_df.columns[:-1])
        for i in range(len(subduction_df) - 1)
    )
    
    dist = []
    polygons = []
    trench_distances = []
    point_lats = []
    point_lons = []

    for res in results:
        dist.extend(res['dist'])
        polygons.extend(res['polygons'])
        trench_distances.extend(res['trench_distances'])
        point_lats.extend(res['point_lats'])
        point_lons.extend(res['point_lons'])
        
        for key in parameters.keys():
            parameters[key].extend(res['parameters'][key])

    df = pd.DataFrame(parameters)
    df['Trench Distance'] = trench_distances
    df['Latitude'] = point_lats
    df['Longitude'] = point_lons

    return df

def process_profile(row1, row2, parameter_columns, n_steps=20):
    results = {
        'dist': [],
        'polygons': [],
        'trench_distances': [],
        'point_lats': [],
        'point_lons': [],
        'parameters': {col: [] for col in parameter_columns}
    }

    y1, y2 = row1['Trench Latitude'], row2['Trench Latitude']
    x1, x2 = row1['Trench Longitude'], row2['Trench Longitude']

    dist = haversine_distance((y1, x1), (y2, x2)).km
    results['dist'].append(dist)

    if dist <= 120.0:
        try:
            angle1 = np.radians(row1['Subduction Normal Angle'])
            angle2 = np.radians(row2['Subduction Normal Angle'])

            dlon1, dlat1 = n_steps * np.sin(angle1), n_steps * np.cos(angle1)
            dlon2, dlat2 = n_steps * np.sin(angle2), n_steps * np.cos(angle2)

            ilon1, ilat1 = x1 + dlon1, y1 + dlat1
            ilon2, ilat2 = x2 + dlon2, y2 + dlat2

            coords = [(x1, y1), (x2, y2), (ilon2, ilat2), (ilon1, ilat1), (x1, y1)]
            polygon = Polygon(coords)
            results['polygons'].append(polygon)

            _, lats, lons = multipoints_from_polygon(polygon, resolution=0.09)
            

            # for lat, lon in zip(lats, lons):
#                 trench_distance = haversine_distance(((y1 + y2) / 2, (x1 + x2) / 2), (lat, lon)).km
#                 results['trench_distances'].append(trench_distance)
#                 results['point_lats'].append(lat)
#                 results['point_lons'].append(lon)
#
#                 for column in parameter_columns:
#                     try:
#                         results['parameters'][column].append((row1[column] + row2[column]) / 2)
#                     except:
#                         pass
        except:
            pass

    return results

   

class RFModel:
    def __init__(self, training_df, training_variables, target_variable, n_estimators=100, max_depth=8,min_samples_split=10,min_samples_leaf=5,max_features='sqrt',parallel=-1,random_state=22):
        """
        Initialize the RFTopoModel with the given parameters.
        
        Parameters:
        - training_df (pd.DataFrame): The input DataFrame containing the data.
        - remove_variable (str): The variable/column to be removed from the features.
        - n_estimators (int): The number of trees in the forest.
        - max_depth (int): The maximum depth of the tree.
        """
        self.df = training_df.copy().drop_duplicates()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.max_features=max_features
        self.random_state=random_state
        self.columns = training_variables
        self.target_variable=target_variable
        self.model = None
        self.n_jobs=parallel
        
    def __repr__(self):
        """
        Return a string representation of the RFTopoModel instance.
        """
        return (f"RFTopoModel(n_estimators={self.n_estimators}, "
                f"Features={self.columns}, "
                f"Target Variable={self.target_variable}")

    def __str__(self):
        """
        Return a readable string representation of the RFTopoModel instance.
        """
        return (f"RFTopoModel with {self.n_estimators} trees, "
                f"max depth of {self.max_depth}, "
                f"trained to predict elevation")

    def __len__(self):
        """
        Return the number of rows in the DataFrame.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Get an item (row) from the DataFrame by index.
        """
        return self.df.iloc[index]

    def __setitem__(self, index, value):
        """
        Set an item (row) in the DataFrame by index.
        """
        self.df.iloc[index] = value

    def __delitem__(self, index):
        """
        Delete an item (row) from the DataFrame by index.
        """
        self.df = self.df.drop(index).reset_index(drop=True)

    def fit(self):
        """
        Create and train a Random Forest model using the provided data.
        
        Returns:
        - df (pd.DataFrame): The DataFrame with the predicted elevation and RMSE columns.
        - model (RandomForestRegressor): The trained RandomForest model.
        - columns (list): List of feature columns used in the model.
        """
        features_nan = self.df[self.columns].copy()
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        # imputer = SimpleImputer(strategy='mean')
        features = pd.DataFrame(imputer.fit_transform(features_nan), columns=features_nan.columns)

        print("Creating RFTimeTopo model based on parameters:")
        for col in self.columns:
            print(col)

        y = self.df[self.target_variable].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=22)
        X_val, X_tes, y_val, y_tes = train_test_split(X_test, y_test, test_size=0.5, random_state=22)

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        self.model.fit(X_train, y_train.ravel())

        y_pred = self.model.predict(X_tes)
        mse = mean_squared_error(y_tes, y_pred)
        print(f"Mean Square Error {np.sqrt(mse)}")

        y_pred_full = self.model.predict(features)
        self.df[f'Predicted {self.target_variable}'] = y_pred_full
        self.df['Difference'] = self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']

        # return self.df, self.model, self.columns

    def predict(self, df):
        """
        Predict elevation using the trained RandomForest model.
        
        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the features for prediction.
        
        Returns:
        - y_pred (np.array): The predicted elevation values.
        """
     

        features_nan = df[self.columns]
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        features = pd.DataFrame(imputer.fit_transform(features_nan), columns=features_nan.columns)
        features = features.reindex(columns=self.columns)
        
        y_pred = self.model.predict(features)
        return y_pred

    def plot_difference(self, plate_model,kwargs1,kwargs2,quick=True):
        """
        Plot the actual and predicted elevation values on a map.
        
        
        """
        df=self.df
        gdf = gpd.GeoDataFrame(df, geometry=[Point(x, y) for x, y in zip(df['Longitude'].values, df['Latitude'].values)])
        gdf = gdf.set_crs(epsg=4326)
        kwargs = kwargs1
        plotgdf(gdf,plate_model, mollweide=True, column=self.target_variable, cbar=True,quick=quick, **kwargs)
        plotgdf(gdf,plate_model, mollweide=True, column=f'Predicted {self.target_variable}', cbar=True, **kwargs)
        kwargs = kwargs2
        plotgdf(gdf,plate_model, mollweide=True, column='Difference', cbar=True,quick=quick, **kwargs)
    
    
    def plot_feature_importance(self):
           """
           Plot the feature importance of the trained Random Forest model.
           """
           if self.model is None:
               print("Model has not been trained yet.")
               return
        
           importances = self.model.feature_importances_
           indices = np.argsort(importances)[::-1]

           plt.figure(figsize=(12, 6))
           plt.title("Feature Importances")
           plt.bar(range(len(importances)), importances[indices], align="center")
           plt.xticks(range(len(importances)), [self.columns[i] for i in indices], rotation=90)
           plt.xlabel("Feature")
           plt.ylabel("Importance")
           plt.tight_layout()
           plt.show()
    
    def get_error(self):
        """
        Calculate the Root Mean Squared Error (RMSE) of the predicted elevation values.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame containing the actual and predicted elevation values.
        
        Returns:
        - error (float): The RMSE value.
        """
        if f'Predicted {self.target_variable}' in self.df.columns:
            return np.mean(np.sqrt((self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']) ** 2))
        else:
            self.Fit()
            return np.mean(np.sqrt((self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']) ** 2))
            

    def save_model(self, loc):
        """
        Save the DataFrame, trained RandomForest model, and columns to the specified location.
        
        Parameters:
        - loc (str): Directory location to save the files.
        """
        try:

        
            # Save df to CSV
            self.df.to_csv(f'{loc}/df_data.csv', index=False)

            # Save model using pickle
            with open(f'{loc}/random_forest_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)

            # Save columns to text file
            with open(f'{loc}/columns.txt', 'w') as f:
                f.write('\n'.join(self.columns))

            with open(f'{loc}/random_forest_class.pkl', 'wb') as f:
                pickle.dump(self, f)
                
            print(f"Saved DataFrame, model, and columns to {loc}")
            
        except Exception as e:
            print(f"Error saving files: {str(e)}")

    @staticmethod
    def load_model(loc):
        """
        Load the DataFrame, trained RandomForest model, and columns from the specified location.
        
        Parameters:
        - loc (str): Directory location where files are saved.
        
        Returns:
        - df (pd.DataFrame): The loaded DataFrame.
        - model (RandomForestRegressor): The loaded RandomForest model.
        - columns (list): List of column names used in the model.
        """
        df = None
        model = None
        columns = None

        try:
            # Read DataFrame from CSV
            df = pd.read_csv(f'{loc}/df_data.csv')

            # Read trained model from pickle
            # with open(f'{loc}/random_forest_class.pkl', 'rb') as f:
#                 RFMain = pickle.load(f)


            with open(f'{loc}/random_forest_model.pkl', 'rb') as f:
                model = pickle.load(f)

            # Read columns from text file
            with open(f'{loc}/columns.txt', 'r') as f:
                columns = f.read().splitlines()

            print(f"Loaded DataFrame, model, and columns from {loc}")
        except FileNotFoundError as fnf_error:
            print(f"File not found error: {str(fnf_error)}")
        except Exception as e:
            print(f"Error loading files: {str(e)}")

        return df, model,columns
        
        


class EBMModel:
    def __init__(self,df, remove_variable, max_bins=255, max_interactions=0):
        """
        Initialize the EBMTopoModel with parameters.

        Parameters:
        - remove_variable (str): The column to be removed from the DataFrame.
        - max_bins (int): Maximum number of bins per feature.
        - max_interactions (int): Maximum number of interaction terms.
        """
        self.remove_variable = remove_variable
        self.max_bins = max_bins
        self.max_interactions = max_interactions
        self.model = None
        self.df = df
        self.columns = None

    def fit(self):
        """
        Fit the EBM model to the DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing features and target.
        """
        self.df =self.df.drop_duplicates()
        features_nan = self.df.drop(self.remove_variable, axis=1).copy()
        imputer = SimpleImputer(strategy='mean')
        features = pd.DataFrame(imputer.fit_transform(features_nan), columns=features_nan.columns)
        self.columns = features.columns
        y = self.df['Elevation'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=22)
        
        self.model = ExplainableBoostingRegressor(max_bins=self.max_bins, max_interactions=self.max_interactions, random_state=22)
        self.model.fit(X_train, y_train.ravel())

        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Square Error: {np.sqrt(mse)}")

        y_pred_full = self.model.predict(features)
        self.df['Predicted Elevation'] = y_pred_full
        self.df['RMSE'] = np.sqrt((self.df['Elevation'] - self.df['Predicted Elevation']) ** 2)
        
    def plot_difference(self, plate_model,kwargs1,kwargs2,quick=True):
        """
        Plot the actual and predicted elevation values on a map.
        
        
        """
        df=self.df
        gdf = gpd.GeoDataFrame(df, geometry=[Point(x, y) for x, y in zip(df['Longitude'].values, df['Latitude'].values)])
        gdf = gdf.set_crs(epsg=4326)
        kwargs = kwargs1
        plotgdf(gdf,plate_model, mollweide=True, column=self.target_variable, cbar=True,quick=quick, **kwargs)
        plotgdf(gdf,plate_model, mollweide=True, column=f'Predicted {self.target_variable}', cbar=True, **kwargs)
        kwargs = kwargs2
        plotgdf(gdf,plate_model, mollweide=True, column='Difference', cbar=True,quick=quick, **kwargs)

    def predict(self, df):
        """
        Predict the target using the trained EBM model.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing features.

        Returns:
        - y_pred (np.array): The predicted values.
        """
        if 'Smoothed Elevation' in self.columns:
            self.columns.remove("Smoothed Elevation")
        if 'Elevation' in self.columns:
            self.columns.remove("Elevation")

        features_nan = df[self.columns]
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        features = pd.DataFrame(imputer.fit_transform(features_nan), columns=features_nan.columns)
        features = features.reindex(columns=self.columns)
        
        y_pred = self.model.predict(features)
        return y_pred

    def plot_feature_importance(self):
        """
        Plot the feature importance of the trained EBM model.
        """
        if self.model is None:
            print("Model has not been trained yet.")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [self.columns[i] for i in indices], rotation=90)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()

    def plot_prediction(self):
        """
        Plot the actual and predicted elevations along with RMSE.
        """
        gdf = gpd.GeoDataFrame(self.df, geometry=[Point(x, y) for x, y in zip(self.df['Longitude'].values, self.df['Latitude'].values)])
        gdf = gdf.set_crs(epsg=4326)
        kwargs = {'vmin': -11000, 'vmax': 6000, 'cmap': Etopo_REED}
        plotgdf(gdf, mollweide=True, column='Elevation', cbar=True, **kwargs)
        plotgdf(gdf, mollweide=True, column='Predicted Elevation', cbar=True, **kwargs)
        kwargs = {'cmap': 'cool', 'label': 'RMSE(m)', 'vmin': 0, 'vmax': 1000}
        plotgdf(gdf, mollweide=True, column='RMSE', cbar=True, **kwargs)

    def save_model(self, loc):
        """
        Save DataFrame, trained EBM model, and columns to specified location.

        Parameters:
        - loc (str): Directory location to save the files.
        """
        try:
            # Save df to CSV
            self.df.to_csv(f'{loc}/df_data.csv', index=False)

            # Save model using pickle
            with open(f'{loc}/ebm_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)

            # Save columns to text file
            with open(f'{loc}/columns.txt', 'w') as f:
                f.write('\n'.join(self.columns))

            print(f"Saved DataFrame, model, and columns to {loc}")
        except Exception as e:
            print(f"Error saving files: {str(e)}")

    def load_model(self, loc):
        """
        Read DataFrame, trained EBM model, and columns from specified location.

        Parameters:
        - loc (str): Directory location where files are saved.

        Returns:
        - df (pd.DataFrame): Loaded DataFrame.
        - model (ExplainableBoostingRegressor): Loaded EBM model.
        - columns (list): List of column names used in the model.
        """
        try:
            # Read DataFrame from CSV
            self.df = pd.read_csv(f'{loc}/df_data.csv')

            # Read trained model from pickle
            with open(f'{loc}/ebm_model.pkl', 'rb') as f:
                self.model = pickle.load(f)

            # Read columns from text file
            with open(f'{loc}/columns.txt', 'r') as f:
                self.columns = f.read().splitlines()

            print(f"Loaded DataFrame, model, and columns from {loc}")
        except FileNotFoundError as fnf_error:
            print(f"File not found error: {str(fnf_error)}")
        except Exception as e:
            print(f"Error loading files: {str(e)}")

        return self.df, self.model, self.columns

    def __repr__(self):
        """
        Return a string representation of the EBMTopoModel instance.
        """
        return (f"EBMTopoModel(max_bins={self.max_bins}, "
                f"max_interactions={self.max_interactions}, "
                f"remove_variable={self.remove_variable})")

    def __str__(self):
        """
        Return a readable string representation of the EBMTopoModel instance.
        """
        return (f"EBMTopoModel with max bins {self.max_bins}, "
                f"max interactions {self.max_interactions}, "
                f"trained to predict elevation with the column '{self.remove_variable}' removed.")

    def __len__(self):
        """
        Return the number of rows in the DataFrame.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Get an item (row) from the DataFrame by index.
        """
        return self.df.iloc[index]

    def __setitem__(self, index, value):
        """
        Set an item (row) in the DataFrame by index.
        """
        self.df.iloc[index] = value

    def __delitem__(self, index):
        """
        Delete an item (row) from the DataFrame by index.
        """
        self.df = self.df.drop(index).reset_index(drop=True)
   

        
   


from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer
from tensorflow.keras import layers
from tensorflow.keras import models, callbacks, optimizers

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer
from tensorflow.keras import layers, models, callbacks, optimizers

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from tensorflow.keras import Sequential, layers, callbacks, optimizers
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


class DLModel_tf:
    def __init__(self, training_df, training_variables, target_variable, nodes=8, n_epoch=20, dropout=0.2, bn=True, lr="OnPlateau", scaler="RobustScaler"):
        self.df = training_df.copy()
        self.columns = training_variables
        self.target_variable = target_variable
        self.nodes = max(8, nodes)
        self.n_epoch = n_epoch
        self.dropout = dropout
        self.bn = bn
        self.lr = lr
        self.scaler = scaler
        self.model = None
        self.scaler1 = None
        self.scaler2 = None
        self.history = None
    
    def fit(self):
        features_nan = self.df[self.columns].copy()
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        features = pd.DataFrame(imputer.fit_transform(features_nan), columns=features_nan.columns)
        
        if self.scaler == "MinMaxScaler":
            self.scaler1 = MinMaxScaler()
            self.scaler2 = MinMaxScaler()
        elif self.scaler == "RobustScaler":
            self.scaler1 = RobustScaler()
            self.scaler2 = RobustScaler()
        elif self.scaler == "StandardScaler":
            self.scaler1 = StandardScaler()
            self.scaler2 = StandardScaler()
        else:
            self.scaler1 = QuantileTransformer(output_distribution='uniform')
            self.scaler2 = QuantileTransformer(output_distribution='uniform')
        print("Creating RFTimeTopo model based on parameters:")
        for col in self.columns:
            print(col)
        normalized_features = self.scaler1.fit_transform(features)
        normalized_y = self.scaler2.fit_transform(self.df[self.target_variable].values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(normalized_features, normalized_y, test_size=0.7, random_state=22)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=22)

        dropout=self.dropout
        activation="relu"
        model = Sequential()
        model.add(Dense(16*self.nodes,input_dim=len(self.columns)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Activation(activation)) 
        model.add(Dense(8*self.nodes))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Activation(activation)) 
        model.add(Dense(4*self.nodes))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Activation(activation)) 
        # model.add(Dense(2*self.nodes))
        # model.add(BatchNormalization())
        # model.add(Dropout(dropout))
        # model.add(Activation(activation))
        model.add(Dense(self.nodes))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Activation(activation))
        model.add(Dense(1))
        print(model)
        if self.lr == "Step":
            reduce_lr = callbacks.LearningRateScheduler(self.step_decay, verbose=1)
        elif self.lr == "OnPlateau":
            reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        else:
            reduce_lr = None
        
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
        
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam())
        
        history = model.fit(X_train, y_train, epochs=self.n_epoch, batch_size=32, validation_data=(X_val, y_val),
                            callbacks=[reduce_lr, early_stopping], verbose=1)
        
        mse = model.evaluate(X_test, y_test)
        print(f"Root Mean Squared Error: {np.sqrt(mse)}")
        
        y_pred = model.predict(normalized_features)
        scaled_back_y_pred = self.scaler2.inverse_transform(y_pred.reshape(-1, 1))
        self.df[f'Predicted {self.target_variable}'] = scaled_back_y_pred 
        self.df['Difference'] = self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']
        
        self.model = model
        self.history = history
        self.scaler1 = self.scaler1
        self.scaler2 = self.scaler2
        self.df = self.df
    
    def step_decay(self, epoch):
        initial_lr = 0.01
        drop = 0.5
        epochs_drop = 10
        new_lr = initial_lr * (drop ** (epoch // epochs_drop))
        return new_lr
    def predict(self, df):
        """
        Predict the target using the trained deep learning model.
        """
        columns = list(self.columns)
      
        features_nan = df[columns]
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        features = pd.DataFrame(imputer.fit_transform(features_nan), columns=features_nan.columns)

        features = features.reindex(columns=columns)

        normalized_features = self.scaler1.transform(features)
        y_pred = self.model.predict(normalized_features)
        scaled_back_y_pred = self.scaler2.inverse_transform(y_pred.reshape(-1, 1))

        return scaled_back_y_pred
    
    def get_error(self):
            """
            Calculate the Root Mean Squared Error (RMSE) of the predicted elevation values.
        
            Parameters:
            - df (pd.DataFrame): The DataFrame containing the actual and predicted elevation values.
        
            Returns:
            - error (float): The RMSE value.
            """
            if f'Predicted {self.target_variable}' in self.df.columns:
                return np.sqrt(np.mean((self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']) ** 2))
            else:
                self.Fit()
                return np.sqrt(np.mean((self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']) ** 2))
            
    

    def plot_loss_curve(self):
        """
        Plot the loss curve during training.
        """
        if self.history is None:
            print("Model has not been trained yet.")
            return

        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_prediction(self):
        """
        Plot the actual and predicted elevations along with RMSE.
        """
        gdf = gpd.GeoDataFrame(self.df, geometry=[Point(x, y) for x, y in zip(self.df['Longitude'].values, self.df['Latitude'].values)])
        gdf = gdf.set_crs(epsg=4326)
        kwargs = {'vmin': -11000, 'vmax': 6000, 'cmap': 'viridis'}
        plotgdf(gdf, mollweide=True, column='Elevation', cbar=True, **kwargs)
        plotgdf(gdf, mollweide=True, column='Predicted Elevation', cbar=True, **kwargs)
        kwargs = {'cmap': 'cool', 'label': 'RMSE(m)', 'vmin': 0, 'vmax': 1000}
        plotgdf(gdf, mollweide=True, column='RMSE', cbar=True, **kwargs)

    def save_model(self, loc):
        """
        Save DataFrame, trained model, scalers, columns, and training history to specified location.
        """
        try:
            if not os.path.exists(loc):
                os.makedirs(loc)

            self.df.to_csv(f'{loc}/df_data.csv', index=False)
            self.model.save(f'{loc}/deep_time_topo_model.h5')

            with open(f'{loc}/scaler1.pkl', 'wb') as f:
                pickle.dump(self.scaler1, f)
            with open(f'{loc}/scaler2.pkl', 'wb') as f:
                pickle.dump(self.scaler2, f)

            with open(f'{loc}/columns.txt', 'w') as f:
                f.write('\n'.join(self.columns))

            with open(f'{loc}/training_history.pkl', 'wb') as f:
                pickle.dump(self.history.history, f)

            print(f"Saved model components to {loc}")
        except Exception as e:
            print(f"Error saving files: {str(e)}")

    def load_model(self, loc):
        """
        Load DataFrame, trained model, scalers, columns, and training history from specified location.
        """
        try:
            self.df = pd.read_csv(f'{loc}/df_data.csv')
            self.model = load_model(f'{loc}/deep_time_topo_model.h5')

            with open(f'{loc}/scaler1.pkl', 'rb') as f:
                self.scaler1 = pickle.load(f)
            with open(f'{loc}/scaler2.pkl', 'rb') as f:
                self.scaler2 = pickle.load(f)

            with open(f'{loc}/columns.txt', 'r') as f:
                self.columns = f.read().splitlines()

            with open(f'{loc}/training_history.pkl', 'rb') as f:
                history_dict = pickle.load(f)
                self.history = type('', (), {})()
                self.history.history = history_dict

            print(f"Loaded model components from {loc}")
        except FileNotFoundError as fnf_error:
            print(f"File not found error: {str(fnf_error)}")
        except Exception as e:
            print(f"Error loading files: {str(e)}")

    def __repr__(self):
        return f"DeepTimeTopoModel(nodes={self.nodes}, n_epoch={self.n_epoch}, dropout={self.dropout}, bn={self.bn}, lr={self.lr}, scaler={self.scaler})"



class DLModel_mbackend:
    def __init__(self, training_df, training_variables, target_variable, backend="tensorflow", nodes=8, n_epoch=20, dropout=0.2, bn=True, lr="OnPlateau", scaler="RobustScaler"):
        self.df = training_df.copy()
        self.columns = training_variables
        self.target_variable = target_variable
        self.backend = backend.lower()
        self.nodes = max(8, nodes)
        self.n_epoch = n_epoch
        self.dropout = dropout
        self.bn = bn
        self.lr = lr
        self.scaler = scaler
        self.model = None
        self.scaler1 = None
        self.scaler2 = None
        self.history = None
    
    def fit(self):
        features_nan = self.df[self.columns].copy()
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        features = pd.DataFrame(imputer.fit_transform(features_nan), columns=features_nan.columns)
        
        # Select scaler
        if self.scaler == "MinMaxScaler":
            self.scaler1 = MinMaxScaler()
            self.scaler2 = MinMaxScaler()
        elif self.scaler == "RobustScaler":
            self.scaler1 = RobustScaler()
            self.scaler2 = RobustScaler()
        elif self.scaler == "StandardScaler":
            self.scaler1 = StandardScaler()
            self.scaler2 = StandardScaler()
        else:
            self.scaler1 = QuantileTransformer(output_distribution='uniform')
            self.scaler2 = QuantileTransformer(output_distribution='uniform')
        
        print(f"Creating model using {self.backend} backend with the following parameters:")
        for col in self.columns:
            print(col)
        
        # Normalize features and target
        normalized_features = self.scaler1.fit_transform(features)
        normalized_y = self.scaler2.fit_transform(self.df[self.target_variable].values.reshape(-1, 1))

        # Train-test-validation split
        X_train, X_test, y_train, y_test = train_test_split(normalized_features, normalized_y, test_size=0.7, random_state=22)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=22)
        
        if self.backend == "tensorflow":
            self._fit_tensorflow(X_train, y_train, X_val, y_val, X_test, y_test)
        elif self.backend == "pytorch":
            self._fit_pytorch(X_train, y_train, X_val, y_val, X_test, y_test)
        else:
            raise ValueError(f"Unsupported backend '{self.backend}'. Choose either 'tensorflow' or 'pytorch'.")
    
    def _fit_tensorflow(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # Build TensorFlow model
        dropout = self.dropout
        activation = "relu"
        model = Sequential()
        model.add(layers.Dense(16 * self.nodes, input_dim=len(self.columns)))
        if self.bn:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))
        model.add(layers.Activation(activation))
        model.add(layers.Dense(8 * self.nodes))
        if self.bn:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))
        model.add(layers.Activation(activation))
        model.add(layers.Dense(4 * self.nodes))
        if self.bn:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))
        model.add(layers.Activation(activation))
        model.add(layers.Dense(self.nodes))
        if self.bn:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))
        model.add(layers.Activation(activation))
        model.add(layers.Dense(1))

        print(model.summary())

        if self.lr == "Step":
            reduce_lr = callbacks.LearningRateScheduler(self.step_decay, verbose=1)
        elif self.lr == "OnPlateau":
            reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        else:
            reduce_lr = None
        
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
        
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam())
        
        history = model.fit(X_train, y_train, epochs=self.n_epoch, batch_size=32, validation_data=(X_val, y_val),
                            callbacks=[reduce_lr, early_stopping], verbose=1)
        
        mse = model.evaluate(X_test, y_test)
        print(f"Root Mean Squared Error: {np.sqrt(mse)}")
        
        y_pred = model.predict(X_test)
        scaled_back_y_pred = self.scaler2.inverse_transform(y_pred.reshape(-1, 1))
        
        self.df[f'Predicted {self.target_variable}'] = scaled_back_y_pred 
        self.df['Difference'] = self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']
        
        self.model = model
        self.history = history
    
    def _fit_pytorch(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        # Build PyTorch model
        class NeuralNet(nn.Module):
            def __init__(self, input_dim, nodes, dropout, bn):
                super(NeuralNet, self).__init__()
                self.fc1 = nn.Linear(input_dim, 16 * nodes)
                self.fc2 = nn.Linear(16 * nodes, 8 * nodes)
                self.fc3 = nn.Linear(8 * nodes, 4 * nodes)
                self.fc4 = nn.Linear(4 * nodes, nodes)
                self.fc5 = nn.Linear(nodes, 1)
                self.bn = bn
                self.dropout = nn.Dropout(dropout)
                self.activation = nn.ReLU()
                if bn:
                    self.bn_layers = nn.ModuleList([nn.BatchNorm1d(16 * nodes), nn.BatchNorm1d(8 * nodes), nn.BatchNorm1d(4 * nodes), nn.BatchNorm1d(nodes)])

            def forward(self, x):
                x = self.fc1(x)
                if self.bn: x = self.bn_layers[0](x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.fc2(x)
                if self.bn: x = self.bn_layers[1](x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.fc3(x)
                if self.bn: x = self.bn_layers[2](x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.fc4(x)
                if self.bn: x = self.bn_layers[3](x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.fc5(x)
                return x

        model = NeuralNet(input_dim=X_train.shape[1], nodes=self.nodes, dropout=self.dropout, bn=self.bn)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

        # Training loop
        model.train()
        for epoch in range(self.n_epoch):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{self.n_epoch}, Loss: {running_loss/len(train_loader)}")
        
        # Validation and testing
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_tensor).squeeze().numpy()
            scaled_back_y_pred = self.scaler2.inverse_transform(test_preds.reshape(-1, 1))

        self.df[f'Predicted {self.target_variable}'] = scaled_back_y_pred
        self.df['Difference'] = self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']
        
        self.model = model
        
            
    def step_decay(self, epoch):
        initial_lr = 0.01
        drop = 0.5
        epochs_drop = 10
        new_lr = initial_lr * (drop ** (epoch // epochs_drop))
        return new_lr
    def predict(self, df):
        """
        Predict the target using the trained deep learning model.
        """
        columns = list(self.columns)
      
        features_nan = df[columns]
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        features = pd.DataFrame(imputer.fit_transform(features_nan), columns=features_nan.columns)

        features = features.reindex(columns=columns)

        normalized_features = self.scaler1.transform(features)
        y_pred = self.model.predict(normalized_features)
        scaled_back_y_pred = self.scaler2.inverse_transform(y_pred.reshape(-1, 1))

        return scaled_back_y_pred
    
    def get_error(self):
            """
            Calculate the Root Mean Squared Error (RMSE) of the predicted elevation values.
        
            Parameters:
            - df (pd.DataFrame): The DataFrame containing the actual and predicted elevation values.
        
            Returns:
            - error (float): The RMSE value.
            """
            if f'Predicted {self.target_variable}' in self.df.columns:
                return np.sqrt(np.mean((self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']) ** 2))
            else:
                self.Fit()
                return np.sqrt(np.mean((self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']) ** 2))
            
    

    def plot_loss_curve(self):
        """
        Plot the loss curve during training.
        """
        if self.history is None:
            print("Model has not been trained yet.")
            return

        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_prediction(self):
        """
        Plot the actual and predicted elevations along with RMSE.
        """
        gdf = gpd.GeoDataFrame(self.df, geometry=[Point(x, y) for x, y in zip(self.df['Longitude'].values, self.df['Latitude'].values)])
        gdf = gdf.set_crs(epsg=4326)
        kwargs = {'vmin': -11000, 'vmax': 6000, 'cmap': 'viridis'}
        plotgdf(gdf, mollweide=True, column='Elevation', cbar=True, **kwargs)
        plotgdf(gdf, mollweide=True, column='Predicted Elevation', cbar=True, **kwargs)
        kwargs = {'cmap': 'cool', 'label': 'RMSE(m)', 'vmin': 0, 'vmax': 1000}
        plotgdf(gdf, mollweide=True, column='RMSE', cbar=True, **kwargs)

    def save_model(self, loc):
        """
        Save DataFrame, trained model, scalers, columns, and training history to specified location.
        """
        try:
            if not os.path.exists(loc):
                os.makedirs(loc)

            self.df.to_csv(f'{loc}/df_data.csv', index=False)
            self.model.save(f'{loc}/deep_time_topo_model.h5')

            with open(f'{loc}/scaler1.pkl', 'wb') as f:
                pickle.dump(self.scaler1, f)
            with open(f'{loc}/scaler2.pkl', 'wb') as f:
                pickle.dump(self.scaler2, f)

            with open(f'{loc}/columns.txt', 'w') as f:
                f.write('\n'.join(self.columns))

            with open(f'{loc}/training_history.pkl', 'wb') as f:
                pickle.dump(self.history.history, f)

            print(f"Saved model components to {loc}")
        except Exception as e:
            print(f"Error saving files: {str(e)}")

    def load_model(self, loc):
        """
        Load DataFrame, trained model, scalers, columns, and training history from specified location.
        """
        try:
            self.df = pd.read_csv(f'{loc}/df_data.csv')
            self.model = load_model(f'{loc}/deep_time_topo_model.h5')

            with open(f'{loc}/scaler1.pkl', 'rb') as f:
                self.scaler1 = pickle.load(f)
            with open(f'{loc}/scaler2.pkl', 'rb') as f:
                self.scaler2 = pickle.load(f)

            with open(f'{loc}/columns.txt', 'r') as f:
                self.columns = f.read().splitlines()

            with open(f'{loc}/training_history.pkl', 'rb') as f:
                history_dict = pickle.load(f)
                self.history = type('', (), {})()
                self.history.history = history_dict

            print(f"Loaded model components from {loc}")
        except FileNotFoundError as fnf_error:
            print(f"File not found error: {str(fnf_error)}")
        except Exception as e:
            print(f"Error loading files: {str(e)}")

    def __repr__(self):
        return f"DeepTimeTopoModel(nodes={self.nodes}, n_epoch={self.n_epoch}, dropout={self.dropout}, bn={self.bn}, lr={self.lr}, scaler={self.scaler})"



        
    

def cumulative_subduction_params(training_data_folder,starttime,endtime,elevation=False,mantle_temperature=False,dynamic_topography=True,Vz=False,precipitation=False,humid_belt=False, lat_band=10):
    times=np.arange(starttime,endtime,1)
    df=pd.read_csv(f"{training_data_folder}/Training_Data_{times[0]}.csv")
    cumulative=df
    if humid_belt:
        cumulative['Humid Belt']=df['Latitude'].apply(lambda x: assign_belt(x, lat_band))

    
    for t in times[1:]:   
        df=pd.read_csv(f"{training_data_folder}/Training_Data_{t}.csv")
        cols=list(df.columns[5:])
        if "geometry" in cols:
            
            cols.remove('geometry')
    
        for col in cols:
            if col=="Latitude":
                pass
            elif col=="Longitude":
                pass
            elif col=="Trench Distance":
                pass
            
            elif col=="Elevation(Scotese)":
                pass

            else:
                cumulative[col]=cumulative[col]+df[col]/15   
        
        if humid_belt:
            cumulative['Humid Belt'] = cumulative['Humid Belt'] + df['Latitude'].apply(lambda x: assign_belt(x, lat_band))

    if mantle_temperature:
        depths=[16,31,47,62,140,155,171, 186,202, 217,233,268,293,323,357,396,439,487,540,597,660]

        for depth in depths:
            coordinates = [(x, y) for x, y in zip(cumulative['Longitude'].values, cumulative['Latitude'].values)]
        
            raster_file=f"/Users/ssin4735/Documents/PROJECT/PhD Project/Codes and Data/DeepTimeTopo/EarthByte_STELLAR_Plate_Motion_Model-Phase2/Mantle Convection/MT/{times[0]}/{depth}.tif"
            mt_data= rasterio.open(raster_file)
            mt = list(mt_data.sample(coordinates))
            mt=[mt[i][0] for i in range(len(mt))]
            cumulative[f"MT_{depth}"]=mt
    
    
    if Vz:
        depths=[16,31,47,62,140,155,171, 186,202, 217,233,268,293,323,357,396,439,487,540,597,660]

        for depth in depths:
            coordinates = [(x, y) for x, y in zip(cumulative['Longitude'].values, cumulative['Latitude'].values)]
       
            raster_file=f"/Users/ssin4735/Documents/PROJECT/PhD Project/Codes and Data/DeepTimeTopo/EarthByte_STELLAR_Plate_Motion_Model-Phase2/Mantle Convection/Vz/{times[0]}/{depth}.tif"
            dt_data= rasterio.open(raster_file)
            dt = list(dt_data.sample(coordinates))
            dt=[dt[i][0] for i in range(len(dt))]
            cumulative[f"Vz_{depth}"]=dt
            
    
    if dynamic_topography:
        
        coordinates = [(x, y) for x, y in zip(cumulative['Longitude'].values, cumulative['Latitude'].values)]

        raster_file=f"/Users/ssin4735/Documents/PROJECT/PhD Project/Codes and Data/Part 2/A_DEEPTIMETOPO/Data/DynamicTopography/waterloaded_dt/Interpolated_STLR1GaM3_MantleFrame_waterloaded_dt-{times[0]}.nc"
        
        dt_data= rasterio.open(raster_file)
        dt = list(dt_data.sample(coordinates))
        dt=[dt[i][0] for i in range(len(dt))]
        cumulative["Dynamic Topography"]=dt

        
            
           
            
    if elevation:
        raster_file = "/Users/ssin4735/Documents/PROJECT/PhD Project/Codes and Data/Part 2/A_DEEPTIMETOPO/Data/ETopo.tif"
        etopo = rasterio.open(raster_file)
        coordinates = [(x, y) for x, y in zip(cumulative['Longitude'].values, cumulative['Latitude'].values)]
        etopo_point = list(etopo.sample(coordinates))
        etopo_point=[etopo_point[i][0] for i in range(len(etopo_point))]
        smoothed_elevation_data=rasterio.open("/Users/ssin4735/Documents/PROJECT/PhD Project/Codes and Data/Part 2/A_DEEPTIMETOPO/Data/Smoothed Etopo.tif")
        # raster_file = "Data/ETopo.tif"
        etopo = rasterio.open(raster_file)
        smoothed_elevation = list(smoothed_elevation_data.sample(coordinates))
        smoothed_elevation=[smoothed_elevation[i][0] for i in range(len(smoothed_elevation))]

        cumulative['Elevation']=etopo_point
        cumulative['Smoothed Elevation']=smoothed_elevation
        
   
    
    return cumulative


class SimpleNN(nn.Module):
    def __init__(self, input_size, nodes, dropout):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16 * nodes)
        self.bn1 = nn.BatchNorm1d(16 * nodes)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(16 * nodes, 8 * nodes)
        self.bn2 = nn.BatchNorm1d(8 * nodes)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(8 * nodes, 1 * nodes)
        self.bn3 = nn.BatchNorm1d(1 * nodes)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc4 = nn.Linear(1 * nodes, nodes)
        self.bn4 = nn.BatchNorm1d(nodes)
        self.dropout4 = nn.Dropout(dropout)
        
        self.fc5 = nn.Linear(nodes, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = torch.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        
        x = torch.relu(self.fc4(x))
        x = self.bn4(x)
        x = self.dropout4(x)
        
        x = self.fc5(x)
        
        return x

class DLModel_torch:
    def __init__(self, training_df, training_variables, target_variable, nodes=8, n_epoch=20, dropout=0.2, bn=True, lr="OnPlateau", scaler="RobustScaler"):
        self.df = training_df.copy()
        self.columns = training_variables
        self.target_variable = target_variable
        self.nodes = max(8, nodes)
        self.n_epoch = n_epoch
        self.dropout = dropout
        self.bn = bn
        self.lr = lr
        self.scaler = scaler
        self.model = None
        self.scaler1 = None
        self.scaler2 = None
   
 
 
    def fit(self):
        # Data preparation
        features_nan = self.df[self.columns].copy()
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        features = pd.DataFrame(imputer.fit_transform(features_nan), columns=features_nan.columns)
        
        if self.scaler == "MinMaxScaler":
            self.scaler1 = MinMaxScaler()
            self.scaler2 = MinMaxScaler()
        elif self.scaler == "RobustScaler":
            self.scaler1 = RobustScaler()
            self.scaler2 = RobustScaler()
        elif self.scaler == "StandardScaler":
            self.scaler1 = StandardScaler()
            self.scaler2 = StandardScaler()
        else:
            self.scaler1 = QuantileTransformer(output_distribution='uniform')
            self.scaler2 = QuantileTransformer(output_distribution='uniform')
        
        print("Creating RFTimeTopo model based on parameters:")
        for col in self.columns:
            print(col)
        
        normalized_features = self.scaler1.fit_transform(features)
        normalized_y = self.scaler2.fit_transform(self.df[self.target_variable].values.reshape(-1, 1))
    
        X_train, X_test, y_train, y_test = train_test_split(normalized_features, normalized_y, test_size=0.7, random_state=22)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=22)
    
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
        self.model = SimpleNN(len(self.columns), self.nodes, self.dropout)
        
        # Compute weights on unnormalized y_train
        y_train_unscaled = self.scaler2.inverse_transform(y_train)
        y_train_flat = y_train_unscaled.ravel()
        weights = np.ones_like(y_train_flat)
        weights[(y_train_flat > -1000) & (y_train_flat <= 0)] = 1
        weights[(y_train_flat > 0) & (y_train_flat <= 300)] = 2
        weights[(y_train_flat > 300) & (y_train_flat <= 3000)] = 4
        weights[y_train_flat > 3000] = 4
        weights = torch.tensor(weights, dtype=torch.float32)
    
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
        self.history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.n_epoch):
            train_loss = 0.0
            val_loss = 0.0
            
            self.model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                
                # Compute weights for the current batch based on unnormalized y_train values
                y_train_flat_batch = labels.squeeze().detach().numpy()
                y_train_flat_batch_unscaled = self.scaler2.inverse_transform(y_train_flat_batch.reshape(-1, 1)).ravel()
    
                # Ensure y_train_flat_batch_unscaled is not empty
                if y_train_flat_batch_unscaled.size == 0:
                    continue
                
                weights_batch = np.ones_like(y_train_flat_batch_unscaled)
                weights_batch[(y_train_flat_batch_unscaled > -1000) & (y_train_flat_batch_unscaled <= 0)] = 1
                weights_batch[(y_train_flat_batch_unscaled > 0) & (y_train_flat_batch_unscaled <= 300)] = 2
                weights_batch[(y_train_flat_batch_unscaled > 300) & (y_train_flat_batch_unscaled <= 3000)] = 4
                weights_batch[y_train_flat_batch_unscaled > 3000] = 4
                weights_batch = torch.tensor(weights_batch, dtype=torch.float32)
    
                # Apply weights to loss
                weighted_loss = (loss * weights_batch).mean()
                weighted_loss.backward()
                optimizer.step()
                train_loss += weighted_loss.item() * inputs.size(0)
            
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs.squeeze(), labels.squeeze())
                    
                    # Compute weights for the validation batch based on unnormalized y_val values
                    y_val_flat_batch = labels.squeeze().detach().numpy()
                    y_val_flat_batch_unscaled = self.scaler2.inverse_transform(y_val_flat_batch.reshape(-1, 1)).ravel()
    
                    # Ensure y_val_flat_batch_unscaled is not empty
                    if y_val_flat_batch_unscaled.size == 0:
                        continue
                    
                    weights_batch = np.ones_like(y_val_flat_batch_unscaled)
                    weights_batch[(y_val_flat_batch_unscaled > -1000) & (y_val_flat_batch_unscaled <= 0)] = 1
                    weights_batch[(y_val_flat_batch_unscaled > 0) & (y_val_flat_batch_unscaled <= 300)] = 2
                    weights_batch[(y_val_flat_batch_unscaled > 300) & (y_val_flat_batch_unscaled <= 3000)] = 4
                    weights_batch[y_val_flat_batch_unscaled > 3000] = 4
                    weights_batch = torch.tensor(weights_batch, dtype=torch.float32)
    
                    # Apply weights to loss
                    weighted_loss = (loss * weights_batch).mean()
                    val_loss += weighted_loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            scheduler.step(val_loss)
        
        # Evaluate on test data
        test_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                
                # Compute weights for the test batch based on unnormalized y_test values
                y_test_flat_batch = labels.squeeze().detach().numpy()
                y_test_flat_batch_unscaled = self.scaler2.inverse_transform(y_test_flat_batch.reshape(-1, 1)).ravel()
    
                # Ensure y_test_flat_batch_unscaled is not empty
                if y_test_flat_batch_unscaled.size == 0:
                    continue
                
                weights_batch = np.ones_like(y_test_flat_batch_unscaled)
                weights_batch[(y_test_flat_batch_unscaled > -1000) & (y_test_flat_batch_unscaled <= 0)] = 1
                weights_batch[(y_test_flat_batch_unscaled > 0) & (y_test_flat_batch_unscaled <= 300)] = 2
                weights_batch[(y_test_flat_batch_unscaled > 300) & (y_test_flat_batch_unscaled <= 3000)] = 4
                weights_batch[y_test_flat_batch_unscaled > 3000] = 4
                weights_batch = torch.tensor(weights_batch, dtype=torch.float32)
    
                # Apply weights to loss
                weighted_loss = (loss * weights_batch).mean()
                test_loss += weighted_loss.item() * inputs.size(0)
        
        test_loss /= len(test_loader.dataset)
        print(f"Root Mean Squared Error: {np.sqrt(test_loss)}")
        
        # Predict on full data
        full_dataset = torch.tensor(normalized_features, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(full_dataset).numpy()
        
        scaled_back_y_pred = self.scaler2.inverse_transform(y_pred.reshape(-1, 1))
        self.df[f'Predicted {self.target_variable}'] = scaled_back_y_pred
        self.df['Difference'] = self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']
    
  
        
 

    
    def predict(self, df):
        columns = list(self.columns)
        features_nan = df[columns]
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        features = pd.DataFrame(imputer.fit_transform(features_nan), columns=features_nan.columns)
        features = features.reindex(columns=columns)
        normalized_features = self.scaler1.transform(features)
        full_dataset = torch.tensor(normalized_features, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(full_dataset).numpy()
        
        scaled_back_y_pred = self.scaler2.inverse_transform(y_pred.reshape(-1, 1))
        return scaled_back_y_pred
    
    def plot_loss_curve(self):
        if self.history is None:
            print("Model has not been trained yet.")
            return
        
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def save_model(self, loc):
        try:
            if not os.path.exists(loc):
                os.makedirs(loc)
            
            self.df.to_csv(f'{loc}/df_data.csv', index=False)
            torch.save(self.model.state_dict(), f'{loc}/deep_time_topo_model.pth')
            
            with open(f'{loc}/scaler1.pkl', 'wb') as f:
                pickle.dump(self.scaler1, f)
            with open(f'{loc}/scaler2.pkl', 'wb') as f:
                pickle.dump(self.scaler2, f)
            
            with open(f'{loc}/columns.txt', 'w') as f:
                f.write('\n'.join(self.columns))
            
            with open(f'{loc}/training_history.pkl', 'wb') as f:
                pickle.dump(self.history, f)
            
            print(f"Saved model components to {loc}")
        except Exception as e:
            print(f"Error saving files: {str(e)}")
    
    def load_model(self, loc):
        try:
            self.df = pd.read_csv(f'{loc}/df_data.csv')
            self.model = SimpleNN(len(self.columns), self.nodes, self.dropout)
            self.model.load_state_dict(torch.load(f'{loc}/deep_time_topo_model.pth'))
            
            with open(f'{loc}/scaler1.pkl', 'rb') as f:
                self.scaler1 = pickle.load(f)
            with open(f'{loc}/scaler2.pkl', 'rb') as f:
                self.scaler2 = pickle.load(f)
            
            with open(f'{loc}/columns.txt', 'r') as f:
                self.columns = f.read().splitlines()
            
            with open(f'{loc}/training_history.pkl', 'rb') as f:
                self.history = pickle.load(f)
            
            print(f"Loaded model components from {loc}")
        except FileNotFoundError as fnf_error:
            print(f"File not found error: {str(fnf_error)}")
        except Exception as e:
            print(f"Error loading files: {str(e)}")
    
    def __repr__(self):
        return f"DeepTimeTopoModel(nodes={self.nodes}, n_epoch={self.n_epoch}, dropout={self.dropout}, bn={self.bn}, lr={self.lr}, scaler={self.scaler})"


from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer
from tensorflow.keras import layers
from tensorflow.keras import models, callbacks, optimizers

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer
from tensorflow.keras import layers, models, callbacks, optimizers

class DLModel_tensorflow:
    def __init__(self, training_df, training_variables, target_variable, nodes=8, n_epoch=20, dropout=0.2, bn=True, lr="OnPlateau", scaler="RobustScaler"):
        self.df = training_df.copy()
        self.columns = training_variables
        self.target_variable = target_variable
        self.nodes = max(8, nodes)
        self.n_epoch = n_epoch
        self.dropout = dropout
        self.bn = bn
        self.lr = lr
        self.scaler = scaler
        self.model = None
        self.scaler1 = None
        self.scaler2 = None
        self.history = None
    
    def fit(self):
        features_nan = self.df[self.columns].copy()
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        features = pd.DataFrame(imputer.fit_transform(features_nan), columns=features_nan.columns)
        
        if self.scaler == "MinMaxScaler":
            self.scaler1 = MinMaxScaler()
            self.scaler2 = MinMaxScaler()
        elif self.scaler == "RobustScaler":
            self.scaler1 = RobustScaler()
            self.scaler2 = RobustScaler()
        elif self.scaler == "StandardScaler":
            self.scaler1 = StandardScaler()
            self.scaler2 = StandardScaler()
        else:
            self.scaler1 = QuantileTransformer(output_distribution='uniform')
            self.scaler2 = QuantileTransformer(output_distribution='uniform')
        print("Creating RFTimeTopo model based on parameters:")
        for col in self.columns:
            print(col)
        normalized_features = self.scaler1.fit_transform(features)
        normalized_y = self.scaler2.fit_transform(self.df[self.target_variable].values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(normalized_features, normalized_y, test_size=0.7, random_state=22)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=22)

        dropout=self.dropout
        activation="relu"
        model = Sequential()
        model.add(Input(shape=(len(self.columns),)))  # Updated input layer
        model.add(Dense(16 * self.nodes))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Activation(activation)) 
        model.add(Dense(8 * self.nodes))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Activation(activation)) 
        model.add(Dense(4 * self.nodes))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Activation(activation)) 
        model.add(Dense(self.nodes))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Activation(activation))
        model.add(Dense(1))
        
        print(model.summary())
        if self.lr == "Step":
            reduce_lr = callbacks.LearningRateScheduler(self.step_decay, verbose=1)
        elif self.lr == "OnPlateau":
            reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        else:
            reduce_lr = None
        
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
        
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam())
        
        # history = model.fit(X_train, y_train, epochs=self.n_epoch, batch_size=32, validation_data=(X_val, y_val),
        #                     callbacks=[reduce_lr, early_stopping], verbose=1)
        history = model.fit(X_train, y_train, epochs=self.n_epoch, batch_size=32, validation_data=(X_val, y_val), verbose=1)
        
        mse = model.evaluate(X_test, y_test)
        print(f"Root Mean Squared Error: {np.sqrt(mse)}")
        
        y_pred = model.predict(normalized_features)
        scaled_back_y_pred = self.scaler2.inverse_transform(y_pred.reshape(-1, 1))
        self.df[f'Predicted {self.target_variable}'] = scaled_back_y_pred 
        self.df['Difference'] = self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']
        
        self.model = model
        self.history = history
        self.scaler1 = self.scaler1
        self.scaler2 = self.scaler2
        self.df = self.df
    
    def step_decay(self, epoch):
        initial_lr = 0.01
        drop = 0.5
        epochs_drop = 10
        new_lr = initial_lr * (drop ** (epoch // epochs_drop))
        return new_lr
    def predict(self, df):
        """
        Predict the target using the trained deep learning model.
        """
        columns = list(self.columns)
      
        features_nan = df[columns]
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        features = pd.DataFrame(imputer.fit_transform(features_nan), columns=features_nan.columns)

        features = features.reindex(columns=columns)

        normalized_features = self.scaler1.transform(features)
        y_pred = self.model.predict(normalized_features)
        scaled_back_y_pred = self.scaler2.inverse_transform(y_pred.reshape(-1, 1))
        

        return scaled_back_y_pred
    
    def get_error(self):
            """
            Calculate the Root Mean Squared Error (RMSE) of the predicted elevation values.
        
            Parameters:
            - df (pd.DataFrame): The DataFrame containing the actual and predicted elevation values.
        
            Returns:
            - error (float): The RMSE value.
            """
            if f'Predicted {self.target_variable}' in self.df.columns:
                return np.sqrt(np.mean((self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']) ** 2))
            else:
                self.Fit()
                return np.sqrt(np.mean((self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']) ** 2))
            
    

    def plot_loss_curve(self):
        """
        Plot the loss curve during training.
        """
        if self.history is None:
            print("Model has not been trained yet.")
            return

        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_prediction(self):
        """
        Plot the actual and predicted elevations along with RMSE.
        """
        gdf = gpd.GeoDataFrame(self.df, geometry=[Point(x, y) for x, y in zip(self.df['Longitude'].values, self.df['Latitude'].values)])
        gdf = gdf.set_crs(epsg=4326)
        kwargs = {'vmin': -11000, 'vmax': 6000, 'cmap': 'viridis'}
        plotgdf(gdf, mollweide=True, column='Elevation', cbar=True, **kwargs)
        plotgdf(gdf, mollweide=True, column='Predicted Elevation', cbar=True, **kwargs)
        kwargs = {'cmap': 'cool', 'label': 'RMSE(m)', 'vmin': 0, 'vmax': 1000}
        plotgdf(gdf, mollweide=True, column='RMSE', cbar=True, **kwargs)

    def save_model(self, loc):
        """
        Save DataFrame, trained model, scalers, columns, and training history to specified location.
        """
        try:
            if not os.path.exists(loc):
                os.makedirs(loc)

            self.df.to_csv(f'{loc}/df_data.csv', index=False)
            self.model.save(f'{loc}/deep_time_topo_model.h5')

            with open(f'{loc}/scaler1.pkl', 'wb') as f:
                pickle.dump(self.scaler1, f)
            with open(f'{loc}/scaler2.pkl', 'wb') as f:
                pickle.dump(self.scaler2, f)

            with open(f'{loc}/columns.txt', 'w') as f:
                f.write('\n'.join(self.columns))

            with open(f'{loc}/training_history.pkl', 'wb') as f:
                pickle.dump(self.history.history, f)

            print(f"Saved model components to {loc}")
        except Exception as e:
            print(f"Error saving files: {str(e)}")

    def load_model(self, loc):
        """
        Load DataFrame, trained model, scalers, columns, and training history from specified location.
        """
        try:
            self.df = pd.read_csv(f'{loc}/df_data.csv')
            self.model = load_model(f'{loc}/deep_time_topo_model.h5')

            with open(f'{loc}/scaler1.pkl', 'rb') as f:
                self.scaler1 = pickle.load(f)
            with open(f'{loc}/scaler2.pkl', 'rb') as f:
                self.scaler2 = pickle.load(f)

            with open(f'{loc}/columns.txt', 'r') as f:
                self.columns = f.read().splitlines()

            with open(f'{loc}/training_history.pkl', 'rb') as f:
                history_dict = pickle.load(f)
                self.history = type('', (), {})()
                self.history.history = history_dict

            print(f"Loaded model components from {loc}")
        except FileNotFoundError as fnf_error:
            print(f"File not found error: {str(fnf_error)}")
        except Exception as e:
            print(f"Error loading files: {str(e)}")

    def __repr__(self):
        return f"DeepTimeTopoModel(nodes={self.nodes}, n_epoch={self.n_epoch}, dropout={self.dropout}, bn={self.bn}, lr={self.lr}, scaler={self.scaler})"


class RFModel:
    def __init__(self, training_df, training_variables, target_variable, n_estimators=100, max_depth=8,min_samples_split=10,min_samples_leaf=5,max_features='sqrt',parallel=-1,random_state=22):
        """
        Initialize the RFTopoModel with the given parameters.
        
        Parameters:
        - training_df (pd.DataFrame): The input DataFrame containing the data.
        - remove_variable (str): The variable/column to be removed from the features.
        - n_estimators (int): The number of trees in the forest.
        - max_depth (int): The maximum depth of the tree.
        """
        self.df = training_df.copy().drop_duplicates()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.max_features=max_features
        self.random_state=random_state
        self.columns = training_variables
        self.target_variable=target_variable
        self.model = None
        self.n_jobs=parallel
        
    def __repr__(self):
        """
        Return a string representation of the RFTopoModel instance.
        """
        return (f"RFTopoModel(n_estimators={self.n_estimators}, "
                f"Features={self.columns}, "
                f"Target Variable={self.target_variable}")

    def __str__(self):
        """
        Return a readable string representation of the RFTopoModel instance.
        """
        return (f"RFTopoModel with {self.n_estimators} trees, "
                f"max depth of {self.max_depth}, "
                f"trained to predict elevation")

    def __len__(self):
        """
        Return the number of rows in the DataFrame.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Get an item (row) from the DataFrame by index.
        """
        return self.df.iloc[index]

    def __setitem__(self, index, value):
        """
        Set an item (row) in the DataFrame by index.
        """
        self.df.iloc[index] = value

    def __delitem__(self, index):
        """
        Delete an item (row) from the DataFrame by index.
        """
        self.df = self.df.drop(index).reset_index(drop=True)

    def fit(self):
        """
        Create and train a Random Forest model using the provided data.
        
        Returns:
        - df (pd.DataFrame): The DataFrame with the predicted elevation and RMSE columns.
        - model (RandomForestRegressor): The trained RandomForest model.
        - columns (list): List of feature columns used in the model.
        """
        features_nan = self.df[self.columns].copy()
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        # imputer = SimpleImputer(strategy='mean')
        features = pd.DataFrame(imputer.fit_transform(features_nan), columns=features_nan.columns)

        print("Creating RFTimeTopo model based on parameters:")
        for col in self.columns:
            print(col)

        y = self.df[self.target_variable].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=22)
        X_val, X_tes, y_val, y_tes = train_test_split(X_test, y_test, test_size=0.5, random_state=22)

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        # weights = y_train / np.max(y_train)
        # Initialize the weight array with default weight 1
      # Flatten y_train to ensure it's a 1-dimensional array
        y_train_flat = y_train.ravel()
        
        # Initialize the weight array with default weight 1
        weights = np.ones_like(y_train_flat)
        
        # Apply the conditions to update the weights
        weights[(y_train_flat > 300) & (y_train_flat <= 1000)] = 2
        weights[(y_train_flat > 1000) & (y_train_flat <= 3000)] = 3
        weights[y_train_flat > 3000] = 4

        self.model.fit(X_train, y_train.ravel(),sample_weight=weights)

        y_pred = self.model.predict(X_tes)
        mse = mean_squared_error(y_tes, y_pred)
        print(f"Mean Square Error {np.sqrt(mse)}")

        y_pred_full = self.model.predict(features)
        self.df[f'Predicted {self.target_variable}'] = y_pred_full
        self.df['Difference'] = self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']

        # return self.df, self.model, self.columns

    def predict(self, df):
        """
        Predict elevation using the trained RandomForest model.
        
        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the features for prediction.
        
        Returns:
        - y_pred (np.array): The predicted elevation values.
        """
     

        features_nan = df[self.columns]
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        features = pd.DataFrame(imputer.fit_transform(features_nan), columns=features_nan.columns)
        features = features.reindex(columns=self.columns)
        
        y_pred = self.model.predict(features)
        return y_pred

    def plot_difference(self, plate_model,kwargs1,kwargs2,quick=True):
        """
        Plot the actual and predicted elevation values on a map.
        
        
        """
        df=self.df
        gdf = gpd.GeoDataFrame(df, geometry=[Point(x, y) for x, y in zip(df['Longitude'].values, df['Latitude'].values)])
        gdf = gdf.set_crs(epsg=4326)
        kwargs = kwargs1
        plotgdf(gdf,plate_model, mollweide=True, column=self.target_variable, cbar=True,quick=quick, **kwargs)
        plotgdf(gdf,plate_model, mollweide=True, column=f'Predicted {self.target_variable}', cbar=True, **kwargs)
        kwargs = kwargs2
        plotgdf(gdf,plate_model, mollweide=True, column='Difference', cbar=True,quick=quick, **kwargs)
    
    
    def plot_feature_importance(self):
           """
           Plot the feature importance of the trained Random Forest model.
           """
           if self.model is None:
               print("Model has not been trained yet.")
               return
        
           importances = self.model.feature_importances_
           indices = np.argsort(importances)[::-1]

           plt.figure(figsize=(12, 6))
           plt.title("Feature Importances")
           plt.bar(range(len(importances)), importances[indices], align="center")
           plt.xticks(range(len(importances)), [self.columns[i] for i in indices], rotation=90)
           plt.xlabel("Feature")
           plt.ylabel("Importance")
           plt.tight_layout()
           plt.show()
    
    def get_error(self):
        """
        Calculate the Root Mean Squared Error (RMSE) of the predicted elevation values.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame containing the actual and predicted elevation values.
        
        Returns:
        - error (float): The RMSE value.
        """
        if f'Predicted {self.target_variable}' in self.df.columns:
            return np.mean(np.sqrt((self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']) ** 2))
        else:
            self.Fit()
            return np.mean(np.sqrt((self.df[self.target_variable] - self.df[f'Predicted {self.target_variable}']) ** 2))
            

    def save_model(self, loc):
        """
        Save the DataFrame, trained RandomForest model, and columns to the specified location.
        
        Parameters:
        - loc (str): Directory location to save the files.
        """
        try:

        
            # Save df to CSV
            self.df.to_csv(f'{loc}/df_data.csv', index=False)

            # Save model using pickle
            with open(f'{loc}/random_forest_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)

            # Save columns to text file
            with open(f'{loc}/columns.txt', 'w') as f:
                f.write('\n'.join(self.columns))

            with open(f'{loc}/random_forest_class.pkl', 'wb') as f:
                pickle.dump(self, f)
                
            print(f"Saved DataFrame, model, and columns to {loc}")
            
        except Exception as e:
            print(f"Error saving files: {str(e)}")

    @staticmethod
    def load_model(loc):
        """
        Load the DataFrame, trained RandomForest model, and columns from the specified location.
        
        Parameters:
        - loc (str): Directory location where files are saved.
        
        Returns:
        - df (pd.DataFrame): The loaded DataFrame.
        - model (RandomForestRegressor): The loaded RandomForest model.
        - columns (list): List of column names used in the model.
        """
        df = None
        model = None
        columns = None

        try:
            # Read DataFrame from CSV
            df = pd.read_csv(f'{loc}/df_data.csv')

            # Read trained model from pickle
            # with open(f'{loc}/random_forest_class.pkl', 'rb') as f:
#                 RFMain = pickle.load(f)


            with open(f'{loc}/random_forest_model.pkl', 'rb') as f:
                model = pickle.load(f)

            # Read columns from text file
            with open(f'{loc}/columns.txt', 'r') as f:
                columns = f.read().splitlines()

            print(f"Loaded DataFrame, model, and columns from {loc}")
        except FileNotFoundError as fnf_error:
            print(f"File not found error: {str(fnf_error)}")
        except Exception as e:
            print(f"Error loading files: {str(e)}")

        return df, model,columns
        
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.special import expit
import pandas as pd
from collections import Counter

# Generate a regression dataset (replace this with your actual data)
X, y = make_regression(n_samples=100, n_features=4, noise=0.2)

# Simulate some negative values for demonstration purposes
y = y - 50

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Different weighting strategies
def calculate_weights(strategy, y_train):
    if strategy == 'exponential':
        return np.exp(np.abs(y_train) / np.max(np.abs(y_train)))
    elif strategy == 'logarithmic':
        return np.log1p(np.abs(y_train) / np.max(np.abs(y_train)))
    elif strategy == 'quantiles':
        quantiles = pd.qcut(y_train, q=4, labels=False)
        return quantiles + 1
    elif strategy == 'sigmoid':
        return expit(np.abs(y_train) / np.max(np.abs(y_train)))
    elif strategy == 'inverse_frequency':
        freq = Counter(y_train)
        weights = np.array([1.0 / freq[val] for val in y_train])
        return weights / np.max(weights)  # Normalize weights
    elif strategy == 'distance_from_mean':
        mean_y = np.mean(y_train)
        return np.abs(y_train - mean_y) / np.max(np.abs(y_train - mean_y))
    else:
        return np.ones_like(y_train)





