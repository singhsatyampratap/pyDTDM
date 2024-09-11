# pyDTDM Input Files Configuration for Paleotopgraphy

All input files are shared through STELLAR sharepoints.

This repository uses a YAML input file to configure various files required for running machine learning tasks on spatiotemporal geological data. Below is a summary of the input files required by the project:

## 1. Plate Kinematics
This section defines the plate model and related files necessary for plate kinematic calculations:
- **model_name**: Name of the plate model (e.g., `'phase2NNR'`).
- **model_dir**: Directory where the model files are stored.(e.g., `'Flat_Slabs.gpml'`,`'Plate_Boundaries.gpml'`)
- **rotation_files**: List of plate rotation files (e.g., `'CombinedRotations.rot'`).
- **topology_files**: List of GPlates topology files
- **agegrid**: File path to the seafloor age grid.
- **coastline_file**: Shapefile for global coastlines.
- **static_polygon**: Shapefile for present-day static plate polygons.
- **continents**: Shapefile for global continental polygons.

## 2. Mantle Convection
This section manages the time-dependent raster data for mantle convection (netcdf):
- **depth**: List of depth levels in kilometers.
- **original_temp_folder**: Directory for original temperature raster files.
- **original_vel_folder**: Directory for original vertical velocity (Vz) files.
- **original_velx_folder**: Directory for original horizontal velocity (Vx) files.
- **original_vely_folder**: Directory for original horizontal velocity (Vy) files.
- **original_visc_folder**: Directory for original viscosity files.
- **new_temp_folder**, **new_vel_folder**, **new_velx_folder**, **new_vely_folder**, **new_visc_folder**: Directories for interpolated mantle convection data.
- **model_name**: Name of the convection model (e.g., `'phase2'`).
- **model_dir**: Directory where the model files are stored.
- **rotation_files**: List of rotation files.
- **topology_files**: List of mantle convection topology files, such as:
  - `'Flat_Slabs.gpml'`
  - `'Plate_Boundaries.gpml'`

## 3. Precipitation
This section defines the precipitation data and related files:
- **precipitation_folder**: Directory containing precipitation data.
- **model_name**: Precipitation model name (e.g., `'PALEOMAP'`).
- **model_dir**: Directory for the precipitation model.
- **rotation_files**: List of rotation files for the precipitation model.
- **topology_files**: Topology files for precipitation, such as:
  - `'PALEOMAP_PlatePolygons.gpml'`
  - `'PALEOMAP_PoliticalBoundaries.gpml'`

## 4. Training Files (Raster Data)
This section defines the raster files used for training:
- **ETOPO_FILE**: Path to the ETOPO raster file, which represents the Earth's topography (can be NetCDF or GeoTiff).
- **Raster_type**: Type of raster data (e.g., `"Elevation"`).

## 5. Output Directories
- **output_dir**: Directory where the output files will be stored.

## 6. Time Parameters
- **time_min**: Start time of the simulation (e.g., `0` million years).
- **time_max**: End time of the simulation (e.g., `10` million years).
- **time_step**: Time increment for the simulation.

## 7. Other Parameters
- **time_window_size**: Size of the time window for calculating the mean.
- **weighted_mean**: Whether to use a weighted mean (recent times have greater weights).
- **mesh_refinement_level**: Controls the initial positions of crustal points (higher refinement levels increase precision).
- **mantle_optimised_id**: Reference frame for mantle optimization.
- **paleomag_id**: ID for paleomagnetic reference.
- **number_of_cpus**: Number of CPU cores to use for multiprocessing (set `-1` for all cores).

## 8. Grid Parameters
- **grid_spacing**: Resolution of the output NetCDF files (e.g., `0.1` degrees).
- **compression**: Compression settings for NetCDF files:
  - `zlib`: Enable compression.
  - `complevel`: Compression level (e.g., `5`).
