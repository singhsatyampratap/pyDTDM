# pyDTDM
pyDTDM: A Python library for deep time data mining, offering efficient tools to analyze spatiotemporal raster, vector and gpml datasets. 



## Installation

Follow these steps to set up the environment and install the necessary dependencies:

1. **Create a new conda environment:**

    ```bash
    conda create --name pyDTDM python=3.11
    ```

2. **Activate the environment:**

    ```bash
    conda activate pyDTDM
    ```

3. **Install pyGPlates from conda-forge:**

    ```bash
    conda install -c conda-forge pygplates
    ```

4. **Install additional dependencies using `pip`:**

    ```bash
    pip install . -e
    ```

## Usage

After the installation, the environment is set up, and you can start working with pyGPlates and the other dependencies.

To activate the environment in the future:

```bash
conda activate pyDTDM
```



## Input Configuration

The workflow parameters can be easily configured through the input YAML file.

### Required Input Files and Folders

To run the workflow, you'll need specific input rasters and files organized into a main folders:

1. **Plate Reconstruction Model:**
   
   This folder contains all the necessary rotation files and topology files needed to run the model. 
   

2. **Input Rasters:**
   
   This folder stores the raster files required for the workflow, which supports netCDF and GeoTIFF formats. Ensure the presence of appropriate input raster grids such as:
   
   - **ETOPO Grid**


3 **Geodynamics Rasters:** 
   
   CitcomS output as time dependent netCDFs.
   
   
Note: More details on how to configure input files in [`ConfigFile_Instruction.md`](/InputFiles/ConfigFile_Instruction.md)
   


### Running the Workflow

Once the required files are in place and the YAML configuration is set up, you can proceed to run the workflow with the provided inputs.

To start Jupyter-Lab:
```bash
jupyter-lab
```


