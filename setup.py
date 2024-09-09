from setuptools import setup, find_packages

setup(
    name='pyDTDM', 
    version='0.1.0',          
    author='Satyam Pratap Singh',
    author_email='singhsatyampratap@gmail.com',
    description='A library ',
    long_description=open('README.md').read(),  # Load your README file
    long_description_content_type='text/markdown',  # Content type for the README (e.g., markdown)
    url='https://github.com/singhsatyampratap/pyDTDM',  # GitHub repo link
    packages=find_packages(),  # Automatically find all packages
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Minimum Python version
    install_requires=[
        'scipy',
        'joblib',              # Utilities
        'pandas',
        'geopandas',
        'gplately',
        'PlateTectonicTools',
        'numpy',
        'matplotlib',          # Visualization
        'seaborn',
        'cartopy',
        'cmcrameri'
,        'stripy',              # Geospatial libraries
        'shapely',
        'xarray',
        'rasterio',
        'geopy',
        'scikit-learn',        # Machine Learning
        'tensorflow',
        'torch',
        'keras',
        'interpret',
        'pickle5',
        'pyarrow',
        'parquet',
        'jupyterlab',
        'adjustText'             
    ],
    test_suite='tests',  # Where your tests are located
)
