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
    python_requires='>=3.6',  # Minimum Python version
    install_requires=[
        'pygplates',           # Core libraries
        'pandas',
        'geopandas',
        'gplately',
        'numpy',
        'matplotlib',          # Visualization
        'seaborn',
        'cartopy',
        'glob2',               # glob is part of the Python standard library; glob2 for backward compatibility
        'os',                  # os is part of Python’s standard library (no need to install, just a note)
        'stripy',              # Geospatial libraries
        'shapely',
        'xarray',
        'rasterio',
        'scikit-learn',        # Machine Learning
        'tensorflow',
        'keras',
        'joblib',              # Utilities
        'scipy',
        'warnings',            # warnings is part of Python’s standard library
        'pickle5',
        'pyarrow',
        'parquet'              # pickle is part of the standard library, but you may use `pickle5` for compatibility
    ],
    test_suite='tests',  # Where your tests are located
)
