from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="pyDTDM",
    version="0.1.0",
    author="Satyam Pratap Singh",
    author_email="singhsatyampratap@gmail.com",
    description="A library for Deep Time Digital Earth and Explainable Boosting Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/singhsatyampratap/pyDTDM",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "scipy",
        "joblib",
        "pandas",
        "geopandas",
        "gplately",
        "PlateTectonicTools",
        "numpy",
        "matplotlib",
        "seaborn",
        "cartopy",
        "cmcrameri",
        # "stripy",
        "shapely",
        "xarray",
        "rasterio",
        "geopy",
        "scikit-learn",
        # "tensorflow",
        # "torch",
        # "keras",
        "interpret",
        # "pickle5",
        "pyarrow",
        "parquet",
        "jupyterlab",
        "adjustText",
        "pygmt"
    ],
    include_package_data=True,
    package_data={
        "pyDTDM": ["cpt/*.cpt"],  # Include CPT color files
    },
    test_suite="tests",
)
