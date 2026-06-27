from setuptools import setup, find_packages

setup(
    name="eda_toolkit",
    version="0.0.30",
    author="Leonid Shpaner, Oscar Gil",
    author_email="lshpaner@ucla.edu",
    description="A Python library for EDA, including visualizations, directory management, data preprocessing, reporting, and more.",
    long_description=open("README_min.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",  # Type of the long description
    package_dir={"": "src"},  # Directory where package files are located
    # Automatically find packages in the specified directory
    packages=find_packages(where="src"),
    project_urls={  # Optional
        "Leon Shpaner's Website": "https://www.leonshpaner.com",
        "Oscar Gil's Website": "https://www.oscargildata.com",
        "Documentation": "https://datasciencedynamics.com/eda_toolkit_docs/",
        "Zenodo Archive": "https://zenodo.org/records/13162633",
        "Source Code": "https://github.com/lshpaner/eda_toolkit/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],  # Classifiers for the package
    python_requires=">=3.8",
    install_requires=[
        "jinja2>=3.0.0",
        "matplotlib>=3.5.3,<=3.9.2",
        "nbformat>=4.2.0,<=5.10.4",
        "numpy>=1.21.6,<=2.1.2",
        "pandas>=1.3.5,<=2.2.3",
        "plotly>=5.18.0,<=5.24.1",
        "scikit-learn>=1.0.2,<=1.2.2",
        "scipy>=1.7.3",
        "seaborn>=0.12.2,<=0.13.2",
        "tqdm>=4.66.4,<=4.67.1",
        "xlsxwriter==3.2.0",
    ],
)
