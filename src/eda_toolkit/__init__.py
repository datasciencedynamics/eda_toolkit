# Import package modules
from .plots import *
from .data_manager import *
from .art import *

# Detailed Documentation
detailed_doc = """
Welcome to EDA Toolkit, a collection of utility functions designed to streamline 
your exploratory data analysis (EDA) tasks. This repository offers tools for 
directory management, some data preprocessing, reporting, visualizations, and 
more, helping you efficiently handle various aspects of data manipulation and 
analysis.

PyPI: https://pypi.org/project/eda-toolkit/
Documentation: https://datasciencedynamics.com/eda_toolkit_docs/

Authors: Leonid Shpaner, Oscar Gil

Acknowledgements

We would like to express our deepest gratitude to Dr. Ebrahim Tarshizi of the 
Shiley-Marcos School of Engineering at the University of San Diego for his 
mentorship in the M.S. in Applied Data Science Program. His unwavering dedication 
and guidance played a pivotal role in our academic journey, supporting our 
successful completion of the program and our pursuit of careers as data scientists.

We thank Robert Lanzafame, PhD, for his feedback, encouragement, and thoughtful 
discussion following our presentation at JupyterCon, and Panayiotis Petousis, PhD, 
and Arthur Funnell from the CTSI UCLA Health data science team for their helpful 
comments, constructive feedback, and continued encouragement throughout the 
development of this library.

Finally, Leon Shpaner would like to personally acknowledge his mentor, former 
manager, and friend, Gustavo Prado, who hired him at the Los Angeles Film School. 
Gustavo believed in him early on, gave him the opportunity to grow, and was patient 
as he developed professionally. He saw potential before it was fully formed and 
sparked an early interest in data by demonstrating the importance of tools like 
VLOOKUP. His guidance and trust had a lasting impact. May he rest in peace.

Version: 0.0.30
"""

# Assign only the detailed documentation to __doc__
__doc__ = detailed_doc

# Metadata
__version__ = "0.0.30"
__author__ = "Leonid Shpaner, Oscar Gil"
__email__ = "lshpaner@ucla.edu; info@oscargildata.com"

def about():
    """Print the eda_toolkit logo, package overview, and version."""
    print(eda_toolkit_logo)
    print(detailed_doc)
