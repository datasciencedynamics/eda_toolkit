# Import package modules
from .plots import *
from .data_manager import *
from .art import print_art, about
from .art import eda_toolkit_logo as _eda_toolkit_logo

# Detailed Documentation
_detailed_doc = """
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

Leon Shpaner also wishes to acknowledge Gustavo Prado, his former manager,
mentor, and friend at the Los Angeles Film School, who hired him early in his
career, gave him room to grow, and first showed him what careful work with data
could do. His trust and guidance left a lasting mark. May he rest in peace.

Version: 0.1.0
"""

# Assign only the detailed documentation to __doc__
__doc__ = _eda_toolkit_logo + _detailed_doc

# Metadata
__version__ = "0.1.0"
__author__ = "Leonid Shpaner, Oscar Gil"
__email__ = "lshpaner@ucla.edu; info@oscargildata.com"

# Keep the package namespace to eda_toolkit's own functions, classes and
# submodules. Star-imports above also pull in third-party names (pandas,
# typing aliases, flags) that would otherwise clutter help(eda_toolkit).
import types as _types

for _name, _obj in list(globals().items()):
    if _name.startswith("_") or isinstance(_obj, _types.ModuleType):
        continue
    if callable(_obj) and getattr(_obj, "__module__", "").startswith("eda_toolkit"):
        continue
    del globals()[_name]

del _name, _obj, _types
