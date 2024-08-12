MxIFPublic
================
Repository contains all original code we used to process MxIF data as described in ["Multi-omic profiling of follicular lymphoma reveals changes in tissue architecture and enhanced stromal remodeling in high-risk patients"](https://doi.org/10.1016/j.ccell.2024.02.001).

Installation
------------
To install the package, it is recommended to use a virtual environment with Python 3.8, as newer versions may produce errors. Once the virtual environment is set up, navigate to the root directory of Cell_Atlas_MxIF and simply run the following command: ``pip install .``

After successfully installing the package, navigate to the notebooks folder to explore the example notebooks. For tessellation analysis, open the ``tessellation_example.ipynb`` notebook. The data required for these examples is available in the IDR repository under the accession number idr0158.

Requirements
------------

Python 3.8+.

Dependencies
------------

Dependencies are defined in:

- ``requirements.in``

- ``requirements.txt``

- ``dev-requirements.in``

- ``dev-requirements.txt``

For community generation PyG was used. Please install it as specified here https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html.
