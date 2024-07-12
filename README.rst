Credit Pipeline
===============

A general pipeline for Trustworthy Machine Learning on credit datasets,
including functionalities for fairness, reject inference, and
explainability. The package facilitates the development of more ethical
algorithms at every step of the ML pipeline for credit modelling.

Install
-------

-  Clone the repository:

``git clone https://github.com/hiaac-finance/credit_pipeline.git``

-  Then, at the root folder of the repository, run the command to
   install the package and its dependencies:

``pip install .``

- Download the data from the following link and place it in the folder ``data/``:

https://drive.google.com/file/d/1Y7bTNsxDv-te40FnJsoca1YeB4da6TCq/view?usp=sharing

To preprocess the data use the following command at the root folder of the repository:

``python -m credit_pipeline.data``

Usage
-----

The package is distributed with a Dockerfile that can be used to run the code in a container, using the same environment as the one used for development. The Dockerfile is located at the root of the repository, including commands to build and run the container. (Note: it is recommended to have prior knowledge of Docker to use this feature.)

One can also use the packaged in a conda environment with installed dependencies. 

Examples of using the diverse functionalities package are located in Jupyter Notebooks at ``examples/usage.ipynb`` (WIP).

Contact
-------

Dr. Marcos Medeiros Raimundo, project coordinator:
mraimundo@ic.unicamp.br