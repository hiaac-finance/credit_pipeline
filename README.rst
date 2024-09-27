Credit Pipeline
===============

A general pipeline for Trustworthy Machine Learning on credit datasets,
including functionalities for fairness, reject inference, and
explainability. The package facilitates the development of more ethical
algorithms at every step of the ML pipeline for credit modelling.
Supplementary material for the paper "Best Practices for Responsible Machine Learning in Credit Scoring".

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

Another option is to use install the package within a Docker container. 
The file `Dockerfile` contains the specification of the container and the commands to build and run it.
 (Note: it is recommended to have prior knowledge of Docker to use this feature.)

Usage
-----

Examples of using the diverse functionalities package are located in Jupyter Notebooks at ``examples/usage.ipynb``.


Experiments
-----

Experiments code is present at ``scripts/experiments.py``. To run, use the following command at the path ``scripts/``:

``python experiments.py --experiment credit_models --dataset german --seed 0 --n_trials 100 --n_jobs 1``

You can select the experiment between ["credit_models", "fairness"] and dataset between ["german", "taiwan", "homecredit"]. Code to generate the tables is present in ``examples/analysis_results.ipynb``.

Experiments of reject inference are present in ``examples/reject_inference.ipynb`` and for explainability in ``examples/explainability.ipynb``.

Contact
-------

Dr. Marcos Medeiros Raimundo, project coordinator:
mraimundo@ic.unicamp.br