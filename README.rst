Credit Pipeline
===============

A general pipeline for Trustworthy Machine Learning on credit datasets,
including functionalities for fairness, reject inference, and
explainability. The package facilitates the development of more ethical
algorithms at all the steps of the machine learning pipeline.

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

Examples of using the diverse functionalities package are located in
Jupyter Notebooks at ``examples/``. The necessary data is also presented in the folder ``data/``.
 We now briefly describe the functions.

Data Exploration
~~~~~~~~~~~~~~~~

Exemplified at the file ``examples/data_exploration.ipynb``, a diverse set of functions is present that is able to deal with
missing values, identify categorical and numerical features, and more sophisticated tasks such as identifying outliers, looking
for correlation, calculating mutual information.

Credit Models
~~~~~~~~~~~~~

Considering three open credit scoring datasets (Home Credit, Taiwan, and German), we provide documentation of our experiments
using Logistic Regression, Neural Networks, Random Forest, and Gradient Boosting. Each experiment is located in Python notebooks
within the folder ``examples/`` with the name of the dataset.

Fairness
~~~~~~~~

Our code presents a comparison of a pre-processing, three in-processing, and one post-processing fairness metric.
The mitigation techniques are performed with all three datasets and inside each of the notebooks in ``examples/`` folder.

Reject Inference
~~~~~~~~~~~~~~~~

Our code also presents a comparison of different reject inference techniques, with the approach of augmentation, extrapolation
, and label spreading. These experiments are present in the notebook ``examples/reject_inference.ipynb`` only with the HomeCredit dataset.
This was necessary as a special data treatment was needed to simulate a scenario of reject inference.

Explainability
~~~~~~~~~~~~~~

Explainability experiments were performed using the German dataset and are localized in the notebook ``examples/german.ipynb``.
Global and local interpretability techniques were used to demonstrate the advantages and pitfalls of each one.


Contact
-------

Dr. Marcos Medeiros Raimundo, project coordinator:
mraimundo@ic.unicamp.br