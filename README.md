# mle-training
Private repo for Tiger Analytics MLE training.

# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
* Steps:
1. Clone the repo in your system using the command `git clone git@github.com:tigersatish/mle-training.git` .
2. Go to the project rppt directory and execute: `conda env create -f env.yml` to create the conda environment.
3. After this, run the commans `pip install .` to build and install the house_price_prediction_satish package in your system.
4. After this is done, run the command `python driver.py` to execute the functions of ingest_data, train and score in the packages.
5. Model pickle files are present in `~/mle-training/artifacts/`
   Input and output data are present in `~/mle-training/datasets/housing/`
   Logs are present in `~/mle-training/logs`
6. Execute `pytest` command to run the test cases.
7. To generate sphinx documentation, go to `docs` folder and execute `sphinx-apidoc -f -o . ..//src//house_price_prediction_satish` to generate the rst files. After this execute `make html` commande to generate the html files containing the edocumentation from the scripts. These html files are present inside the `docs/_build/html` folder.

