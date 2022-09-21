import argparse
import os
import pickle
import mlflow

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_INPUT_PATH = "datasets//housing"
MODEL_INPUT_PATH = "artifacts"


def get_prepared_data(data_input_path=DATA_INPUT_PATH):
    """Function to load prepared data from a path.
    Parameters
    ----------
            data_input_path:
                    path to read the prepared data from.

    Returns
    -------
            housing_prepared:
                    features of test data
            housing_labels:
                    target column of test data
            X_test_prepared:
                    features of test data for random forest
            y_test:
                    target column of test data for random forest
    """
    mlflow.log_param("data_input_path", data_input_path)

    housing_prepared = pd.read_csv(data_input_path + "//housing_prepared.csv")
    housing_labels = pd.read_csv(data_input_path + "//housing_labels.csv")
    X_test_prepared = pd.read_csv(data_input_path + "//X_test_prepared.csv")
    y_test = pd.read_csv(data_input_path + "//y_test.csv")

    return (housing_prepared, housing_labels, X_test_prepared, y_test)


def score_lin_reg(model_input_path=MODEL_INPUT_PATH):
    """Function to score model in linear regression.
    Parameters
    ----------
            model_input_path:
                    path to read the pickle file from.

    Returns
    -------
            lin_rmse:
                rmse of prediction from the test data
    """
    mlflow.log_param("model_input_path", model_input_path)

    (
        housing_prepared,
        housing_labels,
        X_test_prepared,
        y_test,
    ) = get_prepared_data()
    lin_reg = pickle.load(open(model_input_path + "//lin_reg.pkl", "rb"))

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print("Linear regression rmse: ", lin_rmse)

    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    print("Linear regression mae: ", lin_mae, "\n")
    mlflow.log_metric(key="Linear regression rmse", value=lin_rmse)
    mlflow.log_metric(key="Linear regression mae", value=lin_mae)

    return lin_rmse


def score_tree_reg(model_input_path=MODEL_INPUT_PATH):
    """Function to score model in decision tree regression.
    Parameters
    ----------
            model_input_path:
                    path to read the pickle file from.

    Returns
    -------
            tree_rmse:
                 rmse of prediction from the test data
    """
    (
        housing_prepared,
        housing_labels,
        X_test_prepared,
        y_test,
    ) = get_prepared_data()
    tree_reg = pickle.load(open(model_input_path + "//tree_reg.pkl", "rb"))
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print("Decision tree rmse: ", tree_rmse, "\n")
    mlflow.log_metric(key="Decision tree rmse", value=tree_rmse)

    return tree_rmse


def score_randfor_reg(model_input_path=MODEL_INPUT_PATH):
    """Function to score model in random forest regression.
    Parameters
    ----------
            model_input_path:
                    path to read the pickle file from.

    Returns
    -------
            final_rmse:
                rmse of prediction from the test data
    """
    (
        housing_prepared,
        housing_labels,
        X_test_prepared,
        y_test,
    ) = get_prepared_data()
    print("Random forest results:")
    rnd_search = pickle.load(open(model_input_path + "//rnd_search.pkl", "rb"))
    cvres = rnd_search.cv_results_

    print("Random search mean_test_score and params:")

    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    grid_search = pickle.load(
        open(model_input_path + "//grid_search.pkl", "rb")
    )

    print("Grid search mean_test_score and params:")
    grid_search.best_params_
    cvres = grid_search.cv_results_

    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    final_model = pickle.load(
        open(model_input_path + "//rand_for_model.pkl", "rb")
    )
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("Final model rmse: ", final_rmse)
    mlflow.log_metric(key="Random forest rmse", value=final_rmse)

    return final_rmse


def main():
    """ Main function that calls other functions in order.
    Parameters
    ----------
    Returns
    -------
    """

    score_lin_reg()
    score_tree_reg()
    score_randfor_reg()


if __name__ == "__main__":
    """Driver function that has argument parser and calls main function.
    Parameters
    ----------
    Returns
    -------
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dif",
        "--data_input_folder",
        help="Input folder path for prepared data:",
    )

    parser.add_argument(
        "-mif",
        "--model_input_folder",
        help="Input folder path for model pickle:",
    )

    args = parser.parse_args()

    if args.data_input_folder:
        DATA_INPUT_PATH = args.datafolder
    else:
        DATA_INPUT_PATH = "datasets//housing"

    if args.model_input_folder:
        MODEL_INPUT_PATH = args.model_input_folder
    else:
        MODEL_INPUT_PATH = "artifacts"

    main()
