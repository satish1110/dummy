import argparse
import os
import pickle

import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
import mlflow


# read housing, train, test data


MODEL_OUTPUT_PATH = "artifacts"
DATA_INPUT_PATH = "datasets//housing"


def get_train_data(data_input_path=DATA_INPUT_PATH):
    """Function to load train data from a path.
    Parameters
    ----------
            data_input_path:
                    path to read the train data from.

    Returns
    -------
            housing_prepared:
                    features of train data
            housing_labels:
                    target column of train data
    """
    mlflow.log_param("data_input_path", data_input_path)

    housing_prepared = pd.read_csv(data_input_path + "//housing_prepared.csv")
    housing_labels = pd.read_csv(data_input_path + "//housing_labels.csv")

    return (housing_prepared, housing_labels)


def train_lin_reg(model_output_path=MODEL_OUTPUT_PATH):
    """Function to train model in linear regression and write pickle file to a path.
    Parameters
    ----------
            model_output_path:
                    path to write the pickle file to.

    Returns
    -------
    """
    mlflow.log_param("model_output_path", model_output_path)

    housing_prepared, housing_labels = get_train_data()
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    pickle.dump(lin_reg, open(model_output_path + "//lin_reg.pkl", "wb"))
    mlflow.log_artifact(model_output_path + "//lin_reg.pkl")
    mlflow.sklearn.log_model(lin_reg, "Linear regression model")


def train_tree_reg(model_output_path=MODEL_OUTPUT_PATH):
    """Function to train model in decision tree and write pickle file to a path.
    Parameters
    ----------
            model_output_path:
                    path to write the pickle file to.

    Returns
    -------
    """
    housing_prepared, housing_labels = get_train_data()
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    pickle.dump(tree_reg, open(model_output_path + "//tree_reg.pkl", "wb"))
    mlflow.log_artifact(model_output_path + "//tree_reg.pkl")
    mlflow.sklearn.log_model(tree_reg, "Decision tree model")


def train_randfor_reg(model_output_path=MODEL_OUTPUT_PATH):
    """Function to train model in random forest using random search and grid search and
        write pickle file to a path.
    Parameters
    ----------
            model_output_path:
                    path to write the pickle file to.

    Returns
    -------
    """
    if not (os.path.exists(model_output_path)):
        os.makedirs(model_output_path)

    housing_prepared, housing_labels = get_train_data()

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )

    rnd_search.fit(housing_prepared, housing_labels)
    pickle.dump(rnd_search, open(model_output_path + "//rnd_search.pkl", "wb"))
    mlflow.log_artifact(model_output_path + "//rnd_search.pkl")
    mlflow.sklearn.log_model(rnd_search, "Random forest random search model")

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )

    grid_search.fit(housing_prepared, housing_labels)

    pickle.dump(
        grid_search, open(model_output_path + "//grid_search.pkl", "wb")
    )

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    final_model = grid_search.best_estimator_
    print("Final random forest model: ", final_model)
    pickle.dump(
        final_model, open(model_output_path + "//rand_for_model.pkl", "wb")
    )
    mlflow.log_artifact(model_output_path + "//rand_for_model.pkl")
    mlflow.sklearn.log_model(final_model, "Random forest grid search model")


def main():
    """ Main function that calls other functions in order.
    Parameters
    ----------
    Returns
    -------
    """

    train_lin_reg()
    train_tree_reg()
    train_randfor_reg()


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
        "-mof",
        "--model_output_folder",
        help="Output folder path for model pickle:",
    )

    args = parser.parse_args()

    DATA_INPUT_PATH = None

    if args.data_input_folder:
        DATA_INPUT_PATH = args.datafolder
    else:
        DATA_INPUT_PATH = "datasets//housing"

    if args.model_output_folder:
        MODEL_OUTPUT_PATH = args.model_output_folder
    else:
        MODEL_OUTPUT_PATH = "artifacts"

    main()
