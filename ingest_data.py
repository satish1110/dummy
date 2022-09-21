import argparse
import os
import tarfile

import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import mlflow


data_path = "datasets//housing"

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Column indices used in CombinedAttributesAdder
ROOMS_IX = None
BEDROOMS_IX = None
POPULATION_IX = None
HOUSEHOLDS_IX = None

DATA_OUTPUT_PATH = data_path


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Transformer Class used to create new attributes from existing ones and return them.
    TransformerMixin is used to inherit the fit_transform method.
    BaseEstimator provides two extra methods (get_params() and set_params()) that will be useful
    for automatic hyperparameter tuning.
    """

    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        mlflow.log_param("add_bedrooms_per_room", add_bedrooms_per_room)

        # self.df = None

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        # self.df = df
        # X = df.values
        rooms_per_household = X[:, ROOMS_IX] / X[:, HOUSEHOLDS_IX]
        population_per_household = X[:, POPULATION_IX] / X[:, HOUSEHOLDS_IX]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, BEDROOMS_IX] / X[:, ROOMS_IX]
            return np.c_[
                X,
                rooms_per_household,
                bedrooms_per_room,
                population_per_household,
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

    # to get i/p and added features for column recovery
    # temporary hack, need to generalize
    # def get_fetaure_names_out(self):
    #   df = self.df
    #   if self.add_bedrooms_per_room:
    #      columns = list(df.columns)+["rooms_per_household", "bedrooms_per_room", "population_per_household"]
    #  else:
    #      columns = list(df.columns)+["rooms_per_household", "population_per_household"]

    #  return columns


def get_feature_names_from_column_transformer(col_trans):
    """Get feature names from a sklearn column transformer.

    The `ColumnTransformer` class in `scikit-learn` supports taking in a
    `pd.DataFrame` object and specifying `Transformer` operations on columns.
    The output of the `ColumnTransformer` is a numpy array that can used and
    does not contain the column names from the original dataframe. The class
    provides a `get_feature_names` method for this purpose that returns the
    column names corr. to the output array. Unfortunately, not all
    `scikit-learn` classes provide this method (e.g. `Pipeline`) and still
    being actively worked upon.

	NOTE: This utility function is a temporary solution until the proper fix is
    available in the `scikit-learn` library.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder as skohe

    # SimpleImputer has `add_indicator` attribute that distinguishes it from other transformers
    # Encoder had `get_feature_names` attribute that distinguishes it from other transformers
    # The last transformer is ColumnTransformer's 'remainder'

    # o/p
    col_name = []
    # loop over transformers_ types
    for transformer_in_columns in col_trans.transformers_:
        is_pipeline = 0
        # get input cols
        raw_col_name = list(transformer_in_columns[2])

        if isinstance(transformer_in_columns[1], Pipeline):
            # if pipeline, get the last transformer
            transformer = transformer_in_columns[1].steps[-1][1]
            is_pipeline = 1
        else:
            # some method
            transformer = transformer_in_columns[1]

        try:
            if isinstance(transformer, str):
                # print(1)
                if transformer == "passthrough":
                    # print(1.1)
                    # list of ip features
                    names = transformer._feature_names_in[
                        raw_col_name
                    ].tolist()

                elif transformer == "drop":
                    # empty
                    names = []
                    # print(1.2)

                else:
                    raise RuntimeError(
                        f"Unexpected transformer action for unaccounted cols :"
                        f"{transformer} : {raw_col_name}"
                    )

            elif isinstance(transformer, skohe):
                # print(2)
                names = [
                    "rooms_per_household",
                    "bedrooms_per_room",
                    "population_per_household",
                ]

                # ip_col_c1, ip_col_c2...
                names.extend(list(transformer.get_feature_names(raw_col_name)))

            # If True, a MissingIndicator transform will stack onto output of the imputer’s transform.
            # This allows a predictive estimator to account for missingness despite imputation. If a feature
            #  has no missing values at fit/train time, the feature won’t appear on the missing indicator even
            #  if there are missing values at transform/test time.
            elif (
                isinstance(transformer, SimpleImputer)
                and transformer.add_indicator
            ):
                # print(3)

                missing_indicator_indices = transformer.indicator_.features_
                missing_indicators = [
                    raw_col_name[idx] + "_missing_flag"
                    for idx in missing_indicator_indices
                ]

                names = raw_col_name + missing_indicators
                print(missing_indicators)

            else:
                # print("4")
                # doesn't work for CombinedAttributesAdder ???
                print(transformer.get_feature_names_out())
                names = list(transformer.get_feature_names_out())

        except AttributeError as error:
            # just return ip cols
            names = raw_col_name

        if is_pipeline:
            # removed pipeline_ prefix since not needed
            names = [f"{col_}" for col_ in names]

        col_name.extend(names)

    # print(col_name)
    return col_name


def transform_df(df):
    """Function to transform and prepare housinf df data using pipelines. Mainly adds 3 attributes,
       does Median imputation for missing values and One-hot encoding for categorical variables.
    Parameters
    ----------
            df:
                dataframe on which pipeline transformation needs to be applied.

    Returns
    -------
            df_prepared:
                Final dataframe after pipeline transformation and recovering columns
    """
    # Pipeline for numerical columns, do Median imputation and add 3 attributes
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
        ]
    )

    df_num = df.drop(["ocean_proximity", "median_house_value"], axis=1)

    num_attribs = list(df_num)
    cat_attribs = ["ocean_proximity"]

    mlflow.log_param("numerical_attributes", num_attribs)
    mlflow.log_param("categorical_attributes", cat_attribs)

    # Transform columns using Full pipeline: One pipeline for numerical columns, other pipeline for
    # categorical columns which consists of Onehot encoding.
    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", dtype="int"
                ),
                cat_attribs,
            ),
        ]
    )

    df_prepared = full_pipeline.fit_transform(df)

    cols = get_feature_names_from_column_transformer(full_pipeline)

    # recover the columns
    df_prepared = pd.DataFrame(df_prepared, columns=cols)

    return df_prepared


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Function to fetch housing data from the web and write it to a path.
    Parameters
    ----------
            housing_url:
                    Web url to fetch the housing data from.
            housing_path:
                path to write the fetched data from.

    Returns
    -------
    """
    mlflow.log_param("housing_url", housing_url)
    mlflow.log_param("housing_path", housing_path)

    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """Function to load housing data from a path. Also adds a column to the data.
    Parameters
    ----------
            housing_path:
                    path to read the housing data from.

    Returns
    -------
            housing:
                housing data frame
    """

    csv_path = os.path.join(housing_path, "housing.csv")
    housing = pd.read_csv(csv_path)
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return housing


# train test split
def split_train_test(data_output_path=DATA_OUTPUT_PATH):
    """Function to split data into train and test and write it.
    Parameters
    ----------
            data_output_path:
                    path to write the split data.

    Returns
    -------
    """
    mlflow.log_param("data_output_path", data_output_path)

    housing = load_housing_data()
    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    strat_train_set.to_csv(
        data_output_path + "//strat_train_set.csv", index=False
    )
    strat_test_set.to_csv(
        data_output_path + "//strat_test_set.csv", index=False
    )
    test_set.to_csv(data_output_path + "//test_set.csv", index=False)


def income_cat_proportions(data):
    """Function to return the proportions of income category from the dataframe
    Parameters
    ----------
            data:
                dataframe with the income_cat column

    Returns
    -------
            proportions of income category from the dataframe
    """
    return data["income_cat"].value_counts() / len(data)


# compare income cat proportions in the various data splits
def perform_eda(data_output_path=DATA_OUTPUT_PATH):
    """Function to perform exploratory data analysis on the data. Also modifies strat train/test set
    Parameters
    ----------
            data_output_path:
                    path to read the data from.

    Returns
    -------
    """
    test_set = pd.read_csv(data_output_path + "//test_set.csv")
    strat_train_set = pd.read_csv(data_output_path + "//strat_train_set.csv")
    strat_test_set = pd.read_csv(data_output_path + "//strat_test_set.csv")
    housing = load_housing_data()

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()

    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )

    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    strat_train_set.to_csv(
        data_output_path + "//strat_train_set.csv", index=False
    )
    strat_test_set.to_csv(
        data_output_path + "//strat_test_set.csv", index=False
    )

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    # correlation matrix
    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)

    housing["rooms_per_household"] = (
        housing["total_rooms"] / housing["households"]
    )

    housing["bedrooms_per_room"] = (
        housing["total_bedrooms"] / housing["total_rooms"]
    )

    housing["population_per_household"] = (
        housing["population"] / housing["households"]
    )


def prepare_train_test_data(data_output_path=DATA_OUTPUT_PATH):
    """Function to prepare the train and test data for training and write it.
    Parameters
    ----------
            data_output_path:
                    path to read the data from.

    Returns
    -------
    """
    if not (os.path.exists(data_output_path)):
        os.makedirs(data_output_path)

    strat_train_set = pd.read_csv(data_output_path + "//strat_train_set.csv")
    strat_test_set = pd.read_csv(data_output_path + "//strat_test_set.csv")

    col_names = "total_rooms", "total_bedrooms", "population", "households"

    global ROOMS_IX, BEDROOMS_IX, POPULATION_IX, HOUSEHOLDS_IX

    ROOMS_IX, BEDROOMS_IX, POPULATION_IX, HOUSEHOLDS_IX = [
        strat_train_set.columns.get_loc(c) for c in col_names
    ]  # get the column indices

    # prepare train

    housing_labels = strat_train_set["median_house_value"].copy()
    housing_prepared = transform_df(strat_train_set)

    housing_prepared.to_csv(
        data_output_path + "//housing_prepared.csv", index=False
    )
    housing_labels.to_csv(
        data_output_path + "//housing_labels.csv", index=False
    )

    # random forest data
    X_test_prepared = transform_df(strat_test_set)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_prepared.to_csv(
        data_output_path + "//X_test_prepared.csv", index=False
    )
    y_test.to_csv(data_output_path + "//y_test.csv", index=False)


def main():
    """ Main function that calls other functions in order.
    Parameters
    ----------
    Returns
    -------
    """

    fetch_housing_data()
    split_train_test()
    perform_eda()
    prepare_train_test_data()


if __name__ == "__main__":
    """Driver function that has argument parser and calls main function.
    Parameters
    ----------
    Returns
    -------
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dof",
        "--data_output_folder",
        help="Output folder path for prepared data:",
    )

    args = parser.parse_args()

    if args.data_output_folder:
        DATA_OUTPUT_PATH = args.data_output_folder
    else:
        DATA_OUTPUT_PATH = data_path

    main()
