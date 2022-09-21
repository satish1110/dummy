# Driver script to import the installed packages and call the functions accordingly

from house_price_prediction_satish import ingest_data, score, train
from logger import configure_logger

import mlflow
import mlflow.sklearn

LOG_LEVEL = "DEBUG"
LOG_PATH = "logs//scripts.log"
IS_CONSOLE_LOG = True

my_logger = configure_logger(
    log_file=LOG_PATH, console=IS_CONSOLE_LOG, log_level=LOG_LEVEL
)

if __name__ == "__main__":
    remote_server_uri = "http://0.0.0.0:5000"  # set to your server URI
    mlflow.set_tracking_uri(
        remote_server_uri
    )  # or set the MLFLOW_TRACKING_URI in the env
    exp_name = "House-price-prediction"
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="PARENT_RUN-driver") as parent_run:

        mlflow.log_param("parent-driver", "yes")

        with mlflow.start_run(
            run_name="CHILD_RUN-ingest_data", nested=True
        ) as child_run:
            mlflow.log_param("child-ingest_data", "yes")
            ingest_data.fetch_housing_data()
            my_logger.info("Housing data downloaded!")
            my_logger.info("Housing data loaded!")

            ingest_data.split_train_test()
            my_logger.info("Train and test data split done and written!")

            ingest_data.perform_eda()
            my_logger.info("EDA for Housing data done!")

            ingest_data.prepare_train_test_data()
            my_logger.info("Prepared housing data written!")
            my_logger.info("Prepared housing data for random forest written!")

        with mlflow.start_run(
            run_name="CHILD_RUN-train", nested=True
        ) as child_run2:

            mlflow.log_param("child-train", "yes")
            train.train_lin_reg()
            my_logger.info("Training data read!")
            my_logger.info("Training done for linear regression!")

            train.train_tree_reg()
            my_logger.info("Training data read!")
            my_logger.info("Training done for decision tree regression!")

            train.train_randfor_reg()
            my_logger.info("Training data read!")
            my_logger.info("Training done for random forest regression!")

        with mlflow.start_run(
            run_name="CHILD_RUN-score", nested=True
        ) as child_run3:

            mlflow.log_param("child-score", "yes")

            my_logger.info("Prepared data read!")
            score.score_lin_reg()
            my_logger.info("Scoring done for linear regression!")

            my_logger.info("Prepared data read!")
            score.score_tree_reg()
            my_logger.info("Scoring done for decision tree regression!")

            my_logger.info("Prepared data read!")
            score.score_randfor_reg()
            my_logger.info("Scoring done for random forest regression!")
