import os

from catboost import CatBoostClassifier
from sklift.models import SoloModel, TwoModels
from sklift.datasets import fetch_x5
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump


def load_data(features_path, train_path):
    dataset = fetch_x5()

    df_clients = dataset.data["clients"].set_index("client_id")
    df_train = pd.concat(
        [dataset.data["train"], dataset.treatment, dataset.target], axis=1
    ).set_index("client_id")

    df_features = df_clients.copy()
    df_features["first_issue_time"] = (
        pd.to_datetime(df_features["first_issue_date"]) - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta("1s")
    df_features["first_redeem_time"] = (
        pd.to_datetime(df_features["first_redeem_date"]) - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta("1s")
    df_features["issue_redeem_delay"] = (
        df_features["first_redeem_time"] - df_features["first_issue_time"]
    )
    df_features = df_features.drop(["first_issue_date", "first_redeem_date"], axis=1)

    df_features.to_parquet(features_path)
    df_train.to_parquet(train_path)

    print("Saved to parquet")


def get_model(m="solo"):
    params = {
        "iterations": 20,
        "thread_count": 2,
        "random_state": 42,
        "silent": True,
    }

    if m == "solo":
        init_model = SoloModel(estimator=CatBoostClassifier(**params))
    else:
        init_model = TwoModels(
            estimator_trmnt=CatBoostClassifier(**params),
            estimator_ctrl=CatBoostClassifier(**params),
            method="vanilla",
        )

    return init_model


def train_model(init_model, features_path, train_path, **kwargs):
    df_features = pd.read_parquet(features_path)
    df_train = pd.read_parquet(train_path)

    indices_learn, _ = train_test_split(
        df_train.index, test_size=0.05, random_state=123
    )

    X_train = df_features.loc[indices_learn, :]
    y_train = df_train.loc[indices_learn, "target"]
    treat_train = df_train.loc[indices_learn, "treatment_flg"]

    model = init_model.fit(
        X_train,
        y_train,
        treat_train,
        **kwargs,
    )

    return model


def main():
    features_path = "data/df_features.parquet"
    train_path = "data/df_train.parquet"

    if not os.path.exists(features_path):
        load_data(features_path, train_path)

    for m in ["solo", "two"]:
        model = get_model(m)

        if m == "solo":
            model = train_model(
                model,
                features_path,
                train_path,
                estimator_fit_params={"cat_features": ["gender"]},
            )
        else:
            model = train_model(
                model,
                features_path,
                train_path,
                estimator_trmnt_fit_params={"cat_features": ["gender"]},
                estimator_ctrl_fit_params={"cat_features": ["gender"]},
            )

        dump(model, f"{m}_cb.joblib")


if __name__ == "__main__":
    main()
