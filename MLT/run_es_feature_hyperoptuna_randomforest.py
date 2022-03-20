"""
Copyright Blackarbs LLC.
Use entirely at your own risk.
This algorithm contains open source code from other sources and no claim is being
made to that code.

This implements a hyperparameter optimization script.

author: Brian Christopher, CFA, Blackarbs LLC
contact: bcr@blackarbs.com
"""
# import python scientific stack
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import neptune
import neptunecontrib.monitoring.optuna as optuna_utils
import optuna
from config_local import API_KEY  # GET YOUR API KEY FOR FREE FROM neptune.ai
from blk_ml_utils import get_relative_project_dir, cprint

RANDOM_STATE = 844

REPO_NAME = "blackarbs_algo_strategy_dev"
print("\n", REPO_NAME)
project_dir = get_relative_project_dir(REPO_NAME)
data_dir = project_dir / "data"
external = data_dir / "external"
processed = data_dir / "processed"
viz = project_dir / "viz"

# =========================================================================
# import data

infile = processed / "spy_features.h5"
with pd.HDFStore(infile) as store:
    spy = store.get("/spy/1m")


sym = "spy"
freq = "5m"
lookback = "10_day"
key = f"{sym}/{freq}/{lookback}"
print("hdf data key:", key)

data = pd.read_hdf(infile, key=key)
cprint(data)
# =========================================================================
# get binary outcomes

return_outcomes = pd.DataFrame(index=spy.index)
binary_outcomes = pd.DataFrame(index=spy.index)
periods = [
    1,
    2,
    3,
    4,
    5,
    10,
    20,
    30,
    45,
    60,
    120,
    240,
    1440,
    1440 * 2,
    1440 * 3,
    1440 * 5,
    1440 * 10,
    1440 * 20,
]
for p in tqdm(periods):
    return_outcomes[f"return_{p}"] = spy.close.pct_change(-p)
    binary_outcomes[f"return_{p}"] = return_outcomes[f"return_{p}"].apply(np.sign)
cprint(return_outcomes)
cprint(binary_outcomes)
# =========================================================================
# split train, validate, and test sets

df_train = data.loc[:"2012"]
df_validate = data.loc["2013":"2015"]
df_test = data.loc["2016":]

# =========================================================================
# create features and target, X,y


# to save time on the computation these are the hardcoded shap feature results
"""top_shap_feat_no_skew_features = [
    "rank_down_288",
    "rvol_average_price_2880",
    "racc_average_price_2880",
    "rvol_close_2880",
    "rank_volume_288",
    "racc_close_2880",
    "rank_up_288",
    "down",
    "rank_ibs_288",
    "ibs",
    "volume",
    "down",
]"""

top_shap_feat = [
    "ibs",
    "rank_ibs_288",
    "up",
    "volume",
    "down",
    "rank_up_288",
    "racc_close_2880",
    "rvol_close_2880",
    "rsi_average_price_2880",
    "rsi_close_2880",
    "rvol_average_price_2880",
    "racc_average_price_2880",
    "rank_rvwap_2880_288",
    "rank_volume_288",
    "rank_average_price_288",
]

feature_cols = top_shap_feat  # .index

y_col = "return_10"

X_train, y_train = get_Xy(df_train, binary_outcomes, feature_cols, y_col)
cprint(X_train)
cprint(y_train)

X_validate, y_validate = get_Xy(df_validate, binary_outcomes, feature_cols, y_col)
cprint(X_validate)
cprint(y_validate)

X_test, y_test = get_Xy(df_test, binary_outcomes, feature_cols, y_col)
cprint(X_test)
cprint(y_test)

# =========================================================================
# Run Neptune Optimization with Optuna


neptune.init("blackarbsceo/hpo-spy-features", api_token=API_KEY)
neptune.create_experiment(
    "hyperparameter-optuna-spy-predict-10min", upload_source_files=["*.py"]
)
neptune_callback = optuna_utils.NeptuneCallback(log_study=True, log_charts=True)


def objective(trial):
    params = {
        "criterion": "entropy",
        "n_estimators": int(
            trial.suggest_discrete_uniform("n_estimators", 50, 1500, 50)
        ),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_split": trial.suggest_uniform("min_samples_split", 0.1, 1.0),
        "max_features": trial.suggest_uniform("max_features", 0.1, 1.0),
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "oob_score": True,
    }

    rf_clf = RandomForestClassifier(**params)
    print("fitting...")
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_validate)
    score = metrics.matthews_corrcoef(y_validate, y_pred)
    print(f"internal score: {score:.4f}")
    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25, callbacks=[neptune_callback])
optuna_utils.log_study(study)
