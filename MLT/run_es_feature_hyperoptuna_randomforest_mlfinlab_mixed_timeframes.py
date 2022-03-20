"""
Copyright Blackarbs LLC.
Use entirely at your own risk.
This algorithm contains open source code from other sources and no claim is being
made to that code.

This implements a hyperparameter optimization script.

author: Brian Christopher, CFA, Blackarbs LLC
contact: bcr@blackarbs.com
"""


def run_experiment():
    # import python scientific stack
    import pandas as pd
    import mlfinlab as ml
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics
    import neptune
    import neptunecontrib.monitoring.optuna as optuna_utils
    import optuna
    from config_local import API_KEY  # GET YOUR API KEY FOR FREE FROM neptune.ai
    from blk_ml_utils import (
        get_relative_project_dir,
        cprint,
        align_data_and_labels_indexes,
        calc_bins,
        get_Xy,
    )

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

    # 5 minute
    sym = "spy"
    freq = "5m"
    lookback = "10_day"
    key = f"{sym}/{freq}/{lookback}"
    print("hdf data key:", key)

    data = pd.read_hdf(infile, key=key)
    cprint(data)

    # 30 minute
    sym = "spy"
    freq = "30m"
    lookback = "10_day"
    key = f"{sym}/{freq}/{lookback}"
    print(key)

    data2 = pd.read_hdf(infile, key=key)
    cprint(data2)

    # =========================================================================
    # Join multiple timeframes

    data_join = data.join(data2, how="left", rsuffix="_30m").ffill()
    cprint(data_join)
    print("*" * 55)
    print("joined data columns:")
    print(data_join.columns.tolist())

    # =========================================================================
    # create features and target, X,y

    # dollar bars
    dbars = ml.data_structures.get_dollar_bars(
        data_join.reset_index().rename(columns={"datetime": "date_time"}),
        threshold=1_000_000,
    ).set_index("date_time")
    cprint(dbars)

    X = (
        dbars.join(
            data_join.drop(["open", "high", "low", "close", "volume"], axis=1),
            how="left",
        )
        .drop_duplicates()
        .dropna()
    )
    cprint(X)

    # =========================================================================
    # split train, validate, and test sets

    df_train = X.loc[:"2012"]
    df_validate = X.loc["2013":"2015"]
    df_test = X.loc["2016":]

    # =========================================================================
    # labeling

    labels, triple_barrier_events = calc_bins(X, pt=1.25, sl=1, n_days=1)
    X, triple_barrier_events = align_data_and_labels_indexes(
        X, labels, triple_barrier_events
    )

    # =========================================================================
    # Split feature data and label data

    # to save time on the computation these are the hardcoded shap feature results
    top_shap_feat = [
        "rank_racc_rvwap_2880_2880_288",
        "rank_average_price_288",
        "up",
        "tick_num",
        "slope_lower_band_rvwap_2880_2880",
        "slope_lower_band_rvwap_480_480",
        "rank_aqr_momo_average_price_2880_288",
        "rank_upper_band_rvwap_480_48",
        "rank_racorr_close_480_48",
        "rank_racorr_average_price_2880_288",
        "slope_upper_band_rvwap_2880_2880",
        "rank_slope_lower_band_rvwap_2880_2880_288",
        "slope_rmin_low_2880_2880",
        "ibs_30m",
        "rank_rvwap_480_48",
        "down",
        "rank_aqr_momo_close_2880_288",
        "aqr_momo_average_price_480",
        "rank_aqr_momo_average_price_480_48",
        "rank_slope_rmin_low_2880_2880_288",
        "rsi_close_480",
        "down_30m",
        "rank_racc_rvwap_480_480_48",
        "racorr_close_480",
        "rank_up_48",
        "rank_average_price_48",
        "slope_rmin_low_480_480",
        "rank_slope_lower_band_rvwap_480_480_48",
        "rank_rvwap_2880_288",
        "rank_ibs_48",
        "slope_rmax_high_2880_2880",
        "rank_up_288",
        "slope_rmax_high_480_480",
        "lower_band_rvwap_2880",
        "rank_aqr_momo_rvwap_480_480_48",
        "rank_aqr_momo_rvwap_2880_2880_288",
        "rank_down_288",
    ]

    feature_cols = top_shap_feat

    y_col = "bin"

    X_train, y_train = get_Xy(df_train, labels, feature_cols, y_col)
    cprint(X_train)
    cprint(y_train)

    X_validate, y_validate = get_Xy(df_validate, labels, feature_cols, y_col)
    cprint(X_validate)
    cprint(y_validate)

    # =========================================================================
    # Run Neptune Optimization with Optuna

    neptune.init("blackarbsceo/hpo-spy-features", api_token=API_KEY)
    neptune.create_experiment(
        "hyperparameter-optuna-rf-mlfinlab-mixed-5Min-30Min-1MMdb",
        upload_source_files=["*.py"],
    )
    neptune_callback = optuna_utils.NeptuneCallback(log_study=True, log_charts=True)

    def objective(trial):
        params = {
            "criterion": "entropy",
            "n_estimators": int(
                trial.suggest_discrete_uniform("n_estimators", 100, 2500, 100)
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
    study.optimize(objective, n_trials=30, callbacks=[neptune_callback])
    optuna_utils.log_study(study)


if __name__ == "__main__":
    run_experiment()
