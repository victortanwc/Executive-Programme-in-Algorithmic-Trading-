from pathlib import Path
import pandas as pd
import numpy as np
import mlfinlab as ml

# =========================================================================
def cprint(df: pd.DataFrame, nrows: int = None):
    """
    Custom dataframe print function
    """
    if not isinstance(df, (pd.DataFrame,)):
        try:
            df = df.to_frame()
        except:
            raise ValueError("object cannot be coerced to df")

    if not nrows:
        nrows = 5
    print("-" * 79)
    print("dataframe information")
    print("-" * 79)
    print(f"HEAD num rows: {nrows}")
    print(df.head(nrows))
    print("-" * 25)
    print(f"TAIL num rows: {nrows}")
    print(df.tail(nrows))
    print("-" * 50)
    print(df.info())
    print("-" * 79)
    print()


# =========================================================================
# get project dir
def get_relative_project_dir(project_repo_name=None, partial=True):
    """helper fn to get local project directory"""
    current_working_directory = Path.cwd()
    cwd_parts = current_working_directory.parts
    if partial:
        while project_repo_name not in cwd_parts[-1]:
            current_working_directory = current_working_directory.parent
            cwd_parts = current_working_directory.parts
    else:
        while cwd_parts[-1] != project_repo_name:
            current_working_directory = current_working_directory.parent
            cwd_parts = current_working_directory.parts
    return current_working_directory


# =========================================================================


def get_Xy(indf, outcome_df, feature_cols, y_col):
    X = indf[feature_cols]
    X = X[np.isfinite(X).all(1)]
    y = (outcome_df[y_col].to_frame().reindex(X.index).sort_index().squeeze()).dropna()
    X = X.loc[y.index]
    return X, y


# =========================================================================
def calc_bins(in_data, n_days=1, pt=1, sl=2, min_ret=0.005, vol_lookback=22.0):

    daily_vol = ml.util.get_daily_vol(close=in_data["close"], lookback=vol_lookback)
    cusum_events = ml.filters.cusum_filter(
        in_data["close"], threshold=daily_vol.mean() * 0.5
    )

    t_events = cusum_events
    vertical_barriers = ml.labeling.add_vertical_barrier(
        t_events=t_events, close=in_data["close"], num_days=n_days
    )

    pt_sl = [pt, sl]

    triple_barrier_events = ml.labeling.get_events(
        close=in_data["close"],
        t_events=t_events,
        pt_sl=pt_sl,
        target=daily_vol,
        min_ret=min_ret,
        num_threads=10,
        vertical_barrier_times=vertical_barriers,
        side_prediction=None,
    )

    labels = ml.labeling.get_bins(triple_barrier_events, in_data["close"])

    print("**************************************")
    print(f"labels head:\n {labels.head()}")
    print(f"labels bin count:\n {labels.bin.value_counts()}")

    return labels, triple_barrier_events


# =========================================================================
def align_data_and_labels_indexes(in_X, in_labels, in_triple_barrier_events):
    out_triple_barrier_events = in_triple_barrier_events.loc[
        in_X.index.intersection(in_labels.index)
    ]
    out_X = in_X.loc[in_X.index.intersection(in_labels.index)]
    return out_X, out_triple_barrier_events
