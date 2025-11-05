import pandas as pd
import numpy as np


def get_eta_correlation(groups: pd.Series, values: pd.Series) -> float:
    data = pd.DataFrame({"groups": groups, "values": values}).dropna()
    y_mean = data["values"].mean()
    ss_between, ss_within = 0, 0

    for group, vals in data.groupby("groups")["values"]:
        ss_between += len(vals) * (vals.mean() - y_mean) ** 2
        ss_within += ((vals - vals.mean()) ** 2).sum()

    return np.sqrt(ss_between / (ss_between + ss_within)) if (ss_between + ss_within) > 0 else 0
