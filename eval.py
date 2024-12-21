from typing import List, Tuple

import geobleu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns


def caluc_test_score(subs: dict) -> Tuple[List, List]:
    """評価用データに関する精度結果を表示

    Args:
        subs (dict): 提出用subs

    Returns:
        list: 各uidごとのgeobleuの値
        list: 各uidごとのdtwの値
    """

    list_geobleu_val = []
    list_dtw_val = []
    for i in range(len(subs["reference"])):

        generated = subs["generated"][i]
        reference = subs["reference"][i]

        geobleu_val = geobleu.calc_geobleu(generated, reference, processes=3)
        list_geobleu_val.append(geobleu_val)

        dtw_val = geobleu.calc_dtw(generated, reference, processes=3)
        list_dtw_val.append(dtw_val)

    return np.mean(list_geobleu_val), np.mean(list_dtw_val)
