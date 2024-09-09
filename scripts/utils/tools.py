import matplotlib.pyplot as plt
import plotly.express as px
import polars as pl
import seaborn as sns


def check_pk(df_pl: pl.DataFrame, cols: list) -> None:
    """primery keyのチェック"""

    # 重複のない組み合わせの数
    unique_combinations = df_pl.unique(subset=cols).height

    # データ全体の行数
    total_rows = len(df_pl)

    # 比較して一意性を確認
    if unique_combinations == total_rows:
        print(f"{', '.join(cols)} の組み合わせは一意であり、PKとして使用できます。")
    else:
        print(
            f"{', '.join(cols)} の組み合わせには重複があります。重複している行数は {total_rows - unique_combinations} 行です。"
        )


def check_n_unique(df_pl: pl.DataFrame, col: str) -> None:
    """指定したカラムのユニークカウント数を表示"""

    _n_unique = df_pl[col].n_unique()
    print(f"{col}のユニーク件数： {_n_unique}")


def plot_record_cnts_by_d(city_df: pl.DataFrame) -> None:
    """日単位(d)でレコード数の推移を可視化するグラフを描画"""

    # プロットのためにデータ加工
    agg_df = city_df.group_by("d").len().sort("d")

    # グラフの描画
    plt.figure(figsize=(10, 6))
    plt.plot(agg_df["d"], agg_df["len"], marker="o", label="Data Count")

    plt.xlabel("d")
    plt.ylabel("counts")
    plt.xlim()
    plt.ylim()
    plt.grid(True)
    plt.legend()  # 凡例を追加
    plt.show()


def plot_record_cnts_heatmap(city_df: pl.DataFrame) -> None:
    """x, y 座標ごとのデータ数をヒートマップで可視化"""

    # 999で位置がマスクされた実績は排除
    city_df = city_df.filter((pl.col("x") != 999) & (pl.col("y") != 999))

    # x, y 座標ごとのデータ数をカウント
    agg_df = city_df.group_by(["x", "y"]).agg(pl.len()).sort(["x", "y"])

    # 外れ値があるので対数変換してから可視化
    agg_df_log = agg_df.with_columns((pl.col("len") + 1).log())

    # データをピボットテーブルに変換
    pivot_df = agg_df_log.pivot(values="len", index="y", columns="x")

    # x, yで欠損していた場合には0埋めを行う
    pivot_df = pivot_df.fill_null(0)

    # y列をindexで扱うためpolarsからpandasへ返還
    pd_pivot_df = pivot_df.to_pandas()
    pd_pivot_df = pd_pivot_df.set_index("y").sort_index(ascending=True)

    # ヒートマップの描画
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd_pivot_df, annot=False, fmt="d", cmap="YlGnBu", cbar=True)

    plt.title("Heatmap of Data Counts on log scale by (x, y) Coordinates")
    plt.xlabel("x Coordinate")
    plt.ylabel("y Coordinate")
    plt.show()


def plot_xy_trajectory(city_df: pl.DataFrame, selected_uids: list[int]) -> None:
    """ユーザーごとのx, yの推移を時間軸に沿って折れ線グラフで可視化

    args
        city_df: Polarsのデータフレーム
        selected_uids: 表示したいユーザーIDのリスト
    """

    # 選択されたUIDのデータをフィルタリングし、時間でソート
    dff = city_df.filter(pl.col("uid").is_in(selected_uids))
    dff = dff.with_columns((pl.col("d") * 48 + pl.col("t")).alias("time"))
    dff = dff.sort("time")

    # Pandasデータフレームに変換
    dff_pd = dff.to_pandas()

    # X座標の推移を描画
    x_line_fig = px.line(
        dff_pd,
        x="time",
        y="x",
        color="uid",
        labels={"x": "Time", "y": "X Position", "uid": "User ID"},
        title="User X Position Over Time",
    )
    x_line_fig.update_traces(mode="lines+markers")

    # Y座標の推移を描画
    y_line_fig = px.line(
        dff_pd,
        x="time",
        y="y",
        color="uid",
        labels={"x": "Time", "y": "Y Position", "uid": "User ID"},
        title="User Y Position Over Time",
    )
    y_line_fig.update_traces(mode="lines+markers")

    # グラフの表示
    x_line_fig.show()
    y_line_fig.show()
