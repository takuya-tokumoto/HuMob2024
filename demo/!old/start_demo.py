# Description: 地図上に実績データと予測データをプロットする
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

# サンプル用の緯度経度データと時間情報を作成する
trajectory_data = pd.DataFrame(
    data=[
        [0, 32.0, 131.1, "2025-03-22T08:00:00", False],
        [0, 33.1, 131.2, "2025-03-23T12:00:00", False],
        [0, 34.2, 131.3, "2025-03-24T16:00:00", False],
        [1, 33.1, 131.2, "2025-03-23T12:00:00", False],
        [1, 34.2, 131.3, "2025-03-24T16:00:00", False],
        [0, 32.0, 131.1, "2025-03-22T08:00:00", True],
        [0, 33.2, 131.0, "2025-03-23T12:00:00", True],
        [0, 34.4, 131.4, "2025-03-24T16:00:00", True],
        [1, 33.2, 131.0, "2025-03-23T12:00:00", True],
        [1, 34.4, 131.4, "2025-03-24T16:00:00", True],
    ],
    columns=["uid", "x", "y", "time", "is_pred"]
)


def AreaMarkerWithLines(
    df: pd.DataFrame,  # データフレーム（移動軌跡データ）
    m: folium.Map,  # Folium の地図オブジェクト
    rad: float,  # 円の半径（km）
    line_color: str,  # 線の色
    marker_color: str  # マーカーの色
) -> None:
    """移動軌跡情報をもとに地図上にマーカーを配置する関数

    Args:
        df (pd.DataFrame): 移動軌跡データを含むデータフレーム
        m (folium.Map): Folium の地図オブジェクト
        rad (float): 円の半径（km）
        line_color (str): 線の色
        marker_color (str): マーカーの色

    Returns:
        None: 地図オブジェクトに直接変更を加える
    """

    # 時間順にデータをソート
    df = df.sort_values(by="time")

    # ポイント間を線でつなぐ
    points = df.apply(lambda r: [r.x, r.y], axis=1).to_list()
    folium.PolyLine(locations=points, color=line_color, weight=2.5, opacity=0.8).add_to(m)

    for _, r in df.iterrows():
        # ピンをおく（時間情報をポップアップに表示）
        folium.Marker(
            location=[r.x, r.y],
            popup=f"UID: {r.uid}<br>時間: {r.time}",  # ポップアップにUIDと時間情報を表示
            tooltip=f"UID: {r.uid} ({r.time})",  # ツールチップにもUIDと時間情報を表示
            icon=folium.Icon(color=marker_color)
        ).add_to(m)

        # 円を重ねる
        folium.Circle(
            radius=rad * 1000,
            location=[r.x, r.y],
            popup=f"UID: {r.uid}",
            color=marker_color,
            fill=True,
            fill_opacity=0.07
        ).add_to(m)

# ------------------------画面作成------------------------

st.title("施策対象者の移動軌跡")  # タイトル

# UID の選択ボックスを作成
selected_uid = st.selectbox("対象者 を選択してください", trajectory_data["uid"].unique())

# 選択された UID に基づいてデータをフィルタリング
filtered_trajectory_data = trajectory_data[(trajectory_data["is_pred"] == False)&(trajectory_data["uid"] == selected_uid)]
filtered_trajectory_data_pred = trajectory_data[(trajectory_data["is_pred"] == True)&(trajectory_data["uid"] == selected_uid)]

# 地図の初期設定
m = folium.Map(location=[filtered_trajectory_data["x"].mean(), filtered_trajectory_data["y"].mean()], zoom_start=12)

# 実績データをプロット
AreaMarkerWithLines(filtered_trajectory_data, m, 0.25, line_color="blue", marker_color="blue")

# 予測データをプロット
AreaMarkerWithLines(filtered_trajectory_data_pred, m, 0.25, line_color="red", marker_color="red")

# 地図情報を表示
folium_static(m)