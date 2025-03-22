import folium
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

# サンプル用の緯度経度データと時間情報を作成する
sales_office = pd.DataFrame(
    data=[
        [32.0, 131.1, "2025-03-22T08:00:00"],
        [33.1, 131.2, "2025-03-23T12:00:00"],
        [34.2, 131.3, "2025-03-24T16:00:00"],
    ],
    index=["本社", "A営業所", "B営業所"],
    columns=["x", "y", "time"]
)

# サンプル用の緯度経度データと時間情報を作成する
sales_office_pred = pd.DataFrame(
    data=[
        [32.0, 131.1, "2025-03-22T08:00:00"],
        [33.2, 131.0, "2025-03-23T12:00:00"],
        [34.4, 131.4, "2025-03-24T16:00:00"],
    ],
    index=["本社", "A営業所", "B営業所"],
    columns=["x", "y", "time"]
)

# データを地図に渡す関数を作成する
def AreaMarkerWithLines(df, m, rad, line_color, marker_color):
    # 時間順にデータをソート
    df = df.sort_values(by="time")

    # ポイント間を線でつなぐ
    points = df.apply(lambda r: [r.x, r.y], axis=1).to_list()
    folium.PolyLine(locations=points, color=line_color, weight=2.5, opacity=0.8).add_to(m)

    for index, r in df.iterrows():
        # ピンをおく（時間情報をポップアップに表示）
        folium.Marker(
            location=[r.x, r.y],
            popup=f"{index}<br>時間: {r.time}",  # ポップアップに時間情報を表示
            tooltip=f"{index} ({r.time})",  # ツールチップにも時間情報を表示
            icon=folium.Icon(color=marker_color)
        ).add_to(m)

        # 円を重ねる
        folium.Circle(
            radius=rad * 1000,
            location=[r.x, r.y],
            popup=index,
            color=marker_color,
            fill=True,
            fill_opacity=0.07
        ).add_to(m)

# ------------------------画面作成------------------------

st.title("サンプル地図")  # タイトル
rad = st.slider('拠点を中心とした円の半径（km）',
                value=40, min_value=5, max_value=50)  # スライダーをつける
st.subheader("各拠点からの距離{:,}km".format(rad))  # 半径の距離を表示

# 地図の初期設定
m = folium.Map(location=[sales_office_pred["x"].mean(), sales_office_pred["y"].mean()], zoom_start=12)

# 実績データをプロット
AreaMarkerWithLines(sales_office, m, rad, line_color="blue", marker_color="blue")

# 予測データをプロット
AreaMarkerWithLines(sales_office_pred, m, rad, line_color="red", marker_color="red")

# 地図情報を表示
folium_static(m)