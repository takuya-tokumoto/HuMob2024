import folium
import pandas as pd
import streamlit as st
from folium.plugins import TimestampedGeoJson
from streamlit_folium import folium_static

# サンプル用の緯度経度データと時間情報を作成する
sales_office = pd.DataFrame(
    data=[
        [32.0, 131.1, "2025-03-22T08:00:00"],
        [33.1, 131.2, "2025-03-23T12:00:00"],
        [34.2, 131.3, "2025-03-24T16:00:00"],
        [34.2, 131.4, "2025-03-24T16:30:00"],
    ],
    index=["実績", "実績", "実績", "予測"],
    columns=["x", "y", "time"]
)

# GeoJSON データを作成する関数
def create_geojson(df):
    features = []
    for index, row in df.iterrows():
        # 実績と予測で色やマークを変更
        # if index == "予測":
        #     color = "blue"  # 予測データのピンの色
        #     icon = "triangle"  # 予測データのマーク
        # else:
        #     color = "red"  # 実績データのピンの色
        #     icon = "triangle"  # 実績データのマーク

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["y"], row["x"]],
            },
            "properties": {
                "time": row["time"],
                "popup": f"{index}: {row['time']}",
                # "icon": icon,
                "iconstyle": {
                    # "fillColor": blue,
                    "fillOpacity": 0.6,
                    "stroke": "true",
                    "radius": 5,
                },
            },
        }
        features.append(feature)
    return {
        "type": "FeatureCollection",
        "features": features,
    }

# ------------------------画面作成------------------------

st.title("サンプル地図")  # タイトル

# 地図の初期設定
m = folium.Map(location=[33.1, 131.0], zoom_start=7)

# GeoJSON データを作成
geojson_data = create_geojson(sales_office)

# TimestampedGeoJson を追加
TimestampedGeoJson(
    geojson_data,
    transition_time=1000,  # アニメーションの速度 (ミリ秒)
    add_last_point=True,  # 最後のポイントを強調表示
    auto_play=False,  # 自動再生を無効化
    loop=True,  # ループ再生を有効化
    period="PT30M",  # 各タイムステップの間隔を30分に設定
).add_to(m)

# 地図情報を表示
folium_static(m)