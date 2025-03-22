import folium
import polars as pl
import streamlit as st
from folium.plugins import TimestampedGeoJson
from streamlit_folium import st_folium


def view_user_trajectory(df_a):
    # マップの中心をデータの平均値に設定
    m = folium.Map(location=[df_a["latitude"].mean(), df_a["longitude"].mean()], zoom_start=12)
    data = []
    uid_groups = df_a.groupby("uid")  # UID ごとにグループ化

    for uid, group in uid_groups:
        # 各 UID の移動軌跡を線でつなぐ
        coordinates = []
        for row in group.iter_rows(named=True):
            time_str = row.get("datetime_str") or str(row["datetime"])
            coordinates.append([row["latitude"], row["longitude"]])  # 緯度・経度を追加

            # 各ポイントを GeoJSON データとして追加
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["longitude"], row["latitude"]],
                },
                "properties": {
                    "time": time_str,
                    "popup": f"uid: {row['uid']}<br>{time_str}",
                    "icon": "circle",
                    "iconstyle": {
                        "fillColor": "red",
                        "fillOpacity": 0.6,
                        "stroke": "true",
                        "radius": 5,
                    },
                },
            }
            data.append(feature)

        # UID ごとに PolyLine を追加
        folium.PolyLine(
            locations=coordinates,
            color="blue",
            weight=2,
            opacity=0.7,
            popup=f"Trajectory for UID: {uid}",
        ).add_to(m)

    # TimestampedGeoJson の設定を調整
    TimestampedGeoJson(
        {
            "type": "FeatureCollection",
            "features": data,
        },
        transition_time=1,
        add_last_point=True,  # 最後のポイントを強調表示
        period="PT1S",
        auto_play=True,  # アニメーションを自動再生
        loop=False,
    ).add_to(m)

    # Streamlit 用にマップを表示
    st_folium(m, width=800, height=600)  # マップサイズを調整


# データ読み込み
df = pl.read_csv("/kaggle/HuMob2024/demo/demo_dataset.csv")
view_user_trajectory(df)