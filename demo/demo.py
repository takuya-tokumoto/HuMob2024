import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd


def load_datasetA(datasetA_path: str) -> pl.DataFrame:
    """HUNOB2024データセットAを読み込み、緯度・経度、詳細時刻、uidごとの方位情報を追加する"""

    # CSV読み込み
    df_a = pl.read_csv(datasetA_path)
    df_a = df_a.filter(pl.col("uid")==0)

    # 緯度経度情報の追加（例: 緯度 = x*0.005 + 34.497, 経度 = y*0.005 + 136.5）
    add_lat_col = (pl.col("x")*0.005 + 34.497).alias("latitude")
    add_lon_col = (pl.col("y")*0.005 + 136.5).alias("longitude")
    df_a = df_a.with_columns(add_lat_col).with_columns(add_lon_col)

    # 詳細時刻の追加
    # d: 0～74, 0 が 2020/01/05 を表す
    # t: 0～47, 0 が 0時、以降30分間隔
    start_date = pl.lit("2020-01-05T00:00:00").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S")
    df_a = df_a.with_columns(
        (
            start_date
            + pl.col("d") * pl.duration(days=1)
            + pl.col("t") * pl.duration(minutes=30)
        ).cast(pl.Datetime).alias("datetime")
    )

    # datetime 列を "YYYY-MM-DD HH:mm" の文字列に変換
    # df_a = df_a.with_columns(
    #     pl.col("datetime").dt.strftime("%Y-%m-%d %H:%M").alias("datetime_str")
    # )

    # uid ごとに、日時順にソート
    df_a = df_a.sort(["uid", "datetime"])

    # 同一 uid 内で、前の行の緯度・経度を取得（最初の行はnullになる）
    df_a = df_a.with_columns(
        pl.col("latitude").shift(1).over("uid").alias("prev_lat"),
        pl.col("longitude").shift(1).over("uid").alias("prev_lon")
    )
    
    # 前の地点と現在の地点から方位を計算する関数
    def calculate_bearing(row: dict) -> float:
        prev_lat = row["prev_lat"]
        prev_lon = row["prev_lon"]
        lat = row["latitude"]
        lon = row["longitude"]
        # 最初の行は前の位置がないので None を返す
        if prev_lat is None or prev_lon is None:
            return None
        # 度をラジアンに変換
        lat1 = np.radians(prev_lat)
        lon1 = np.radians(prev_lon)
        lat2 = np.radians(lat)
        lon2 = np.radians(lon)
        delta_lon = lon2 - lon1
        x = np.sin(delta_lon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon))
        initial_bearing = np.arctan2(x, y)
        initial_bearing = np.degrees(initial_bearing)
        # 0～360度に正規化
        bearing = (initial_bearing + 360) % 360
        return bearing

    # 各行ごとに、前の緯度・経度と現在の緯度・経度から方位を計算
    df_a = df_a.with_columns(
        pl.struct(["prev_lat", "prev_lon", "latitude", "longitude"]).apply(calculate_bearing).alias("bearing")
    )

    return df_a

def view_user_trajectory(df_a):
    m = folium.Map(location=[df_a["latitude"].min(), df_a["longitude"].min()], zoom_start=12)
    data = []
    for row in df_a.iter_rows(named=True):
        # もし datetime 列が Timestamp オブジェクトなら文字列に変換する
        # ここでは、"datetime_str" が存在する場合はそれを使い、なければ str() で変換
        time_str = row.get("datetime_str") or str(row["datetime"])
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["longitude"], row["latitude"]],
            },
            "properties": {
                "time": time_str,  # ここを文字列にしておく
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

    TimestampedGeoJson(
        {
            "type": "FeatureCollection",
            "features": data,
        },
        transition_time=1,
        add_last_point=False,
        period="PT1S",
        auto_play=False,
        loop=False,
    ).add_to(m)
    
    display(m)

df_a = load_datasetA("/kaggle/s3storage/01_public/humob-challenge-2024/input/cityA_groundtruthdata.csv.gz")
view_user_trajectory_v3(df_a)