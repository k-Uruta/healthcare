import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Streamlitの設定
st.title("Weekly Health Anomaly Detection")
st.write("""
    このアプリでは、週ごとの健康データに基づいて異常検知を行います。
    CSVファイルをアップロードして、異常検知を実行します。
""")

# データの読み込み
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # データの構造を確認
    st.subheader("データプレビュー")
    st.write(data.head())
    
    st.subheader("カラム名")
    st.write(data.columns)

    # 日付カラムの処理
    date_column = '/Record/@endDate'
    if date_column not in data.columns:
        st.error(f"'{date_column}' カラムが見つかりません。CSVファイルの構造を確認してください。")
        st.stop()
    
    data['date'] = pd.to_datetime(data[date_column])

    # データタイプと値のカラムを特定
    type_column = '/Record/@type'
    value_column = '/Record/@value'
    
    if type_column not in data.columns or value_column not in data.columns:
        st.error(f"'{type_column}' または '{value_column}' カラムが見つかりません。CSVファイルの構造を確認してください。")
        st.stop()

    # 1週間ごとの集計データを作成
    weekly_data = data.groupby([pd.Grouper(key='date', freq='W'), type_column])[value_column].mean().unstack()

    # 欠損値の処理
    weekly_data = weekly_data.dropna(axis=1, how='all')  # すべての値がNaNの列を削除
    weekly_data = weekly_data.dropna()  # NaNを含む行を削除

    if weekly_data.empty:
        st.error("データの処理後に有効なデータがありません。データを確認してください。")
        st.stop()

    st.subheader("処理後のデータ")
    st.write(weekly_data.head())

    # スケーリング
    scaler = StandardScaler()
    weekly_data_scaled = scaler.fit_transform(weekly_data)

    # 異常検知モデルの構築（Isolation Forest）
    contamination = st.slider("異常値の割合", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    clf = IsolationForest(contamination=contamination, random_state=42)
    weekly_data['anomaly'] = clf.fit_predict(weekly_data_scaled)

    # 異常スコアの計算
    weekly_data['anomaly_score'] = clf.score_samples(weekly_data_scaled)

    st.subheader("異常検知結果")
    st.write(weekly_data[weekly_data['anomaly'] == -1])

    # 特徴量の重要度（決定木ベースの手法で代用）
    X = weekly_data.drop(['anomaly', 'anomaly_score'], axis=1)
    y = weekly_data['anomaly_score']

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
    st.subheader("特徴量の重要度")
    st.write(feature_importance.sort_values('importance', ascending=False))

    # 可視化
    plt.figure(figsize=(12, 6))
    plt.scatter(weekly_data.index, weekly_data['anomaly_score'], c=weekly_data['anomaly'], cmap='viridis')
    plt.title('Weekly Health Anomaly Detection')
    plt.xlabel('Date')
    plt.ylabel('Anomaly Score')
    plt.colorbar(label='Anomaly (-1) vs Normal (1)')
    st.pyplot(plt)
else:
    st.write("CSVファイルをアップロードしてください。")