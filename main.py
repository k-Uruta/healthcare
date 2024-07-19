import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# データの読み込み
data = pd.read_csv('walkstepcounter.csv')

# 日付を解析可能な形式に変換
data['date'] = pd.to_datetime(data['Record/@endDate'])

# 1週間ごとの集計データを作成
weekly_data = data.groupby(pd.Grouper(key='date', freq='W')).agg({
    'HKQuantityTypeIdentifierStepCount': 'sum',
    'HKQuantityTypeIdentifierDistanceWalkingRunning': 'sum',
    'HKQuantityTypeIdentifierActiveEnergyBurned': 'sum',
    'HKQuantityTypeIdentifierBasalEnergyBurned': 'mean',
    'HKQuantityTypeIdentifierWalkingSpeed': 'mean',
    'HKQuantityTypeIdentifierWalkingStepLength': 'mean',
    'HKQuantityTypeIdentifierWalkingDoubleSupportPercentage': 'mean',
    'HKQuantityTypeIdentifierWalkingAsymmetryPercentage': 'mean'
})

# 特徴量エンジニアリング
weekly_data['steps_per_distance'] = weekly_data['HKQuantityTypeIdentifierStepCount'] / weekly_data['HKQuantityTypeIdentifierDistanceWalkingRunning']
weekly_data['energy_ratio'] = weekly_data['HKQuantityTypeIdentifierActiveEnergyBurned'] / weekly_data['HKQuantityTypeIdentifierBasalEnergyBurned']

# 欠損値の処理
weekly_data = weekly_data.dropna()

# スケーリング
scaler = StandardScaler()
weekly_data_scaled = scaler.fit_transform(weekly_data)

# 異常検知モデルの構築（Isolation Forest）
clf = IsolationForest(contamination=0.1, random_state=42)
weekly_data['anomaly'] = clf.fit_predict(weekly_data_scaled)

# 異常スコアの計算
weekly_data['anomaly_score'] = clf.score_samples(weekly_data_scaled)

# 結果の表示
print(weekly_data[weekly_data['anomaly'] == -1])

# 特徴量の重要度（決定木ベースの手法で代用）
from sklearn.ensemble import RandomForestRegressor

X = weekly_data.drop(['anomaly', 'anomaly_score'], axis=1)
y = weekly_data['anomaly_score']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
print(feature_importance.sort_values('importance', ascending=False))

# 可視化
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(weekly_data.index, weekly_data['anomaly_score'], c=weekly_data['anomaly'], cmap='viridis')
plt.title('Weekly Health Anomaly Detection')
plt.xlabel('Date')
plt.ylabel('Anomaly Score')
plt.colorbar(label='Anomaly (-1) vs Normal (1)')
plt.show()

