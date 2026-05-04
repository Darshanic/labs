from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

csv_path = Path(__file__).with_name("VegetablePrice.csv")
if not csv_path.exists():
    raise FileNotFoundError(
        f"Missing dataset: {csv_path}. Place VegetablePrice.csv next to lab6.py or update csv_path."
    )

df = pd.read_csv(csv_path)

df['Date'] = pd.to_datetime(df['Date'])
# Select one commodity
data = df[df['Commodity'] == "Tomato Big(Nepali)"]
# Sort by date
data = data.sort_values('Date')
# Select only average price
data = data[['Date','Average']]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data[['Average']])

def create_dataset(dataset, time_step=10):
    X, y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step),0])
        y.append(dataset[i+time_step,0])
    return np.array(X), np.array(y)
time_step = 10
X, y = create_dataset(scaled_data, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step,1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adagrad', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)