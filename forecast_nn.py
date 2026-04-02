import argparse
import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from trader import readData


def load_price_frame(file_path, indices = [1, 3, 4, 5, 6]):
    data = readData(file_path, indices)
    data.initTechnicals()
    closes = pd.Series(data.closes, dtype=float)

    frame = pd.DataFrame({
        "date": data.dates,
        "open": data.opens,
        "high": data.highs,
        "low": data.lows,
        "close": data.closes,
        "ema_12": closes.ewm(span = 12, adjust = False, min_periods = 12).mean(),
        "ema_24": closes.ewm(span = 24, adjust = False, min_periods = 24).mean(),
        "ema_48": closes.ewm(span = 48, adjust = False, min_periods = 48).mean(),
        "ema_100": data.ema100,
        "ema_200": data.ema200,
        "rsi_14": data.rsi,
        "macd_line": data.macd[:, 0],
        "macd_signal": data.macd[:, 2],
        "macd_hist": data.macd[:, 1],
    })
    return frame.reset_index(drop = True)


def add_technicals(frame):
    closes = frame["close"]
    frame = frame.copy()

    frame["return_1h"] = closes.pct_change()
    frame["log_return_1h"] = np.log(closes / closes.shift(1))
    frame["range_pct"] = (frame["high"] - frame["low"]) / closes

    for period in [6, 12, 24, 48, 72, 168]:
        frame[f"return_{period}h"] = closes.pct_change(period)
        frame[f"volatility_{period}h"] = frame["return_1h"].rolling(period).std()

    for period in [12, 24, 48, 100, 200]:
        frame[f"close_over_ema_{period}"] = closes / frame[f"ema_{period}"] - 1.0

    return frame


def add_sequence_features(frame, window_hours):
    returns = frame["return_1h"]
    closes = frame["close"]
    lag_columns = {}

    for lag in range(1, window_hours + 1):
        lag_columns[f"return_lag_{lag}"] = returns.shift(lag)
        lag_columns[f"close_rel_lag_{lag}"] = closes.shift(lag) / closes - 1.0

    lag_frame = pd.DataFrame(lag_columns, index = frame.index)
    return pd.concat([frame, lag_frame], axis = 1)


def build_feature_matrix(frame, window_hours):
    technical_feature_columns = [
        "return_1h",
        "log_return_1h",
        "range_pct",
        "return_6h",
        "return_12h",
        "return_24h",
        "return_48h",
        "return_72h",
        "return_168h",
        "volatility_24h",
        "volatility_48h",
        "volatility_72h",
        "volatility_168h",
        "close_over_ema_12",
        "close_over_ema_24",
        "close_over_ema_48",
        "close_over_ema_100",
        "close_over_ema_200",
        "rsi_14",
        "macd_line",
        "macd_signal",
        "macd_hist",
    ]

    lag_feature_columns = []
    for lag in range(1, window_hours + 1):
        lag_feature_columns.append(f"return_lag_{lag}")
        lag_feature_columns.append(f"close_rel_lag_{lag}")

    feature_columns = technical_feature_columns + lag_feature_columns
    return frame[feature_columns].copy(), feature_columns


def build_targets(frame, horizon_hours):
    future_close = frame["close"].shift(-horizon_hours)
    future_return = future_close / frame["close"] - 1.0
    future_up = (future_return > 0).astype(int)
    return future_return, future_up


def chronological_split(num_rows, train_ratio = 0.7, validation_ratio = 0.15):
    train_end = math.floor(num_rows * train_ratio)
    validation_end = math.floor(num_rows * (train_ratio + validation_ratio))
    return slice(0, train_end), slice(train_end, validation_end), slice(validation_end, num_rows)


def train_models(features, future_return, future_up):
    regressor = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes = (64, 32),
            activation = "relu",
            solver = "adam",
            alpha = 1e-4,
            learning_rate_init = 1e-3,
            max_iter = 500,
            early_stopping = True,
            validation_fraction = 0.15,
            n_iter_no_change = 20,
            random_state = 42,
        )),
    ])

    classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes = (64, 32),
            activation = "relu",
            solver = "adam",
            alpha = 1e-4,
            learning_rate_init = 1e-3,
            max_iter = 500,
            early_stopping = True,
            validation_fraction = 0.15,
            n_iter_no_change = 20,
            random_state = 42,
        )),
    ])

    regressor.fit(features, future_return)
    classifier.fit(features, future_up)
    return regressor, classifier


def summarize_predictions(name, future_return, future_up, predicted_return, predicted_up_probability):
    predicted_up = (predicted_up_probability >= 0.5).astype(int)
    direction_accuracy = accuracy_score(future_up, predicted_up)
    mae = mean_absolute_error(future_return, predicted_return)
    rmse = np.sqrt(mean_squared_error(future_return, predicted_return))
    r2 = r2_score(future_return, predicted_return)

    summary = pd.DataFrame({
        "future_return": future_return,
        "predicted_return": predicted_return,
        "probability_up": predicted_up_probability,
    })
    top_signals = summary.sort_values("predicted_return", ascending = False).head(20)
    top_signal_realized_return = top_signals["future_return"].mean()

    print(f"{name} metrics:")
    print(f"  Direction accuracy: {round(direction_accuracy * 100, 2)}%")
    print(f"  MAE of future return: {round(mae * 100, 4)}%")
    print(f"  RMSE of future return: {round(rmse * 100, 4)}%")
    print(f"  R^2 of future return: {round(r2, 4)}")
    print(f"  Average realized return for top 20 predicted signals: {round(top_signal_realized_return * 100, 4)}%")
    print("")


def main():
    parser = argparse.ArgumentParser(description = "Train a simple neural baseline for multi-day price forecasting.")
    parser.add_argument("--file", default = "./data/hourly/eth.csv", help = "CSV file with hourly candles.")
    parser.add_argument("--horizon-hours", type = int, default = 72, help = "Forecast horizon in hours.")
    parser.add_argument("--window-hours", type = int, default = 168, help = "How many past hours to expose as lag features.")
    parser.add_argument("--indices", nargs = 5, type = int, default = [1, 3, 4, 5, 6], help = "Column indices: date, open, high, low, close.")
    args = parser.parse_args()

    frame = load_price_frame(args.file, args.indices)
    frame = add_technicals(frame)
    frame = add_sequence_features(frame, args.window_hours)
    features, feature_columns = build_feature_matrix(frame, args.window_hours)
    future_return, future_up = build_targets(frame, args.horizon_hours)

    dataset = pd.concat([
        frame[["date", "close"]],
        features,
        future_return.rename("future_return"),
        future_up.rename("future_up"),
    ], axis = 1).dropna().reset_index(drop = True)

    features = dataset[feature_columns]
    future_return = dataset["future_return"]
    future_up = dataset["future_up"]

    train_slice, validation_slice, test_slice = chronological_split(len(dataset))

    regressor, classifier = train_models(
        features.iloc[train_slice],
        future_return.iloc[train_slice],
        future_up.iloc[train_slice],
    )

    for name, data_slice in [
        ("Train", train_slice),
        ("Validation", validation_slice),
        ("Test", test_slice),
    ]:
        slice_features = features.iloc[data_slice]
        slice_future_return = future_return.iloc[data_slice]
        slice_future_up = future_up.iloc[data_slice]
        predicted_return = regressor.predict(slice_features)
        predicted_up_probability = classifier.predict_proba(slice_features)[:, 1]
        summarize_predictions(name, slice_future_return, slice_future_up, predicted_return, predicted_up_probability)

    test_features = features.iloc[test_slice]
    test_frame = dataset.iloc[test_slice][["date", "close", "future_return"]].copy()
    test_frame["predicted_return"] = regressor.predict(test_features)
    test_frame["probability_up"] = classifier.predict_proba(test_features)[:, 1]

    print("Latest test predictions:")
    print(test_frame.tail(10).to_string(index = False))


if __name__ == "__main__":
    main()