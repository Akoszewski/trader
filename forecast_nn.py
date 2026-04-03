import argparse
import copy
import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from trader import readData

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit("PyTorch is required. Install dependencies from requirements.txt before running forecast_nn.py.") from exc


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


def build_feature_columns():
    return [
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


def build_targets(frame, horizon_hours):
    future_close = frame["close"].shift(-horizon_hours)
    future_return = future_close / frame["close"] - 1.0
    future_up = (future_return > 0).astype(int)
    return future_return, future_up


def chronological_split(num_rows, train_ratio = 0.7, validation_ratio = 0.15):
    train_end = math.floor(num_rows * train_ratio)
    validation_end = math.floor(num_rows * (train_ratio + validation_ratio))
    return slice(0, train_end), slice(train_end, validation_end), slice(validation_end, num_rows)


def build_sequence_dataset(frame, feature_columns, future_return, future_up, window_hours):
    feature_matrix = frame[feature_columns].to_numpy(dtype=np.float32)
    future_return_values = future_return.to_numpy(dtype=np.float32)
    future_up_values = future_up.to_numpy(dtype=np.float32)
    close_values = frame["close"].to_numpy(dtype=np.float32)
    date_values = frame["date"].to_numpy()

    sequences = []
    target_returns = []
    target_ups = []
    current_closes = []
    current_dates = []

    for end_idx in range(window_hours - 1, len(frame)):
        start_idx = end_idx - window_hours + 1
        window = feature_matrix[start_idx:end_idx + 1]
        if np.isnan(window).any():
            continue

        target_return = future_return_values[end_idx]
        target_up = future_up_values[end_idx]
        if np.isnan(target_return) or np.isnan(target_up):
            continue

        sequences.append(window)
        target_returns.append(target_return)
        target_ups.append(target_up)
        current_closes.append(close_values[end_idx])
        current_dates.append(date_values[end_idx])

    return {
        "features": np.asarray(sequences, dtype=np.float32),
        "future_return": np.asarray(target_returns, dtype=np.float32),
        "future_up": np.asarray(target_ups, dtype=np.float32),
        "close": np.asarray(current_closes, dtype=np.float32),
        "date": np.asarray(current_dates),
    }


def scale_sequence_features(train_features, all_features):
    num_features = train_features.shape[-1]
    scaler = StandardScaler()
    scaler.fit(train_features.reshape(-1, num_features))

    scaled = scaler.transform(all_features.reshape(-1, num_features))
    return scaled.reshape(all_features.shape), scaler


class LSTMForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = 1,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.return_head = nn.Linear(hidden_size, 1)
        self.direction_head = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        _, (hidden_state, _) = self.lstm(inputs)
        features = self.dropout(hidden_state[-1])
        features = self.shared(features)
        return_prediction = self.return_head(features).squeeze(-1)
        direction_logit = self.direction_head(features).squeeze(-1)
        return return_prediction, direction_logit


def to_tensor_dataset(features, future_return, future_up):
    return TensorDataset(
        torch.tensor(features, dtype = torch.float32),
        torch.tensor(future_return, dtype = torch.float32),
        torch.tensor(future_up, dtype = torch.float32),
    )


def evaluate_loss(model, data_loader, device, return_mean, return_std):
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_examples = 0

    model.eval()
    with torch.no_grad():
        for batch_features, batch_returns, batch_ups in data_loader:
            batch_features = batch_features.to(device)
            batch_returns = batch_returns.to(device)
            batch_ups = batch_ups.to(device)

            normalized_returns = (batch_returns - return_mean) / return_std
            predicted_returns, predicted_logits = model(batch_features)
            loss = mse_loss(predicted_returns, normalized_returns) + bce_loss(predicted_logits, batch_ups)
            batch_size = batch_features.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / max(total_examples, 1)


def train_model(train_features, train_returns, train_ups, validation_features, validation_returns, validation_ups, hidden_size, learning_rate, batch_size, epochs, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return_mean = float(train_returns.mean())
    return_std = float(train_returns.std())
    if return_std == 0:
        return_std = 1.0

    model = LSTMForecastModel(train_features.shape[-1], hidden_size, 0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(to_tensor_dataset(train_features, train_returns, train_ups), batch_size = batch_size, shuffle = False)
    validation_loader = DataLoader(to_tensor_dataset(validation_features, validation_returns, validation_ups), batch_size = batch_size, shuffle = False)

    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    remaining_patience = patience

    for _ in range(epochs):
        model.train()
        for batch_features, batch_returns, batch_ups in train_loader:
            batch_features = batch_features.to(device)
            batch_returns = batch_returns.to(device)
            batch_ups = batch_ups.to(device)

            normalized_returns = (batch_returns - return_mean) / return_std
            predicted_returns, predicted_logits = model(batch_features)
            loss = mse_loss(predicted_returns, normalized_returns) + bce_loss(predicted_logits, batch_ups)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        validation_loss = evaluate_loss(model, validation_loader, device, return_mean, return_std)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break

    model.load_state_dict(best_state)
    return model, device, return_mean, return_std


def predict_model(model, device, features, batch_size, return_mean, return_std):
    data_loader = DataLoader(TensorDataset(torch.tensor(features, dtype = torch.float32)), batch_size = batch_size, shuffle = False)
    predicted_returns = []
    predicted_probabilities = []

    model.eval()
    with torch.no_grad():
        for (batch_features,) in data_loader:
            batch_features = batch_features.to(device)
            predicted_return_batch, predicted_logit_batch = model(batch_features)
            predicted_returns.append((predicted_return_batch.cpu().numpy() * return_std) + return_mean)
            predicted_probabilities.append(torch.sigmoid(predicted_logit_batch).cpu().numpy())

    return np.concatenate(predicted_returns), np.concatenate(predicted_probabilities)


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
    parser = argparse.ArgumentParser(description = "Train a simple LSTM baseline for multi-day price forecasting.")
    parser.add_argument("--file", default = "./data/hourly/eth.csv", help = "CSV file with hourly candles.")
    parser.add_argument("--horizon-hours", type = int, default = 72, help = "Forecast horizon in hours.")
    parser.add_argument("--window-hours", type = int, default = 168, help = "How many past hours to expose to the LSTM.")
    parser.add_argument("--hidden-size", type = int, default = 64, help = "LSTM hidden size.")
    parser.add_argument("--epochs", type = int, default = 50, help = "Maximum training epochs.")
    parser.add_argument("--batch-size", type = int, default = 64, help = "Batch size.")
    parser.add_argument("--learning-rate", type = float, default = 1e-3, help = "Adam learning rate.")
    parser.add_argument("--patience", type = int, default = 8, help = "Early stopping patience measured in epochs.")
    parser.add_argument("--indices", nargs = 5, type = int, default = [1, 3, 4, 5, 6], help = "Column indices: date, open, high, low, close.")
    args = parser.parse_args()

    frame = load_price_frame(args.file, args.indices)
    frame = add_technicals(frame)
    feature_columns = build_feature_columns()
    future_return, future_up = build_targets(frame, args.horizon_hours)

    dataset = build_sequence_dataset(frame, feature_columns, future_return, future_up, args.window_hours)
    features = dataset["features"]
    future_return = dataset["future_return"]
    future_up = dataset["future_up"]

    train_slice, validation_slice, test_slice = chronological_split(len(features))

    scaled_features, _ = scale_sequence_features(features[train_slice], features)

    model, device, return_mean, return_std = train_model(
        scaled_features[train_slice],
        future_return[train_slice],
        future_up[train_slice],
        scaled_features[validation_slice],
        future_return[validation_slice],
        future_up[validation_slice],
        args.hidden_size,
        args.learning_rate,
        args.batch_size,
        args.epochs,
        args.patience,
    )

    for name, data_slice in [
        ("Train", train_slice),
        ("Validation", validation_slice),
        ("Test", test_slice),
    ]:
        slice_features = scaled_features[data_slice]
        slice_future_return = future_return[data_slice]
        slice_future_up = future_up[data_slice]
        predicted_return, predicted_up_probability = predict_model(
            model,
            device,
            slice_features,
            args.batch_size,
            return_mean,
            return_std,
        )
        summarize_predictions(name, slice_future_return, slice_future_up, predicted_return, predicted_up_probability)

    test_features = scaled_features[test_slice]
    test_predicted_return, test_predicted_probability = predict_model(
        model,
        device,
        test_features,
        args.batch_size,
        return_mean,
        return_std,
    )
    test_frame = pd.DataFrame({
        "date": dataset["date"][test_slice],
        "close": dataset["close"][test_slice],
        "future_return": dataset["future_return"][test_slice],
        "predicted_return": test_predicted_return,
        "probability_up": test_predicted_probability,
    })

    print("Latest test predictions:")
    print(test_frame.tail(10).to_string(index = False))


if __name__ == "__main__":
    main()