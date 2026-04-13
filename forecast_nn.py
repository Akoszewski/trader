import argparse
import copy
import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
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


def add_fourier_features(frame, fft_window_hours):
    frame = frame.copy()
    log_returns = frame["log_return_1h"].to_numpy(dtype = np.float64)

    dominant_period = np.full(len(frame), np.nan, dtype = np.float32)
    spectral_entropy = np.full(len(frame), np.nan, dtype = np.float32)
    short_band_energy = np.full(len(frame), np.nan, dtype = np.float32)
    medium_band_energy = np.full(len(frame), np.nan, dtype = np.float32)
    long_band_energy = np.full(len(frame), np.nan, dtype = np.float32)

    hann_window = np.hanning(fft_window_hours)
    frequency_axis = np.fft.rfftfreq(fft_window_hours, d = 1.0)
    non_zero_mask = frequency_axis > 0
    frequency_axis = frequency_axis[non_zero_mask]

    short_band_mask = (frequency_axis >= 1.0 / 12.0) & (frequency_axis <= 1.0 / 4.0)
    medium_band_mask = (frequency_axis >= 1.0 / 48.0) & (frequency_axis < 1.0 / 12.0)
    long_band_mask = (frequency_axis >= 1.0 / 168.0) & (frequency_axis < 1.0 / 48.0)

    for end_idx in range(fft_window_hours - 1, len(frame)):
        start_idx = end_idx - fft_window_hours + 1
        window = log_returns[start_idx:end_idx + 1]
        if np.isnan(window).any():
            continue

        centered_window = window - window.mean()
        windowed_signal = centered_window * hann_window
        power_spectrum = np.abs(np.fft.rfft(windowed_signal)) ** 2
        power_spectrum = power_spectrum[non_zero_mask]
        total_power = power_spectrum.sum()
        if total_power <= 0:
            continue

        dominant_frequency = frequency_axis[np.argmax(power_spectrum)]
        dominant_period[end_idx] = 1.0 / dominant_frequency

        normalized_power = power_spectrum / total_power
        spectral_entropy[end_idx] = float(
            -(normalized_power * np.log(normalized_power + 1e-12)).sum() / np.log(len(normalized_power))
        )
        short_band_energy[end_idx] = float(power_spectrum[short_band_mask].sum() / total_power)
        medium_band_energy[end_idx] = float(power_spectrum[medium_band_mask].sum() / total_power)
        long_band_energy[end_idx] = float(power_spectrum[long_band_mask].sum() / total_power)

    frame["fft_dominant_period"] = dominant_period
    frame["fft_spectral_entropy"] = spectral_entropy
    frame["fft_energy_4_12h"] = short_band_energy
    frame["fft_energy_12_48h"] = medium_band_energy
    frame["fft_energy_48_168h"] = long_band_energy
    return frame


def add_technicals(frame, fft_window_hours = 168, use_fft_features = True):
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

    if use_fft_features:
        return add_fourier_features(frame, fft_window_hours)
    return frame


def build_feature_columns(use_fft_features = True, feature_set = "full"):
    feature_sets = {
        "full": [
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
        ],
        "compact": [
            "log_return_1h",
            "range_pct",
            "return_6h",
            "return_24h",
            "return_72h",
            "volatility_24h",
            "volatility_72h",
            "close_over_ema_24",
            "close_over_ema_100",
            "rsi_14",
            "macd_hist",
        ],
        "returns-only": [
            "log_return_1h",
            "range_pct",
            "return_6h",
            "return_24h",
            "return_72h",
            "return_168h",
            "volatility_24h",
            "volatility_72h",
            "volatility_168h",
        ],
    }

    if feature_set not in feature_sets:
        raise ValueError(f"Unknown feature set: {feature_set}")

    feature_columns = list(feature_sets[feature_set])

    if use_fft_features:
        feature_columns.extend([
            "fft_dominant_period",
            "fft_spectral_entropy",
            "fft_energy_4_12h",
            "fft_energy_12_48h",
            "fft_energy_48_168h",
        ])

    return feature_columns


def build_targets(frame, horizon_hours, target_return):
    future_close = frame["close"].shift(-horizon_hours)
    future_return = future_close / frame["close"] - 1.0
    closes = frame["close"].to_numpy(dtype = np.float32)
    highs = frame["high"].to_numpy(dtype = np.float32)
    lows = frame["low"].to_numpy(dtype = np.float32)

    future_max_return_values = np.full(len(frame), np.nan, dtype = np.float32)
    future_min_return_values = np.full(len(frame), np.nan, dtype = np.float32)
    target_hit_values = np.full(len(frame), np.nan, dtype = np.float32)
    trade_return_values = np.full(len(frame), np.nan, dtype = np.float32)

    for idx in range(len(frame)):
        end_idx = idx + horizon_hours
        if end_idx >= len(frame):
            continue

        current_close = closes[idx]
        take_profit_price = current_close * (1.0 + target_return)
        stop_loss_price = current_close * (1.0 - target_return)
        future_highs = highs[idx + 1:end_idx + 1]
        future_lows = lows[idx + 1:end_idx + 1]

        future_max_return_values[idx] = np.max(future_highs) / current_close - 1.0
        future_min_return_values[idx] = np.min(future_lows) / current_close - 1.0

        target_value = np.nan
        for future_high, future_low in zip(future_highs, future_lows):
            if future_low <= stop_loss_price and future_high >= take_profit_price:
                target_value = np.nan
                break
            if future_low <= stop_loss_price:
                target_value = 0.0
                trade_return_values[idx] = -target_return
                break
            if future_high >= take_profit_price:
                target_value = 1.0
                trade_return_values[idx] = target_return
                break

        target_hit_values[idx] = target_value

    future_max_return = pd.Series(future_max_return_values, index = frame.index)
    future_min_return = pd.Series(future_min_return_values, index = frame.index)
    target_hit = pd.Series(target_hit_values, index = frame.index)
    trade_return = pd.Series(trade_return_values, index = frame.index)
    return future_return, future_max_return, future_min_return, target_hit, trade_return


def chronological_split(num_rows, train_ratio = 0.7, validation_ratio = 0.15):
    train_end = math.floor(num_rows * train_ratio)
    validation_end = math.floor(num_rows * (train_ratio + validation_ratio))
    return slice(0, train_end), slice(train_end, validation_end), slice(validation_end, num_rows)


def build_sequence_dataset(frame, feature_columns, future_return, future_max_return, future_min_return, target_hit, trade_return, window_hours):
    feature_matrix = frame[feature_columns].to_numpy(dtype=np.float32)
    future_return_values = future_return.to_numpy(dtype=np.float32)
    future_max_return_values = future_max_return.to_numpy(dtype=np.float32)
    future_min_return_values = future_min_return.to_numpy(dtype=np.float32)
    target_hit_values = target_hit.to_numpy(dtype=np.float32)
    trade_return_values = trade_return.to_numpy(dtype=np.float32)
    close_values = frame["close"].to_numpy(dtype=np.float32)
    date_values = frame["date"].to_numpy()

    sequences = []
    target_hits = []
    realized_returns = []
    realized_max_returns = []
    realized_min_returns = []
    realized_trade_returns = []
    current_closes = []
    current_dates = []

    for end_idx in range(window_hours - 1, len(frame)):
        start_idx = end_idx - window_hours + 1
        window = feature_matrix[start_idx:end_idx + 1]
        if np.isnan(window).any():
            continue

        target_return = future_return_values[end_idx]
        target_max_return = future_max_return_values[end_idx]
        target_min_return = future_min_return_values[end_idx]
        target_value = target_hit_values[end_idx]
        trade_value = trade_return_values[end_idx]
        if np.isnan(target_return) or np.isnan(target_max_return) or np.isnan(target_min_return) or np.isnan(target_value) or np.isnan(trade_value):
            continue

        sequences.append(window)
        target_hits.append(target_value)
        realized_returns.append(target_return)
        realized_max_returns.append(target_max_return)
        realized_min_returns.append(target_min_return)
        realized_trade_returns.append(trade_value)
        current_closes.append(close_values[end_idx])
        current_dates.append(date_values[end_idx])

    return {
        "features": np.asarray(sequences, dtype=np.float32),
        "target_hit": np.asarray(target_hits, dtype=np.float32),
        "future_return": np.asarray(realized_returns, dtype=np.float32),
        "future_max_return": np.asarray(realized_max_returns, dtype=np.float32),
        "future_min_return": np.asarray(realized_min_returns, dtype=np.float32),
        "trade_return": np.asarray(realized_trade_returns, dtype=np.float32),
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
        self.hit_head = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        _, (hidden_state, _) = self.lstm(inputs)
        features = self.dropout(hidden_state[-1])
        features = self.shared(features)
        hit_logit = self.hit_head(features).squeeze(-1)
        return hit_logit


def to_tensor_dataset(features, target_hit):
    return TensorDataset(
        torch.tensor(features, dtype = torch.float32),
        torch.tensor(target_hit, dtype = torch.float32),
    )


def evaluate_loss(model, data_loader, device):
    bce_loss = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_examples = 0

    model.eval()
    with torch.no_grad():
        for batch_features, batch_target_hit in data_loader:
            batch_features = batch_features.to(device)
            batch_target_hit = batch_target_hit.to(device)

            predicted_logits = model(batch_features)
            loss = bce_loss(predicted_logits, batch_target_hit)
            batch_size = batch_features.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / max(total_examples, 1)


def train_model(train_features, train_target_hit, validation_features, validation_target_hit, hidden_size, learning_rate, batch_size, epochs, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMForecastModel(train_features.shape[-1], hidden_size, 0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    bce_loss = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(to_tensor_dataset(train_features, train_target_hit), batch_size = batch_size, shuffle = False)
    validation_loader = DataLoader(to_tensor_dataset(validation_features, validation_target_hit), batch_size = batch_size, shuffle = False)

    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    remaining_patience = patience

    for _ in range(epochs):
        model.train()
        for batch_features, batch_target_hit in train_loader:
            batch_features = batch_features.to(device)
            batch_target_hit = batch_target_hit.to(device)

            predicted_logits = model(batch_features)
            loss = bce_loss(predicted_logits, batch_target_hit)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        validation_loss = evaluate_loss(model, validation_loader, device)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break

    model.load_state_dict(best_state)
    return model, device


def predict_model(model, device, features, batch_size):
    data_loader = DataLoader(TensorDataset(torch.tensor(features, dtype = torch.float32)), batch_size = batch_size, shuffle = False)
    predicted_probabilities = []

    model.eval()
    with torch.no_grad():
        for (batch_features,) in data_loader:
            batch_features = batch_features.to(device)
            predicted_logits = model(batch_features)
            predicted_probabilities.append(torch.sigmoid(predicted_logits).cpu().numpy())

    return np.concatenate(predicted_probabilities)


def summarize_predictions(name, target_hit, future_return, future_max_return, future_min_return, trade_return, predicted_probability, target_return, probability_threshold):
    predicted_hit = (predicted_probability >= probability_threshold).astype(int)
    direction_accuracy = accuracy_score(target_hit, predicted_hit)
    precision = precision_score(target_hit, predicted_hit, zero_division = 0)
    recall = recall_score(target_hit, predicted_hit, zero_division = 0)

    summary = pd.DataFrame({
        "target_hit": target_hit,
        "future_return": future_return,
        "future_max_return": future_max_return,
        "future_min_return": future_min_return,
        "trade_return": trade_return,
        "predicted_probability": predicted_probability,
    })
    top_signals = summary.sort_values("predicted_probability", ascending = False).head(20)
    predicted_signals = summary.loc[predicted_hit == 1]
    top_signal_hit_rate = top_signals["target_hit"].mean()
    top_signal_trade_return = top_signals["trade_return"].mean()
    predicted_signal_hit_rate = predicted_signals["target_hit"].mean() if len(predicted_signals) > 0 else float("nan")
    predicted_signal_trade_return = predicted_signals["trade_return"].mean() if len(predicted_signals) > 0 else float("nan")
    predicted_positive_rate = predicted_hit.mean()
    base_positive_rate = summary["target_hit"].mean()

    print(f"{name} metrics:")
    print(f"  Target: hit +{round(target_return * 100, 2)}% before -{round(target_return * 100, 2)}% within horizon (neutral cases excluded)")
    print(f"  Accuracy: {round(direction_accuracy * 100, 2)}%")
    print(f"  Precision: {round(precision * 100, 2)}%")
    print(f"  Recall: {round(recall * 100, 2)}%")
    print(f"  Base hit rate: {round(base_positive_rate * 100, 2)}%")
    print(f"  Predicted positive rate: {round(predicted_positive_rate * 100, 2)}%")
    print(f"  Top 20 hit rate: {round(top_signal_hit_rate * 100, 2)}%")
    print(f"  Top 20 average trade return: {round(top_signal_trade_return * 100, 4)}%")
    if len(predicted_signals) > 0:
        print(f"  Predicted signals hit rate: {round(predicted_signal_hit_rate * 100, 2)}%")
        print(f"  Predicted signals average trade return: {round(predicted_signal_trade_return * 100, 4)}%")
    else:
        print("  Predicted signals hit rate: n/a")
        print("  Predicted signals average trade return: n/a")
    print("")


def main():
    parser = argparse.ArgumentParser(description = "Train an LSTM to predict whether price will hit take-profit before stop-loss within a future horizon.")
    parser.add_argument("--file", default = "./data/hourly/eth.csv", help = "CSV file with hourly candles.")
    parser.add_argument("--horizon-hours", type = int, default = 24, help = "Forecast horizon in hours.")
    parser.add_argument("--window-hours", type = int, default = 168, help = "How many past hours to expose to the LSTM.")
    parser.add_argument("--target-return", type = float, default = 0.01, help = "Target gain threshold, e.g. 0.01 means +1%% within the horizon.")
    parser.add_argument("--probability-threshold", type = float, default = 0.6, help = "Probability threshold used for converting model output into a positive prediction.")
    parser.add_argument("--fft-window-hours", type = int, default = 168, help = "Rolling window used to compute Fourier features from log returns.")
    parser.add_argument("--disable-fft-features", action = "store_true", help = "Disable Fourier-based rolling features for an A/B comparison.")
    parser.add_argument("--feature-set", choices = ["full", "compact", "returns-only"], default = "full", help = "Choose which base feature set to use.")
    parser.add_argument("--hidden-size", type = int, default = 64, help = "LSTM hidden size.")
    parser.add_argument("--epochs", type = int, default = 50, help = "Maximum training epochs.")
    parser.add_argument("--batch-size", type = int, default = 64, help = "Batch size.")
    parser.add_argument("--learning-rate", type = float, default = 1e-3, help = "Adam learning rate.")
    parser.add_argument("--patience", type = int, default = 8, help = "Early stopping patience measured in epochs.")
    parser.add_argument("--indices", nargs = 5, type = int, default = [1, 3, 4, 5, 6], help = "Column indices: date, open, high, low, close.")
    args = parser.parse_args()

    frame = load_price_frame(args.file, args.indices)
    use_fft_features = not args.disable_fft_features
    frame = add_technicals(frame, args.fft_window_hours, use_fft_features)
    feature_columns = build_feature_columns(use_fft_features, args.feature_set)
    print(f"Using FFT features: {use_fft_features}")
    print(f"Feature set: {args.feature_set} ({len(feature_columns)} features)")
    future_return, future_max_return, future_min_return, target_hit, trade_return = build_targets(frame, args.horizon_hours, args.target_return)

    labeled_examples = int(target_hit.notna().sum())
    total_examples = len(target_hit)
    print(f"Labeled examples: {labeled_examples}/{total_examples} ({round(100.0 * labeled_examples / max(total_examples, 1), 2)}%)")

    dataset = build_sequence_dataset(frame, feature_columns, future_return, future_max_return, future_min_return, target_hit, trade_return, args.window_hours)
    features = dataset["features"]
    future_return = dataset["future_return"]
    future_max_return = dataset["future_max_return"]
    future_min_return = dataset["future_min_return"]
    target_hit = dataset["target_hit"]
    trade_return = dataset["trade_return"]

    train_slice, validation_slice, test_slice = chronological_split(len(features))

    scaled_features, _ = scale_sequence_features(features[train_slice], features)

    model, device = train_model(
        scaled_features[train_slice],
        target_hit[train_slice],
        scaled_features[validation_slice],
        target_hit[validation_slice],
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
        slice_target_hit = target_hit[data_slice]
        slice_future_return = future_return[data_slice]
        slice_future_max_return = future_max_return[data_slice]
        slice_future_min_return = future_min_return[data_slice]
        slice_trade_return = trade_return[data_slice]
        predicted_probability = predict_model(model, device, slice_features, args.batch_size)
        summarize_predictions(name, slice_target_hit, slice_future_return, slice_future_max_return, slice_future_min_return, slice_trade_return, predicted_probability, args.target_return, args.probability_threshold)

    test_features = scaled_features[test_slice]
    test_predicted_probability = predict_model(model, device, test_features, args.batch_size)
    test_frame = pd.DataFrame({
        "date": dataset["date"][test_slice],
        "close": dataset["close"][test_slice],
        "future_return": dataset["future_return"][test_slice],
        "future_max_return": dataset["future_max_return"][test_slice],
        "future_min_return": dataset["future_min_return"][test_slice],
        "trade_return": dataset["trade_return"][test_slice],
        "target_hit": dataset["target_hit"][test_slice],
        "probability_hit": test_predicted_probability,
    })

    print("Latest test predictions:")
    print(test_frame.tail(10).to_string(index = False))


if __name__ == "__main__":
    main()