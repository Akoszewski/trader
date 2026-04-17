import argparse

import matplotlib.pyplot as plt
import numpy as np

from trader import readData


def prepare_signal(closes):
    log_prices = np.log(np.asarray(closes, dtype = np.float64))
    x_axis = np.arange(len(log_prices), dtype = np.float64)
    trend = np.polyval(np.polyfit(x_axis, log_prices, 1), x_axis)
    detrended = log_prices - trend
    window = np.hanning(len(detrended))
    return detrended * window


def compute_fft(signal, sample_spacing_hours):
    fft_values = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), d = sample_spacing_hours)
    magnitudes = np.abs(fft_values)
    return frequencies, magnitudes


def describe_dominant_periods(frequencies, magnitudes, top_k, min_period_hours, max_period_hours):
    non_zero = frequencies > 0
    filtered_frequencies = frequencies[non_zero]
    filtered_magnitudes = magnitudes[non_zero]

    if min_period_hours is not None:
        min_frequency = 1.0 / max_period_hours if max_period_hours is not None else 0.0
        max_frequency = 1.0 / min_period_hours
        period_mask = (filtered_frequencies >= min_frequency) & (filtered_frequencies <= max_frequency)
        filtered_frequencies = filtered_frequencies[period_mask]
        filtered_magnitudes = filtered_magnitudes[period_mask]

    if len(filtered_frequencies) == 0:
        return []

    top_indices = np.argsort(filtered_magnitudes)[-top_k:][::-1]
    periods = []
    for index in top_indices:
        periods.append({
            "period_hours": 1.0 / filtered_frequencies[index],
            "magnitude": filtered_magnitudes[index],
        })
    return periods


def plot_fft(file_path, indices, sample_spacing_hours, last_points, output_path, top_k, min_period_hours, max_period_hours):
    data = readData(file_path, indices)
    closes = np.asarray(data.closes, dtype = np.float64)
    dates = np.asarray(data.dates)

    if last_points is not None:
        closes = closes[-last_points:]
        dates = dates[-last_points:]

    if len(closes) < 16:
        raise ValueError("Need at least 16 samples to compute a meaningful FFT plot.")

    signal = prepare_signal(closes)
    frequencies, magnitudes = compute_fft(signal, sample_spacing_hours)

    non_zero = frequencies > 0
    frequencies = frequencies[non_zero]
    magnitudes = magnitudes[non_zero]
    periods = 1.0 / frequencies

    if max_period_hours is None:
        max_period_hours = max(len(closes) * sample_spacing_hours / 4.0, min_period_hours)

    period_mask = (periods >= min_period_hours) & (periods <= max_period_hours)
    periods = periods[period_mask]
    magnitudes = magnitudes[period_mask]
    filtered_frequencies = frequencies[period_mask]

    sort_order = np.argsort(periods)
    periods = periods[sort_order]
    magnitudes = magnitudes[sort_order]
    filtered_frequencies = filtered_frequencies[sort_order]

    dominant_periods = describe_dominant_periods(filtered_frequencies, magnitudes, top_k, min_period_hours, max_period_hours)

    figure, axes = plt.subplots(2, 1, figsize = (14, 9), constrained_layout = True)

    axes[0].plot(closes, color = "#0b7285", linewidth = 1.2)
    axes[0].set_title("Price series")
    axes[0].set_ylabel("Close")
    axes[0].grid(alpha = 0.25)

    tick_count = min(8, len(dates))
    tick_positions = np.linspace(0, len(dates) - 1, tick_count, dtype = int)
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(dates[tick_positions], rotation = 25, ha = "right")

    axes[1].plot(periods, magnitudes, color = "#c92a2a", linewidth = 1.2)
    axes[1].set_title("Fourier magnitude spectrum")
    axes[1].set_xlabel("Period [hours]")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(alpha = 0.25)
    axes[1].set_xscale("log")

    if len(periods) > 1:
        axes[1].set_xlim(periods.min(), periods.max())

    for item in dominant_periods[:3]:
        axes[1].axvline(item["period_hours"], color = "#f08c00", linestyle = "--", alpha = 0.5)
        axes[1].annotate(
            f"{item['period_hours']:.1f}h",
            xy = (item["period_hours"], item["magnitude"]),
            xytext = (5, 8),
            textcoords = "offset points",
            fontsize = 9,
            color = "#f08c00",
        )

    figure.suptitle(f"FFT of prices: {file_path}")
    figure.savefig(output_path, dpi = 160)
    plt.close(figure)

    print(f"Saved FFT chart to {output_path}")
    print("Dominant periods:")
    for item in dominant_periods:
        print(f"  {item['period_hours']:.2f} h | magnitude {item['magnitude']:.4f}")


def main():
    parser = argparse.ArgumentParser(description = "Plot price series together with its Fourier transform magnitude spectrum.")
    parser.add_argument("--file", default = "./data/hourly/eth.csv", help = "CSV file with OHLC candles.")
    parser.add_argument("--indices", nargs = 5, type = int, default = [1, 3, 4, 5, 6], help = "Column indices: date, open, high, low, close.")
    parser.add_argument("--sample-spacing-hours", type = float, default = 1.0, help = "Distance between samples in hours.")
    parser.add_argument("--last-points", type = int, default = 4096, help = "Use only the last N points for the plot.")
    parser.add_argument("--top-k", type = int, default = 5, help = "Number of dominant periods to print.")
    parser.add_argument("--min-period-hours", type = float, default = 4.0, help = "Ignore periods shorter than this in the spectrum plot.")
    parser.add_argument("--max-period-hours", type = float, default = None, help = "Ignore periods longer than this in the spectrum plot.")
    parser.add_argument("--output", default = "./fft_price.png", help = "Where to save the generated chart.")
    args = parser.parse_args()

    plot_fft(args.file, args.indices, args.sample_spacing_hours, args.last_points, args.output, args.top_k, args.min_period_hours, args.max_period_hours)


if __name__ == "__main__":
    main()