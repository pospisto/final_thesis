"""
import yaml
import pathlib
import librosa as li
from ddsp.core import extract_loudness, extract_pitch
import numpy as np
from tqdm import tqdm
from os import makedirs, path
import argparse
import torch


def get_files(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def preprocess(f, sampling_rate, block_size, signal_length, oneshot, **kwargs):
    x, sr = li.load(f, sr=sampling_rate)
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)

    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)

    return x, pitch, loudness


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        self.signals = np.load(path.join(out_dir, "signals.npy"))
        self.pitchs = np.load(path.join(out_dir, "pitchs.npy"))
        self.loudness = np.load(path.join(out_dir, "loudness.npy"))

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signals[idx])
        p = torch.from_numpy(self.pitchs[idx])
        l = torch.from_numpy(self.loudness[idx])
        return s, p, l


def update_config(config_path, dataset_path, preproc_path, signal_length):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    config["data"]["data_location"] = dataset_path
    config["preprocess"]["out_dir"] = preproc_path
    config["preprocess"]["signal_length"] = signal_length

    with open(config_path, "w") as config_file:
        yaml.safe_dump(config, config_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--preproc_path", type=str, required=True, help="Path to preprocessed output directory")
    parser.add_argument("--signal_length", type=int, required=True, help="Signal length for preprocessing")
    args = parser.parse_args()

    config_path = "./config.yaml"
    update_config(config_path, args.dataset_path, args.preproc_path, args.signal_length)

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    files = get_files(**config["data"])
    pb = tqdm(files)

    signals = []
    pitchs = []
    loudness = []

    for f in pb:
        pb.set_description(str(f))
        x, p, l = preprocess(f, **config["preprocess"])
        signals.append(x)
        pitchs.append(p)
        loudness.append(l)

    signals = np.concatenate(signals, 0).astype(np.float32)
    pitchs = np.concatenate(pitchs, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)

    out_dir = config["preprocess"]["out_dir"]
    makedirs(out_dir, exist_ok=True)

    np.save(path.join(out_dir, "signals.npy"), signals)
    np.save(path.join(out_dir, "pitchs.npy"), pitchs)
    np.save(path.join(out_dir, "loudness.npy"), loudness)


if __name__ == "__main__":
    main()
"""


import yaml
import pathlib
import librosa as li
from ddsp.core import extract_loudness, extract_pitch
import numpy as np
from tqdm import tqdm
import numpy as np
from os import makedirs, path
import torch
import matplotlib
import argparse
import os

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import interp1d
from scipy.io import wavfile


def get_files(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def preprocess(x, sampling_rate, block_size, **kwargs):
    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)
    return pitch, loudness


def label_time_series(time_series, sampling_rate, csv_path, parameter, n_fft=1024, noverlap=256):
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
    ax1.plot(time_series, c='salmon')
    ax1.set_xlabel('Time (samples)')
    ax1.set_ylabel('Amplitude')
    ax1.grid()
    ax1.set_xlim(left=0, right=len(time_series))
    Pxx, freqs, bins, im = ax2.specgram(time_series, NFFT=n_fft, Fs=sampling_rate, noverlap=noverlap, cmap='magma')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    points = []
    values = []

    def draw_line():
        ax2.clear()
        ax2.specgram(time_series, NFFT=n_fft, Fs=sampling_rate, noverlap=noverlap, cmap='magma')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        if points:
            ax2.plot(points, values, marker='o', linestyle='-', color='cyan', linewidth=5)
        plt.draw()

    if csv_path is not None:
        with open(csv_path, newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                x = float(row['time'])
                y = float(row[parameter])
                points.append(x)
                values.append(y)
            if parameter != 'pitch (radians)':
                min_val = min(values)
                max_val = max(values)
                target_min = 0
                target_max = sampling_rate / 2
                scaled_values = []
                for val in values:
                    scaled_val = ((val - min_val) / (max_val - min_val)) * (target_max - target_min) + target_min
                    scaled_values.append(scaled_val)
                values = scaled_values
                if len(time_series) < points[-1] * int(sampling_rate):
                    del points[-1]
                    del values[-1]
        draw_line()

    def onclick(event):
        nonlocal points, values
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        if event.button == 1 and (len(points) == 0 or x > points[-1]):
            points.append(x)
            values.append(y)
            draw_line()
        elif event.button == 3:
            if points:
                points.pop()
                values.pop()
                draw_line()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    if not points or not values:
        return 0
    x, y = zip(*sorted(zip(points, values)))
    duration = len(time_series) / sampling_rate
    new_x = np.linspace(0, duration, num=len(time_series))
    new_y = np.interp(new_x, x, y, left=np.nan, right=np.nan)
    mask = np.isnan(new_y)
    new_y[mask] = np.interp(new_x[mask], new_x[~mask], new_y[~mask])
    new_y = (new_y - np.nanmin(new_y)) / (np.nanmax(new_y) - np.nanmin(new_y))
    return new_y


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir, features):
        super().__init__()
        self.signals = np.load(os.path.join(out_dir, "signals.npy"))
        self.pitchs = np.load(os.path.join(out_dir, "pitchs.npy"))
        self.loudness = np.load(os.path.join(out_dir, "loudness.npy"))

        self.features = {}
        for feature in features:
            feature_path = os.path.join(out_dir, f"{feature}.npy")
            if os.path.exists(feature_path):
                self.features[feature] = np.load(feature_path)
            else:
                print(f"Warning: Feature file {feature}.npy not found in {out_dir}. Skipping.")

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signals[idx])
        p = torch.from_numpy(self.pitchs[idx])
        l = torch.from_numpy(self.loudness[idx])

        if self.features:
            f_values = {feature: torch.from_numpy(self.features[feature][idx]) for feature in self.features}
        else:
            f_values = {}  # or `None` if you'd prefer

        return s, p, l, f_values


def update_config(config_path, dataset_path, preproc_path, signal_length):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    config["data"]["data_location"] = dataset_path
    config["preprocess"]["out_dir"] = preproc_path
    config["preprocess"]["signal_length"] = signal_length

    with open(config_path, "w") as config_file:
        yaml.safe_dump(config, config_file)


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio files.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--preproc_path", type=str, required=True, help="Path to preprocessed output directory")
    parser.add_argument("--signal_length", type=int, required=True, help="Signal length for preprocessing")
    parser.add_argument("--features", default=None, help="Comma-separated list of features to process.")
    parser.add_argument("--csv_path", default=None, help="Path to the directory containing CSV files.")
    parser.add_argument("--config", default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--ddsp_params", default="pitch,loudness",
                        help="Comma-separated list of ddsp parameters to process.")
    args = parser.parse_args()

    update_config(args.config, args.dataset_path, args.preproc_path, args.signal_length)

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    audio_files = get_files(**config["data"])
    sampling_rate = config['preprocess']['sampling_rate']
    signal_length = config['preprocess']['signal_length']
    block_size = config['preprocess']['block_size']
    out_dir = config["preprocess"]["out_dir"]

    makedirs(out_dir, exist_ok=True)

    # Inicializace datových struktur
    signals = []
    pitchs = []
    loudness = []

    # Parametry
    ddsp_params = [p.strip() for p in args.ddsp_params.split(",")] if args.ddsp_params else []
    features_list = [f.strip() for f in args.features.split(",")] if args.features else []
    features = {param: [] for param in features_list}
    interpolated_feat = []

    # CSV soubory pro features
    if features_list:
        if args.csv_path is None:
            raise ValueError("Pokud jsou zadané features, musí být zadán také --csv_path.")
        csv_files = sorted([f for f in os.listdir(args.csv_path) if f.endswith('.csv')])
        audio_files = sorted(audio_files, key=lambda x: os.path.basename(x))

        if len(csv_files) != len(audio_files):
            raise ValueError("Počet CSV souborů se musí shodovat s počtem audio souborů.")
    else:
        csv_files = [None] * len(audio_files)  # Placeholdery

    for csv_file, audio_file in tqdm(zip(csv_files, audio_files), total=len(audio_files), desc="Processing files"):
        x, sr = li.load(audio_file, sr=sampling_rate)
        x = x + 1e-8  # noise floor
        N = (signal_length - len(x) % signal_length) % signal_length
        x = np.pad(x, (0, N))
        x_segments = x.reshape(-1, signal_length)

        # DDSP pitch/loudness
        for segment in x_segments:
            p, l = preprocess(segment, sampling_rate, block_size)
            signals.append(segment.astype(np.float32))
            pitchs.append(p.astype(np.float32))
            loudness.append(l.astype(np.float32))
        # Features – jen pokud jsou zadané

        if features_list:
            csv_file_path = os.path.join(args.csv_path, csv_file)
            for param in features_list:
                y_values = label_time_series(x, sampling_rate, csv_file_path, param)
                y_segments = np.array_split(y_values, x_segments.shape[0])
                interpolated_segments = []
                for segment in y_segments:
                    segment_tensor = torch.tensor(segment.astype(np.float32)).unsqueeze(0).unsqueeze(
                        1)  # Přidáme batch a channel dimenzi [1, 1, délka_segmentu]
                    interpolated_tensor = torch.nn.functional.interpolate(segment_tensor, size=len(pitchs[0]),
                                                                          mode='linear', align_corners=False)
                    interpolated_segments.append(
                        interpolated_tensor.squeeze().numpy())  # Odstraníme přidané dimenze a převedeme zpět na NumPy
                features[param].extend(interpolated_segments)

    # Uložení
    signals = np.stack(signals)
    np.save(path.join(out_dir, "signals.npy"), signals)

    if 'pitch' in ddsp_params:
        np.save(path.join(out_dir, "pitchs.npy"), np.stack(pitchs))
    if 'loudness' in ddsp_params:
        np.save(path.join(out_dir, "loudness.npy"), np.stack(loudness))

    for param, feature_values in features.items():
        if feature_values:  # jen pokud máme něco k uložení
            safe_param = param.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            np.save(path.join(out_dir, f"{safe_param}.npy"), np.stack(feature_values))


if __name__ == "__main__":
    main()


