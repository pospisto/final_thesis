'''
import math
import sys

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import argparse
import librosa as li
import os
import csv


def label_time_series(time_series, sampling_rate, csv_path, n_fft=1024, noverlap=256):
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
        """ Redraw the points and line on the spectrogram """
        ax2.clear()
        ax2.specgram(time_series, NFFT=n_fft, Fs=sampling_rate, noverlap=noverlap, cmap='magma')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        if points:
            ax2.plot(points, values, marker='o', linestyle='-', color='cyan', linewidth=5)
        plt.draw()

    # Load CSV file if provided
    if csv_path is not None:
        with open(csv_path, newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                x = float(row['time'])
                y = float(row['revolutions per minute'])
                points.append(x)
                values.append(y)
            maxVal = math.ceil(max(values)/1000)*1000
            for i in range(len(values)):
                values[i] = ((int(sampling_rate)/2)/maxVal) * values[i]
            if len(time_series) < points[-1] * int(sampling_rate):
                del points[-1]
                del values[-1]
        draw_line()  # Draw loaded points


    def onclick(event):
        nonlocal points, values
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        if event.button == 1 and (len(points) == 0 or x > points[-1]):
            # Left mouse button: add a new point
            points.append(x)
            values.append(y)
            draw_line()
        elif event.button == 3:
            # Right mouse button: delete the last placed point
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

def load_audio(path, fs, norm=True):
    x = li.load(path, sr=fs, mono=True)[0]
    if norm:
        x = li.util.normalize(x)
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', help='Directory of the training sound', default='training_sounds')
    parser.add_argument('--audio_name', help='Name of the training sound', default='metal')
    parser.add_argument('--output_directory', help='Where to put the file', default="labels_train")
    parser.add_argument('--feature_name', help='Name of the feature for the output file', default="control_1")
    parser.add_argument('--sampling_rate', type=int, help='Fs of the sounds', default=44100)
    parser.add_argument('--csv_path', help='Directory of the training sound', default= None)



    config = parser.parse_args()
    audio_in = f'{config.audio_path}/{config.audio_name}.wav'
    audio = load_audio(path=audio_in, fs=config.sampling_rate)
    audio = audio+1e-8
    #create a folder if it doesn't exist
    if not os.path.exists(f'{config.output_directory}/{config.audio_name}'):
        os.makedirs(f'{config.output_directory}/{config.audio_name}')

    # Label the time series with mouse clicks and interpolate between points
    y_values = label_time_series(audio, config.sampling_rate, config.csv_path)

    fig, ax = plt.subplots()
    ax.set_title(f'Control parameter for {config.audio_name}')
    ax.plot(y_values)
    ax.legend()
    plt.show()
    #save the file in the output directory
    np.save(f'{config.output_directory}/{config.audio_name}/{config.feature_name}', y_values)'''
''

import math
import sys
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import argparse
import librosa as li
import os
import csv


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
            '''
            maxVal = math.ceil(max(values) / 1000) * 1000
            for i in range(len(values)):
                values[i] = ((int(sampling_rate) / 2) / maxVal) * values[i]
            '''
            # Univerzální škálování
            min_val = min(values)
            max_val = max(values)
            target_min = 0  # Minimální hodnota spektrogramu (frekvence)
            target_max = sampling_rate / 2  # Maximální hodnota spektrogramu (frekvence)

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


def load_audio(path, fs, norm=True):
    x = li.load(path, sr=fs, mono=True)[0]
    if norm:
        x = li.util.normalize(x)
    return x


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', help='Directory of the training sound', default='training_sounds')
    parser.add_argument('--audio_name', help='Name of the training sound', default='metal')
    parser.add_argument('--output_directory', help='Where to put the file', default="labels_train")
    parser.add_argument('--features', type=str, help='Comma-separated list of features to extract', required=True)
    parser.add_argument('--sampling_rate', type=int, help='Fs of the sounds', default=44100)
    parser.add_argument('--csv_path', help='Directory of the training sound', default=None)

    config = parser.parse_args()
    parameters = [param.strip() for param in config.features.split(',')]

    audio_in = f'{config.audio_path}/{config.audio_name}.wav'
    audio = load_audio(path=audio_in, fs=config.sampling_rate)
    audio = audio + 1e-8

    if not os.path.exists(f'{config.output_directory}/{config.audio_name}'):
        os.makedirs(f'{config.output_directory}/{config.audio_name}')

    feature_list = config.features.split(',')

    all_y_values = {}  # Uložíme y_values pro každou feature

    for i, feature in enumerate(feature_list):
        feature = feature.strip()
        print(f"Processing feature: {feature}")
        y_values = label_time_series(audio, config.sampling_rate, config.csv_path, feature)

        all_y_values[feature] = y_values  # Uložíme y_values do dictionary

        safe_feature_name = feature.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        np.save(f'{config.output_directory}/{config.audio_name}/{config.audio_name}_{safe_feature_name}', y_values)

    # Vykreslení všech y_values do jednoho grafu
    plt.figure()
    plt.title("Všechny y_values pro audio: " + config.audio_name)
    for feature, y_values in all_y_values.items():
        plt.plot(y_values, label=feature)
    plt.legend()
    plt.show()
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', help='Directory of the training sounds', required=True)
    parser.add_argument('--csv_path', help='Directory containing CSV files', required=True)
    parser.add_argument('--output_directory', help='Where to put the files', default="labels_train")
    parser.add_argument('--features', type=str, help='Comma-separated list of features to extract', required=True)
    parser.add_argument('--sampling_rate', type=int, help='Fs of the sounds', default=44100)

    config = parser.parse_args()
    parameters = [param.strip() for param in config.features.split(',')]

    audio_files = sorted([f for f in os.listdir(config.audio_path) if f.endswith('.wav')])
    csv_files = sorted([f for f in os.listdir(config.csv_path) if f.endswith('.csv')])

    if len(audio_files) != len(csv_files):
        print("Počet audio souborů a CSV souborů není stejný!", file=sys.stderr)
        sys.exit(1)

    for audio_file, csv_file in zip(audio_files, csv_files):
        audio_in = os.path.join(config.audio_path, audio_file)
        csv_in = os.path.join(config.csv_path, csv_file)
        print(f"Processing: {audio_file} with {csv_file}")

        audio = load_audio(path=audio_in, fs=config.sampling_rate)
        audio = audio + 1e-8

        all_y_values = {}

        for feature in parameters:
            print(f"Processing feature: {feature}")
            y_values = label_time_series(audio, config.sampling_rate, csv_in, feature)
            all_y_values[feature] = y_values

            safe_feature_name = feature.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            os.makedirs(config.output_directory, exist_ok=True)
            file = os.path.splitext(os.path.basename(audio_file))[0]
            np.save(os.path.join(config.output_directory, f'{file}_{safe_feature_name}.npy'), y_values)

        plt.figure()
        plt.title(f"Všechny y_values pro: {audio_file}")
        for feature, y_values in all_y_values.items():
            plt.plot(y_values, label=feature)
        plt.legend()
        plt.show()




