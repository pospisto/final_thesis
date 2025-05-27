
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import argparse
import librosa as li
import os
import csv

def label_time_series(n,csv_path):
    n = int(n)
    x_values = range(n)
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.MaxNLocator(16))

    ax.plot(x_values, [0] * n)
    ax.set_ylim([0, 1])
    plt.grid()
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')

    points = []
    values = []

    if csv_path is not None:
        with open(csv_path, newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                x = float(row['time'])
                y = (1/8000)*float(row['revolutions per minute'])
                points.append(x)
                values.append(y)

    def onclick(event):
        nonlocal points, values
        x, y = event.xdata, event.ydata
        if event.button == 1 and (len(points) == 0 or x > points[-1]):
            # Left mouse button: add a new point
            points.append(x)
            values.append(y)
            ax.plot(points, values, 'ro-')
            plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    print("Načtené body (čas):", points)
    print("Načtené hodnoty (RPM):", values)

    x, y = zip(*sorted(zip(points, values)))
    #new_x = np.linspace(0, n-1, num=n)
    duration = max(points) if points else n
    new_x = np.linspace(0, duration, num=n)
    new_y = np.interp(new_x, x, y)
    new_y = (new_y - np.min(new_y)) / (np.max(new_y) - np.min(new_y))
    return new_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', help='Number of samples to generate', default=2**14)
    parser.add_argument('--output_directory', help='Where to put the file', default="inference_controls/metal")
    parser.add_argument('--feature_name', help='Name of the feature for the output file', default="control_metal_1")
    parser.add_argument('--csv_path', help='Directory of the training sound', default=None)

    config = parser.parse_args()
    if not os.path.exists(config.output_directory):
        os.makedirs(config.output_directory)

    y_values = label_time_series(config.n_samples,config.csv_path)

    fig, ax = plt.subplots()
    ax.plot(y_values)
    ax.set_title(f'Inference control parameter for {config.feature_name}')
    plt.ylim([0, 1])
    plt.show()
    #save the file in the output directory
    np.save(f'{config.output_directory}/{config.feature_name}', y_values)