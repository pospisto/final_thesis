"""
import subprocess

DATASET_PATH = "./audio_dataset"
DDSP_PREPROCESSED = "./preprocessed/DDSP"

DDSP_enviroment = 'C:/Users/pospa/.conda/envs/ddsp_pd/python.exe'

print(" Spouštím preprocessing dat pro DDSP...")
subprocess.run(["python", "./ddsp/preprocess.py")
"""

"""
import subprocess
import os
import multiprocessing

DATASET_PATH = "./audio_dataset/bruckell_16k_partials"
CSV_LOG = "./audio_dataset/bruckell_16k_partials/csv_log"
DDSP_PREPROCESSED = "./preprocessed/DDSP"
NBN_PREPROCESSED = "./preprocessed/NBN"

# enviromenty
DDSP_enviroment = 'C:/Users/pospa/.conda/envs/ddsp_pd/python.exe'
NBN_enviroment ='C:/Users/pospa/.conda/envs/noisebandnet/python.exe'

def spust_ddsp():
    print(" Spouštím preprocessing dat pro DDSP...")
    subprocess.run([DDSP_enviroment, "./ddsp/preprocess.py", "--dataset_path", DATASET_PATH, "--preproc_path", DDSP_PREPROCESSED])

def spust_nbn():
    print(" Spouštím preprocessing dat pro Noisebandnet...")
    subprocess.run([NBN_enviroment, "./noisebandnet/label_data.py", "--audio_path", DATASET_PATH, "--output_directory",
                    NBN_PREPROCESSED, "--features", "revolutions per minute,velocity (m/s),throttle", "--sampling_rate", str(16000), "--csv_path", CSV_LOG])


if __name__ == '__main__':
    # Vytvoření procesů
    ddsp_process = multiprocessing.Process(target=spust_ddsp)
    nbn_process = multiprocessing.Process(target=spust_nbn)

    # Spuštění procesů
    ddsp_process.start()
    nbn_process.start()

    # Čekání na dokončení procesů
    ddsp_process.join()
    nbn_process.join()

    print("Paralelní spouštění skriptů dokončeno.")

"""
import subprocess
import os
import threading


DATASET_PATH = "./audio_resynthesis/gavril_log"
DDSP_PREPROCESSED = "./preprocessed/DDSP/resynth/gavril"


#DATASET_PATH = "./audio_dataset/kc6360_16kHz"
CSV_LOG = "./audio_resynthesis/gavril_log/csv_log"
#DDSP_PREPROCESSED = "./preprocessed/DDSP/kc6360_16kHz"
NBN_PREPROCESSED_CSV = "./preprocessed/NBN/paramcntrl/civetta"

# Prostředí
DDSP_enviroment = 'C:/Users/pospa/.conda/envs/ddsp_pd/python.exe'
NBN_enviroment ='C:/Users/pospa/.conda/envs/noisebandnet/python.exe'

DDSP_CONFIG = './config.yaml'

DDSP_TEST_CONFIG = './trained_models/DDSP/final44/gavril_10kep/2025_04_28_12_03_37/config.yaml'

DDSP_TEST_PTH = "./trained_models/DDSP/final44/gavril_10kep/2025_04_28_12_03_37/state.pth"
DDSP_SAVE_DIR = "DDSP/final44/gavril_20kep"
EPOCHS = 20
BATCH_SIZE = 8
SAMPLE_RATE = 44100
SAVE_INTERVAL = 2
NUM_SAMPLES = 32000
HIDDEN_LAYERS = 128

NBN_SAVE_DIR = "./NBN/kotva/civetta"

TRAINED_PATH = './trained_models/NBN/kotva/civetta/2025_04_29_13_24_52'

"""parser = argparse.ArgumentParser(description='DDSP Training Script')
    parser.add_argument('--config', default='./config.yaml', help='Path to config.yaml file')
    parser.add_argument('--save_dir_path', default='runs_car/debug2', help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for training')
    parser.add_argument('--start_lr', type=float, default=1e-3, help='Starting learning rate')
    parser.add_argument('--stop_lr', type=float, default=1e-3, help='Stopping learning rate')
    parser.add_argument('--decay_over', type=int, default=400000, help='Number of steps to decay learning rate over')
    parser.add_argument('--audio_save_interval', type=int, default=10, help='Save audio every N epochs')
    parser.add_argument('--num_samples', type=int, default=64000, help='Number of samples in signal')
    parser.add_argument('--hidden_size', type=int, default=512, help='Number of samples in signal')"""

def spust_ddsp_preprocess():
    print(" Spouštím preprocessing dat pro DDSP...")
    subprocess.run([DDSP_enviroment, "./ddsp/preprocess.py", "--dataset_path", DATASET_PATH, "--preproc_path", DDSP_PREPROCESSED, "--signal_length", str(NUM_SAMPLES),
                    "--features","revolutions per minute,velocity (m/s),throttle", "--csv_path", CSV_LOG, "--config", DDSP_CONFIG, "--ddsp_params", "loudness,pitch"
                     ])
    print("Preprocesing dat pro DDSP dokončen.")
    #spust_ddsp_train()
    spust_ddsp_test()

def spust_nbn_preprocess():
    print(" Spouštím preprocessing dat pro Noisebandnet...")
    subprocess.run([NBN_enviroment, "./noisebandnet/label_data.py", "--audio_path", DATASET_PATH, "--output_directory",
                     NBN_PREPROCESSED_CSV, "--features", "revolutions per minute,velocity (m/s),throttle", "--sampling_rate", str(44100), "--csv_path", CSV_LOG])
    print("Preprocesing dat pro Noisebandnet dokončen.")
    #spust_nbn_train()
    spust_nbn_test()

def spust_ddsp_train():
    print(" Spouštím trénování DDSP modelu...")
    subprocess.run([DDSP_enviroment, "./ddsp/train.py", "--config", DDSP_CONFIG, "--save_dir_path", DDSP_SAVE_DIR, "--features", "revolutions per minute,velocity (m/s),throttle",
                    "--epochs", str(EPOCHS),"--batch", str(BATCH_SIZE),"--audio_save_interval", str(SAVE_INTERVAL),"--num_samples",str(NUM_SAMPLES),"--hidden_size", str(HIDDEN_LAYERS)])

def spust_nbn_train():
    print(" Spouštím trénování Noisebandnet modelu...")
    subprocess.run([NBN_enviroment, "./noisebandnet/train.py", "--dataset_path", DATASET_PATH, "--save_dir_path", NBN_SAVE_DIR,
                    "--control_params_path", NBN_PREPROCESSED_CSV,
                    #"--auto_control_params", "loudness" , "centroid",
                    "--epochs", str(EPOCHS),"--batch_size", str(BATCH_SIZE),
                    "--sampling_rate", str(SAMPLE_RATE),"--audio_size_samples", str(NUM_SAMPLES), "--hidden_size", str(HIDDEN_LAYERS),
                    "--save_model_step", str(SAVE_INTERVAL) ])

def spust_ddsp_test():
    print(" Spouštím testovani DDSP modelu...")
    subprocess.run([DDSP_enviroment, "./ddsp/test.py", "--config", DDSP_TEST_CONFIG, "--pth_file", DDSP_TEST_PTH, "--audio_file",
                    "./audio_resynthesis/gavril_log/gavril_res.wav","--save_audio_dir", "./resynth_audio/paramcntrl/", "--preproc_dir", DDSP_PREPROCESSED])

def spust_nbn_test():
    print(" Spouštím testovani Noisebandnet modelu...")
    #subprocess.run(
     #   [NBN_enviroment, "./noisebandnet/loudness_transfer.py", "--model_path", TRAINED_PATH, "--audio_file", "./audio_resynthesis/kc6360_16/kc6360_res.wav",
      #   "--save_audio_dir", "./resynth_audio/"])

    subprocess.run(
       [NBN_enviroment, "./noisebandnet/feature_resynthesis.py", "--model_path", TRAINED_PATH, "--param_dir", NBN_PREPROCESSED_CSV,
       "--save_audio_dir", "./resynth_audio/paramcntrl/"])




if __name__ == '__main__':
    # Spuštění preprocessing skriptů paralelně pomocí vláken
    ddsp_preprocess_thread = threading.Thread(target=spust_ddsp_preprocess)
    nbn_preprocess_thread = threading.Thread(target=spust_nbn_preprocess)

    ddsp_preprocess_thread.start()
    #nbn_preprocess_thread.start()

    ddsp_preprocess_thread.join()
    #nbn_preprocess_thread.join()

    print("Všechny skripty dokončeny.")