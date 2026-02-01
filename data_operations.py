import numpy as np
import os
import glob
import pickle
import subprocess
import time
import scipy.fftpack
import librosa

def load_audio(file_path, sr=22050):
    command = [
        'ffmpeg',
        '-i', file_path,        # Plik wejściowy
        '-f', 'f32le',          # Format wyjściowy: float 32-bit 
        '-ac', '1',             # Audio Channels: 1 (Mono) 
        '-ar', str(sr),         # Audio Rate: docelowe próbkowanie
        '-acodec', 'pcm_f32le', # Kodek PCM
        '-'                     # Wyjście na standardowe wyjście (pipe) zamiast do pliku
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
    stdout_data, _ = process.communicate()

    # Zamieniamy surowe bajty na tablicę numpy
    audio_array = np.frombuffer(stdout_data, dtype=np.float32)
    
    return audio_array


def preprocess_dataset_update(input_folder, output_file, target_sr=22050, album_name=None):
    if os.path.exists(output_file):
        print(f"File {output_file} exists, will update it...")
        try:
            with open(output_file, 'rb') as f:
                dataset = pickle.load(f)
        except Exception as e:
            print(f"Error during loading {output_file}: {e}. Creating new dataset...")
            dataset = {}
    else:
        print(f"File {output_file} doesnt exists. Creating new dataset...")
        dataset = {}

    files = glob.glob(os.path.join(input_folder, "*.mp3"))
    for path in files:
        filename = os.path.basename(path)
        try:
            audio_data = load_audio(path, sr=target_sr)
            if album_name is None:
                dataset[filename] = audio_data
            else:
                dataset[filename] = {
                    "audio": audio_data,
                    "album": album_name  
                }
        except Exception as e:
            print(f"Error for file {filename}: {e}")

    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    
def delete_dataset(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"File {file_path} was deleted.")
        except OSError as e:
            print(f"Couldnt delete file {file_path}. Error: {e}")
    else:
        print(f"File {file_path} doesnt exist.")