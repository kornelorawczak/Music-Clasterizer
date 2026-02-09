import numpy as np
import os
import glob
import pickle
import subprocess
import time
import scipy.fftpack
import librosa
from tqdm import tqdm

def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def frame_and_window_signal(signal, sr, size=0.025, stride=0.01):
    signal_length = len(signal)
    frame_length = int(round(size * sr))
    frame_step = int(round(stride * sr))
    
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    '''
        Padding - chcemy aby ostatnia ramka nie była ucięta oraz nie była krótsza niż poprzednie,
        więc uzupełniamy ją zerami
    '''
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)
    '''
        Teraz chcielibyśmy wziąć sygnał 1D i zrobić z niego macierz 2D, która zawiera wektory
        odpowiadające każdej ramce
        Zaczynamy od pierwszego kroku - dostajemy macierz z indeksami względnymi (wewnętrznymi):
        [[0, 1, 2],
         [0, 1, 2],
         [0, 1, 2]]
        Takie coś mówi nam, że z każdej ramki bierzemy kolejno jej 0, 1 i 2 element
        Do tego dodajemy indeksy przesunięcia, czyli np. 
        [[0, 0, 0],  
         [2, 2, 2],  
         [4, 4, 4]]
            Tutaj potrzeba transpozycji 
        Po zsumowaniu dostajemy 
        [[0, 1, 2],
         [2, 3, 4],
         [4, 5, 6]]
        Czyli te wskaźniki, których potrzebujemy do stworzenia tej macierzy ramek 
    '''
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    ''' 
        Przypisanie tych wskaźników (mapy), do faktycznego sygnału - zamiana 1D na 2D
    '''
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    ''' 
        Na koniec stosujemy funkcję hamminga w(n) = 0.54 - 0.46 * cos(2*pi*n / (N-1)),
        która wygładza (wycisza) brzegi aby było to bardziej przystępne dla FFT
    '''
    frames *= np.hamming(frame_length)

    return frames

def power_spectrum(frames, NFFT):
    ''' 
        Ta funkcja liczy fft ramek a potem ich widma mocy. Audio jest rzeczywiste więc używamy RFFT
        NFFT to liczba punktów FFT (zazwyczaj 256 lub 512).
    '''
    fft_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_spec_frames = ((1.0 / NFFT) * (fft_frames ** 2))
    
    return pow_spec_frames

def get_mel_filter_banks(num_filters, NFFT, sr):
    ''' 
        Chcemy stworzyć te trójkątne filtry i je tutaj zastosować
    '''
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)
    '''
        Wykorzystaliśmy mel aby wybrać punkty dzielące na trójkąty, a teraz wracamy już do Hz
    '''
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    ''' 
        Teraz potrzebujemy indeksy tablicy w których zaczynają się trójkąty, mają swój szczyt
        i się kończą. Te indeksy będą dotyczyć tablicy z punktami po zrobieniu FFT na danej ramce.
        
        Cała skala ma sr Hz (dla nas 22050Hz), a więc ułamek hz_points / sr oznacza jaką częścią
        całej skali jest dany dźwięk. 

        Następnie te ułamki mnożymy przez (NFFT + 1), czyli ilość dostępnych 'miejsc' w FFT oraz 
        zaokrąglamy w dół bo chcemy naturalne indeksy
    '''
    indices = np.floor((NFFT + 1) * hz_points / sr)
    ''' 
        Teraz będziemy wypełniać macierz. Zaczynamy od samych zer w docelowym kształcie:
        num_filters x NFFT/2 + 1, gdzie to drugie bierze się z specyfikacji RFFT, która odrzuca
        drugą połowę danych, które dla nas nie są potrzebne
    '''
    fbank = np.zeros((num_filters, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_start = int(indices[m - 1])   # trójkąt się zaczyna 
        f_m_center = int(indices[m])      # szczyt (czubek)
        f_m_end = int(indices[m + 1])     # trójkąt się kończy 
        for k in range(f_m_start, f_m_center):
            # Prosta linia rosnąca między start a center - interpolacja tego od 0 do 1
            fbank[m - 1, k] = (k - indices[m - 1]) / (indices[m] - indices[m - 1])
        for k in range(f_m_center, f_m_end):
            # to samo tylko teraz między start a end - od 1 do 0
            fbank[m - 1, k] = (indices[m + 1] - k) / (indices[m + 1] - indices[m])
    
    return fbank

def compute_mfcc(audio, sr, num_filters=40, num_ceps=12, NFFT=512):
    signal = pre_emphasis(audio)
    frames = frame_and_window_signal(signal, sr)
    pow_spec = power_spectrum(frames, NFFT)
    fbank = get_mel_filter_banks(num_filters, NFFT, sr)
    ''' 
        Przemnażamy macierz z naszymi spektrami mocy (ramki x NFFT/2) z macierzą z filtrami (NFFT/2 x n_f)
        Dostajemy (ramki x num_filters)
        Na koniec zastępujemy 0 przez epsilon aby uniknąć liczenia log(0)
        Potem logarytmujemy i dostajemy wartości w dB
    '''
    filter_banks = np.dot(pow_spec, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    ''' 
        Na koniec robimy na tej macierzy DCT 
        norm='ortho' sprawia że macierz jest ortogonalna

        Zwracamy num_ceps parametrów, zaczynając od tego drugiego, bo pierwszy jest tylko wskaźnikiem
        głośności. DCT układa nam 40 filtrów na czynniki pierwsze od najważniejszego, 
        do najmniej ważnego pod względem ich wag. Dlatego możemy pominąć potem pozostałe np. 27

        Zwracane mfcc to macierz (ramki x num_ceps)
    '''
    mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')
    return mfcc[:, 1 : (num_ceps + 1)]

def stft(audio_data,sr=22050, n_fft=2048, hop_length=512):
    """
    Ręczna implementacja STFT.
    input:
    - nfft: liczba próbek w kadej ramce, wieksze nfft - lepsza rozdzielczosc czestotliwosci ale gorsza czasowa, mniejsze nfft - na odwrót
    -hop_length: liczba próbek o które przesuwamy okno między kolejnymi ramkami
    output:
    - magnitudes: Macierz amplitud (częstotliwość x czas)
    - freq: tablica zakresu czestotliwosci
    """
    #1. Przygotowanie okna (Hanning Window) - statystyka: redukuje listki boczne (wyciek widma)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    
    #2. Podział na ramki
    n_samples = len(audio_data)
    n_frames = 1 + (n_samples - n_fft) // hop_length
    
    # rfft zwraca n_fft/2 + 1 prążków częstotliwości (bo sygnał jest rzeczywisty, druga połowa to lustro)
    n_bins = n_fft // 2 + 1
    magnitudes = np.zeros((n_bins, n_frames))
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        segment = audio_data[start:end]
        spectrum = np.fft.rfft(segment * window)
        
        #interesuje nas amplituda- wartość bezwzględna liczby zespolonej
        magnitudes[:, i] = np.abs(spectrum)

    freq = np.fft.rfftfreq(n_fft, d=1/sr)
        
    return magnitudes, freq

def compute_spectral_centroid(magnitudes,freq_bins):

    numerator = np.sum(magnitudes * freq_bins.reshape(-1, 1), axis=0)
    denominator = np.sum(magnitudes, axis=0)
    
    #dodajemy eps zeby na pewno nie podzielic przez zero
    eps = np.finfo(float).eps
    spectral_centroid = numerator / (denominator + eps)

    return np.array(spectral_centroid)

def copmpute_spectral_rollof(magnitudes,freq_bins):
    threshold_percent = 0.85
    total_energy = np.sum(magnitudes, axis=0)
    threshold_energy = total_energy * threshold_percent
    
    #kumulujemy energię wzdłuż częstotliwości
    cumulative_energy = np.cumsum(magnitudes, axis=0)
    
    #szukamy indeksu, gdzie suma przekracza próg
    rolloff_indices = np.argmax(cumulative_energy >= threshold_energy, axis=0)
    spectral_rolloff = freq_bins[rolloff_indices]
    return np.array(spectral_rolloff)

def compute_spectral_contrast(magnitudes):
    #dziele widmo na 6 pasm i wyliczam dla kazdego pasma spectral contrast

    n_bands = 6
    n_bins = magnitudes.shape[0]
    band_size = n_bins // n_bands
    
    contrasts = []
    
    for i in range(n_bands):
        start = i * band_size
        end = (i + 1) * band_size
        band_magnitude = magnitudes[start:end, :]
        
        # Sortujemy amplitudy w paśmie, żeby znaleźć piki i doliny
        # Quantile method: alpha pika i alpha doliny
        peak = np.percentile(band_magnitude, 98, axis=0) # Górne 2%
        valley = np.percentile(band_magnitude, 2, axis=0) # Dolne 2%
        
        # Kontrast to różnica w skali logarytmicznej (dB)
        # Logarytmujemy, bo ludzkie ucho słyszy głośność logarytmicznie
        contrast = np.log1p(peak) - np.log1p(valley)
        contrasts.append(np.mean(contrast)) #średnia kontrastu w tym paśmie dla całego utworu

    return np.array(contrasts)

def calculate_zcr(audio, size=2048, step=512):
    # Ile ramek zmieści się w audio?
    num_frames = 1 + (len(audio) - size) // step
    frames = np.array([audio[i * step : i * step + size] for i in range(num_frames)])

    # Wyciągamy znaki wszystkich próbek (-1, 0, lub 1)
    signs = np.sign(frames)
    # Mnożymy je. Jeśli wynik jest ujemny, znaczy że znaki były różne (+ * - = -)
    differences = signs[:, :-1] * signs[:, 1:] < 0
    zcr_counts = np.sum(differences, axis=1)
    # Normalizujemy dzieląc przez długość ramki 
    zcr_normalized = zcr_counts / size
    
    return zcr_normalized

def compute_chroma(magnitudes, freqs):
    n_bins, n_frames = magnitudes.shape

    # Liczymy czestotliwosci MIDI dla wszystkich czestotliwosci na raz
    m = 69 + 12 * np.log2(freqs[1:] / 440.0)
    chroma_bins = np.round(m).astype(int) % 12  # 12 poltonow
    chroma_matrix = np.zeros((12, n_frames))

    for t in range(n_frames):
        np.add.at(chroma_matrix[:, t], chroma_bins, magnitudes[1:, t])

    chroma_matrix /= np.sum(chroma_matrix, axis=0, keepdims=True) + 1e-9

    return chroma_matrix

def compute_tempogram(magnitudes, sr=22050, hop_length=512, window_size=128, hop=32, min_bpm=60, max_bpm=200):
    
    energy = np.sum(magnitudes, axis=0)
    energy = (energy - np.mean(energy)) / (np.std(energy) + 1e-9) # normalizacja

    min_lag = int(sr * 60 / max_bpm / hop_length)
    max_lag = int(sr * 60 / min_bpm / hop_length)

    tempos = []
    
    for start in range(0, len(energy) - window_size, hop):
        window = energy[start:start + window_size]

        # porównujemy sygnał ze sobą po jakimś przesunięciu, jeśli wartość jest wysoka, to 
        # najprawdopodobniej mamy jakiś dzwięk z takim bpm
        autocorr = np.correlate(window, window, mode='full')
        autocorr = autocorr[autocorr.size // 2:]

        local_autocorr = autocorr[min_lag:max_lag]
        # local_autocorr[k] - jak bardzo rytm w tym fragmencie pasuje do tempa odpowiadajacego indeksowi
        tempos.append(local_autocorr)

    return np.array(tempos).T

def estimate_bpm(magnitudes, sr=22050, hop_length=512, min_bpm=60, max_bpm=200):
    # ile energii mamy w danym okresie czasowym?
    energy = np.sum(magnitudes, axis=0)
    energy = (energy - np.mean(energy)) / (np.std(energy) + 1e-9) # normalizacja

    # porównujemy sygnał ze sobą po jakimś przesunięciu, sprawdzamy po jakim przeusnięciu
    # korelacja jest największa
    autocorr = np.correlate(energy, energy, mode='full')
    autocorr = autocorr[autocorr.size//2:]  # bierzemy tylko dodatnie lag

    # ograniczamy bpm do sensownego przedziału
    min_lag = int(sr * 60 / max_bpm / hop_length)
    max_lag = int(sr * 60 / min_bpm / hop_length)
    peak_index = np.argmax(autocorr[min_lag:max_lag]) + min_lag

    # zamieniamy na bpm
    period_seconds = peak_index * hop_length / sr
    bpm = 60 / period_seconds
    return bpm


def song_to_representation(audio_data, sr=22050):
    feature_vector = []

    mfccs = compute_mfcc(audio_data, sr)
    feature_vector.extend(np.mean(mfccs, axis=0)) 
    feature_vector.extend(np.std(mfccs, axis=0))  

    zcr = calculate_zcr(audio_data)
    feature_vector.append(np.mean(zcr)) 
    feature_vector.append(np.std(zcr))  

    magnitudes, freqs = stft(audio_data, sr=sr)

    centroid = compute_spectral_centroid(magnitudes, freqs)
    feature_vector.append(np.mean(centroid))
    feature_vector.append(np.std(centroid))

    rolloff = copmpute_spectral_rollof(magnitudes, freqs) 
    feature_vector.append(np.mean(rolloff))
    feature_vector.append(np.std(rolloff))

    contrast = compute_spectral_contrast(magnitudes)
    feature_vector.extend(contrast) 

    chroma = compute_chroma(magnitudes, freqs)
    feature_vector.extend(np.mean(chroma, axis=1)) 
    feature_vector.extend(np.std(chroma, axis=1))  

    bpm = estimate_bpm(magnitudes, sr=sr)
    feature_vector.append(bpm) 

    tempogram = compute_tempogram(magnitudes, sr=sr)
    feature_vector.extend(np.mean(tempogram, axis=1)) 
    feature_vector.extend(np.std(tempogram, axis=1))

    # Konwersja na tablicę numpy float32 (lepsza dla ML)
    return np.array(feature_vector, dtype=np.float32)

def librosa_song_to_representation(audio_data, sr=22050):
    features = []
    
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    features.append(np.mean(mfcc, axis=1))  
    features.append(np.var(mfcc, axis=1))   

    spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    features.append(np.mean(spec_centroid))
    features.append(np.var(spec_centroid))
    
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
    features.append(np.mean(spec_rolloff))
    features.append(np.var(spec_rolloff))
    
    zcr = librosa.feature.zero_crossing_rate(audio_data)
    features.append(np.mean(zcr))
    features.append(np.var(zcr))
    
    chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    features.append(np.mean(chroma_stft, axis=1))
    features.append(np.var(chroma_stft, axis=1))
    
    try:
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]
        features.append([tempo]) 
    except Exception:
        features.append([0]) 

    return np.hstack(features)

def create_feature_dataset(input_pickle_path, librosa=False):
    """
    Wczytuje dataset audio (słownik), mieli go przez pipeline 
    i zwraca macierz X (cechy) oraz listę nazw plików (identyfikatory).
    """
    with open(input_pickle_path, 'rb') as f:
        raw_dataset = pickle.load(f)
    
    X = []
    filenames = []
    albums = []
        
    for filename, item in tqdm(raw_dataset.items()):
        try:
            if isinstance(item, dict):
                audio_data = item['audio']
                album_name = item.get('album', 'Unknown')
                if album_name is None:
                    album_name = 'Unknown'
            else:
                audio_data = item
                album_name = 'Unknown'
                
            if librosa:
                features = librosa_song_to_representation(audio_data)
            else:
                features = song_to_representation(audio_data)
            # Sprawdzenie czy nie ma NaN lub Inf (częste przy logarytmach/dzieleniu przez 0)
            if np.isnan(features).any() or np.isinf(features).any():
                features = np.nan_to_num(features)
                
            X.append(features)
            filenames.append(filename)
            albums.append(album_name) 
        except Exception as e:
            print(f"Błąd przetwarzania {filename}: {e}")
            
    return np.array(X), filenames, albums

                           