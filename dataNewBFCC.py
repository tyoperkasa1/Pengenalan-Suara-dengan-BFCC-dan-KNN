import os
import time
import numpy as np
import pandas
import speechpy
import scipy
import scipy.fftpack as fft
from scipy.io import wavfile
from scipy.signal import get_window

def normalize_audio(audio):
    audio = audio/np.max(np.abs(audio))
    return audio

def frame_audio(audio, FFT_size = 2048, hop_size = 10, sample_rate=44100):
    audio = np.pad(audio, int(FFT_size/2), mode='reflect')
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num, FFT_size))
    for n in range(frame_num):
        frames[n] = audio[n*frame_len : n*frame_len+FFT_size]
    return frames

def freq_to_bark(freq):
    return 6.0 * np.arcsinh(freq / 600.0)

def bark_to_freq(bark):
    return 600.0 * np.sinh(bark / 6.0)

def bark_points(low_freq, high_freq, nfilters, FFT_size, sample_rate):
    global bark_pointz
    fmin_bark = freq_to_bark(low_freq)
    fmax_bark = freq_to_bark(high_freq)
    bark_pointz  = np.linspace(fmin_bark, fmax_bark, nfilters+4)
    freqs = bark_to_freq(bark_pointz)
    return np.floor((FFT_size+1) * (freqs/sample_rate)).astype(int), freqs

def Fm(fb, fc):
    if(fc - 2.5 <= fb <= fc - 0.5):
        return 10**(2.5 * (fb - fc + 0.5))
    elif fc - 0.5 < fb < fc + 0.5:
        return 1
    elif fc + 0.5 <= fb <= fc + 1.3:
        return 10**(-2.5 * (fb - fc - 0.5))
    else:
        return 0

def intensity_power_law(w):
    def f(w, c, p):
        return w**2 + c * 10**p
    E = (f(w, 56.8, 6) * w**4) / (f(w, 6.3, 6) * f(w, .38, 9) *f(w**3, 9.58, 26))
    return E**(1 / 3)

def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)
    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
    return basis
    
def convertBFCC():
    print("--Mulai--")
    lokasiDataBaru = "./dataNew/"
    namaFile = [f for f in os.listdir(lokasiDataBaru) if os.path.isfile(os.path.join(lokasiDataBaru, f))]
    namaFile.sort()
    indexFileTerakhir = int((len(namaFile))-1)
    i = indexFileTerakhir
    print("Memproses :",namaFile[i])

    fiturmean = np.empty((40, 1))
    time.sleep(2)
    print("Ekstraksi fitur suara dengan BFCC")
    time.sleep(0.5)
    sample_rate, audio = wavfile.read(lokasiDataBaru + namaFile[indexFileTerakhir])
    waktuSekarang = time.time()
    print("\t - Membaca audio...\t\t\t\t(done)")

    if (len(audio.shape) > 1):
        audio1 = normalize_audio(audio[:,0])
    else:
        audio1 = normalize_audio(audio)
        
    threshold=0.1
    awal = 0
    audiohasil = audio1
    for x in range (len(audio1)):
        if np.abs(audio1[x]) >= threshold:
            awal=x
            break
    audiohasil = audio1[awal:len(audio1)]

    for x in range (len(audiohasil)):
        if np.abs(audiohasil[x]) >=threshold:
            akhir=x
    audiohasil2=audiohasil[0:akhir]
    print("\t - Normalisasi sinyal...\t\t\t(done)")

    hop_size = 12 
    FFT_size = 2048
    audio_framed = frame_audio(audiohasil2, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
    print("\t - Audio Framing...\t\t\t\t(done)")
    
    window = get_window("hamming", FFT_size, fftbins=True)
    audio_win = audio_framed * window
    print("\t - Windowing...\t\t\t\t\t(done)")
    
    audio_winT = np.transpose(audio_win)
    audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
    audio_fft = np.transpose(audio_fft)
    print("\t - Fast Fourier Transform...\t\t\t(done)")

    audio_power = np.square(np.abs(audio_fft))
    print("\t - Menghitung Audio Power...\t\t\t(done)")

    low_freq = 0
    high_freq = sample_rate / 2
    filterbank = 10

    filter_points, bark_freqs = bark_points(low_freq, high_freq, filterbank, FFT_size, sample_rate)
    def bark_filterbank(filter_points, FFT_size):
        filters = np.zeros([filterbank, int(FFT_size // 2 + 1)])
        for i in range(2, filterbank + 2):
            for j in range(int(filter_points[i - 2]), int(filter_points[i + 2])):
                fc = bark_pointz[i]
                fb = freq_to_bark((j * sample_rate) / (FFT_size + 1))
                filters[i - 2, j] = Fm(fb, fc)
        return np.abs(filters)

    filters = bark_filterbank(filter_points, FFT_size)
    enorm = 2.0 / (bark_freqs[2:filterbank + 2] - bark_freqs[:filterbank])
    filters *= enorm[:, np.newaxis]

    print("\t - Menghitung Filter Point...\t\t\t(done)")
    audio_filtered = np.dot(filters, np.transpose(audio_power))
    prob = replaceZeroes(audio_filtered)
    audio_log = 10.0 * np.log10(audio_filtered)
    print("\t - Melakukan Filterisasi Sinyal...\t\t(done)")

    dct_filter_num = 40
    dct_filters = dct(dct_filter_num, filterbank)
    cepstral_coefficents = np.dot(dct_filters, audio_log)
    print("\t - Generate Nilai Cepstral Coefficient...\t(done)")
    cepstral_coefficents = speechpy.processing.cmvn(cepstral_coefficents,True)

    for xpos in range(len(cepstral_coefficents)):
        sigmax = 0
        for xn in cepstral_coefficents[xpos,:]:
            sigmax += xn
        fiturmean[xpos,0] = sigmax/len(np.transpose(cepstral_coefficents))
    print("\t - Normalisasi Nilai Cepstral Coefficient...\t(done)")
    print("--Selesai--")
    print("Time : {0} detik".format(str(time.time()-waktuSekarang)))    

    indextable = []
    for i in range(40):
        indextable.append("fitur" + str(i+1))

    df = pandas.DataFrame(np.transpose(fiturmean),columns=indextable)
    df.to_excel("./model/newData.xlsx", index=False)
