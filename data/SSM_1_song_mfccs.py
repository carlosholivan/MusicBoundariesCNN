"""
SCRIPT NÚMERO: 3
Este script calcula el Espectrograma de Mel y las matriz de Recurrencia para
una canción de SALAMI.
Este método sigue los pasos del paper "MUSIC BOUNDARY DETECTION USING NEURAL 
NETWORKS ON SPECTROGRAMS AND SELF-SIMILARITY LAG MATRICES"
https://pythonhosted.org/msaf/_modules/msaf/algorithms/foote/segmenter.html
"""
#%%
import librosa
import logging
import numpy as np
from scipy.spatial import distance
from scipy import signal
from scipy.ndimage import filters
import librosa.display
import matplotlib.pyplot as plt
import time
import skimage.measure
import scipy
import math
import extract_labels_from_txt

def median_filter(X, M=8):
    """Median filter along the first axis of the feature matrix X."""
    for i in range(X.shape[1]):
        X[:, i] = filters.median_filter(X[:, i], size=M)
    return X


def compute_gaussian_krnl(M):
    """Creates a gaussian kernel following Foote's paper."""
    g = signal.gaussian(M, M // 3., sym=True)
    G = np.dot(g.reshape(-1, 1), g.reshape(1, -1))
    G[M // 2:, :M // 2] = -G[M // 2:, :M // 2]
    G[:M // 2, M // 2:] = -G[:M // 2, M // 2:]
    return G


def compute_ssm(X, metric="cosine"):
    """Computes the self-similarity matrix of X."""
    D = distance.pdist(X, metric=metric)
    D = distance.squareform(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if np.isnan(D[i,j]):
                D[i,j] = 0
    D /= D.max()
    return 1 - D


#%%
"""=================================SIGNAL=============================="""
song = "10"
song_path = "E:\\UNIVERSIDAD\\MÁSTER INGENIERÍA INDUSTRIAL\\TFM\\Database\\salami-data-public\\songs\\" + song + ".mp3"
window_size = 2048 #(samples/frame)
hop_length = 1024 #overlap 50% (samples/frame) 
sr_desired = 44100
y, sr = librosa.load(song_path, sr=None)

if sr != sr_desired:
    y = librosa.core.resample(y, sr, sr_desired)
    sr = sr_desired

p = 6

"""========================ESPECTROGRAMA DE MEL========================"""
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=window_size, hop_length=hop_length, n_mels=80, fmin=80, fmax=16000)
S_to_dB = librosa.power_to_db(S,ref=np.max) #convert S in dB

"""==============================MFCCs================================="""
#max pooling of p=2 factor (columns dimension of time series becomes N/p)
x_prime = skimage.measure.block_reduce(S_to_dB, (1,p), np.max) #Mel Spectrogram downsampled

#MFCCs calculation by computing the Discrete Cosine Transform of type II (DCT-II)
MFCCs = scipy.fftpack.dct(x_prime, axis=0, type=2, norm='ortho')
MFCCs = MFCCs[1:,:]
#MFCCslibrosa = librosa.feature.mfcc(S=x_prime, sr=sr, n_mfcc=80)

#Bagging frames
m = 2 #baggin parameter in frames
x = [np.roll(MFCCs,n,axis=1) for n in range(m)]
x_hat = np.concatenate(x, axis=0)
x_hat = np.transpose(x_hat)

ssm = compute_ssm(x_hat)

plt.figure(1)
plt.imshow(ssm, origin='lower')
plt.show()
