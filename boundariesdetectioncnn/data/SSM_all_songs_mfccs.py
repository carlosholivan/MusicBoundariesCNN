"""
SCRIPT NÚMERO: 1
Este script calcula el Espectrograma de Mel y las matriz de Lag para TODAS las 
canciones de SALAMI de acorde a los contextos de lag L = 14s y 88s tal y como
se describe en el paper "MUSIC BOUNDARY DETECTION USING NEURAL NETWORKS ON 
SPECTROGRAMS AND SELF-SIMILARITY LAG MATRICES"
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time
import skimage.measure
import scipy
from scipy.spatial import distance
import os

start_time = time.time()

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

"=================================Functions================================="
def mel_spectrogram(sr_desired, name_song, window_size, hop_length):
    "This function calculates the mel spectrogram in dB with Librosa library"
    y, sr = librosa.load(name_song, sr=None)
    if sr != sr_desired:
        y = librosa.core.resample(y, sr, sr_desired)
        sr = sr_desired
        
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=window_size, hop_length=hop_length, n_mels=80, fmin=80, fmax=16000)
    S_to_dB = librosa.power_to_db(S,ref=np.max) #convert S in dB
    return S_to_dB #S_to_dB is the spectrogam in dB 

def max_pooling(spectrogram_padded, pooling_factor):
    x_prime = skimage.measure.block_reduce(spectrogram_padded, (1, pooling_factor), np.max) 
    return x_prime
  
    
def ssm(spectrogram, pooling_factor):

    """This part max-poolend the spectrogram in time axis by a factor of p"""
    x_prime = max_pooling(spectrogram, pooling_factor)

    """"This part calculates a circular Self Similarity Matrix given
    the mel spectrogram padded and max-pooled"""
    #MFCCs calculation from DCT-Type II
    MFCCs = scipy.fftpack.dct(x_prime, axis=0, type=2, norm='ortho')
    MFCCs = MFCCs[1:,:] #0 componen ommited
    
    #Bagging frames
    m = 2 #baggin parameter in frames
    x = [np.roll(MFCCs,n,axis=1) for n in range(m)]
    x_hat = np.concatenate(x, axis=0)
    x_hat = np.transpose(x_hat)

    ssm = compute_ssm(x_hat)
    
    #Check if SSLM has nans and if it has them, substitute them by 0
    for i in range(ssm.shape[0]):
        for j in range(ssm.shape[1]):
            if np.isnan(ssm[i,j]):
                ssm[i,j] = 0
        
    return ssm

"=================================Variables================================="
window_size = 2048 #(samples/frame)
hop_length = 1024 #overlap 50% (samples/frame) 
sr_desired = 44100
p = 6 #pooling factor

"=================================Paths====================================="
song_path = "E:\\UNIVERSIDAD\\MÁSTER INGENIERÍA INDUSTRIAL\\TFM\\Database\\salami-data-public\\songs\\"
im_path_MFCCs_near = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\np SSM from MFCCs\\"
 
if not os.path.exists(im_path_MFCCs_near):
    os.makedirs(im_path_MFCCs_near)

"===============================Main Loop===================================="    
i = 0
for root, dirs, files in os.walk(song_path):
    for name_song in files:
        start_time_song = time.time()
        i += 1
        song_id = name_song[:-4] #delete .mp3 characters from the string
        print("song", song_id, "prepared to be processed.")
        if str(song_id) + ".npy" not in os.listdir(im_path_MFCCs_near):
            y, sr = librosa.load(song_path + name_song, sr=None)
            mel = mel_spectrogram(sr_desired, song_path + name_song, window_size, hop_length)
            ssm_near = ssm(mel, p)
            
            
            #Save mels matrices and sslms as numpy arrays in separate paths 
            np.save(im_path_MFCCs_near + song_id, ssm_near)
                    
            print("song", song_id, "converted.",  i,"/",len(files), "song converted to all files in : {:.2f}s".format(time.time() - start_time_song))
        
        else:
            print("song", song_id, "already in the directory.")
        
print("All images have benn converted in: {:.2f}s".format(time.time() - start_time))