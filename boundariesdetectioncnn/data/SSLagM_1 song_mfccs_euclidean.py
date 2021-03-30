"""
SCRIPT NÚMERO: 3
Este script calcula el Espectrograma de Mel y las matriz de Lag para una 
canción de SALAMI de acorde a un contexto de lag L.
Este método sigue los pasos del paper "MUSIC BOUNDARY DETECTION USING NEURAL 
NETWORKS ON SPECTROGRAMS AND SELF-SIMILARITY LAG MATRICES"
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time
import skimage.measure
import scipy
from scipy.spatial import distance
import math
import extract_labels_from_txt

start_time = time.time()
"""=================================SIGNAL=============================="""
song = "10"
song_path = "/media/carlos/FILES1/SALAMI/songs/" + song + ".mp3"
window_size = 2048 #(samples/frame)
hop_length = 1024 #overlap 50% (samples/frame) 
sr_desired = 44100
y, sr = librosa.load(song_path, sr=None)

if sr != sr_desired:
    y = librosa.core.resample(y, sr, sr_desired)
    sr = sr_desired

p = 6
L_sec = 14 #lag context in seconds
L = round(L_sec*sr/hop_length) #conversion of lag L seconds to frames

"""--------------------------------------------------------------------""" 
"""========================ESPECTROGRAMA DE MEL========================"""
"""--------------------------------------------------------------------""" 

S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=window_size, hop_length=hop_length, n_mels=80, fmin=80, fmax=16000)
S_to_dB = librosa.power_to_db(S,ref=np.max) #convert S in dB   
    
padding_factor = L  #frames 
pad = np.full((S_to_dB.shape[0], padding_factor), -70) #matrix of 80x30frames of -70dB corresponding to padding
S_padded = np.concatenate((pad, S_to_dB), axis=1) #padding 30 frames with noise at -70dB at the beginning
#S_padded = np.concatenate((S_padded, pad), axis=1)

"""--------------------------------------------------------------------""" 
"""==============================MFCCs================================="""
"""--------------------------------------------------------------------""" 
#max pooling of p=2 factor (columns dimension of time series becomes N/p)
x_prime = skimage.measure.block_reduce(S_padded, (1,p), np.max) #Mel Spectrogram downsampled

#MFCCs calculation by computing the Discrete Cosine Transform of type II (DCT-II)
MFCCs = scipy.fftpack.dct(x_prime, axis=0, type=2, norm='ortho')
MFCCs = MFCCs[1:,:]


#Bagging frames
m = 2 #baggin parameter in frames
x = [np.roll(MFCCs,n,axis=1) for n in range(m)]
x_hat = np.concatenate(x, axis=0)

#Cosine distance calculation: D[N/p,L/p] matrix
distances = np.zeros((x_hat.shape[1], padding_factor//p)) #D has as dimensions N/p and L/p
for i in range(x_hat.shape[1]): #iteration in columns of x_hat
    for l in range(padding_factor//p):
        if i-(l+1) < 0:
            dist = 1
        elif i-(l+1) < padding_factor//p:
            dist = 1
        else:
            dist = distance.euclidean(x_hat[:,i], x_hat[:,i-(l+1)]) #cosine distance between columns i and i-L
        distances[i,l] = dist
      
#Threshold epsilon[N/p,L/p] calculation
kappa = 0.1
epsilon = np.zeros((distances.shape[0], padding_factor//p)) #D has as dimensions N/p and L/p
for i in range(padding_factor//p, distances.shape[0]): #iteration in columns of x_hat
    for l in range(padding_factor//p):
        epsilon[i,l] = np.quantile(np.concatenate((distances[i-l,:], distances[i,:])), kappa)

distances = distances[padding_factor//p:,:]  
epsilon = epsilon[padding_factor//p:,:] 
x_prime = x_prime[:,padding_factor//p:] 

#Self Similarity Lag Matrix
sslm = scipy.special.expit(1-distances/epsilon) #aplicación de la sigmoide
sslm = np.transpose(sslm)

#Check if SSLM has nans and if it has them, substitute them by 0
for i in range(sslm.shape[0]):
    for j in range(sslm.shape[1]):
        if np.isnan(sslm[i,j]):
            sslm[i,j] = 0

#Plot SSLM
plt.figure(1)
plt.title("MLS")
plt.imshow(x_prime, origin='lower', cmap='plasma')
plt.show()

#Plot SSLM
plt.figure(2)
plt.title("SSLM")
fig = plt.imshow(sslm, origin='lower', cmap='viridis')
#fig.axes.get_xaxis().set_visible(False)
#fig.axes.get_yaxis().set_visible(False)
plt.show()

"""Now, with the SSLM calculated, we plot the transitions along time axis 
reading them from the .txt annotations file"""
path = "/media/carlos/FILES1/SALAMI/annotations/" + song + "/parsed/"
file = "textfile1_functions.txt"
labels_sec = extract_labels_from_txt.ReadDataFromtxt(path, file)
labels = [(float(labels_sec[i])*sr/(p*hop_length)) for i in range(len(labels_sec))]
plt.imshow(sslm, origin='lower', cmap='gray')
for x in range(len(labels)):
    plt.axvline(labels[x], ymin=0.8, color='y', lw=2, linestyle='-')
plt.show()


if sslm.shape[1] == x_prime.shape[1]:
    print("SSLM and MLS have the same time dimension (columns).")
else:
    print("ERROR. Time dimension of SSLM and MLS mismatch.")
    print("MLS has", x_prime.shape[1], "lag bins and the SSLM", sslm.shape[1])