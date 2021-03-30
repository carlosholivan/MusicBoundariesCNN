import matplotlib.pyplot as plt
import torch
from model_CNN_MLS import CNN_Fusion
import numpy as np
from extract_labels_from_txt import ReadDataFromtxt
from scipy import signal

import mir_eval
import os
from torchvision import transforms, utils
from data import SSMDataset, normalize_image, padding_MLS, padding_SSLM, borders
from torch.utils.data import DataLoader

"""
This script evaluates the precision of the boundaries predictions for the 
whole dataset.
"""

epochs = "180" #load model trained this number of epochs
#Model loading
output_channels = 32 #mapas características de salida de la capa 1 de la CNN
model = CNN_Fusion(output_channels, output_channels)
model.load_state_dict(torch.load("/media/carlos/FILES/INVESTIGACION/Proyectos/Boundaries Detection/Trained models/MLS/saved_model_" + epochs + "epochs.bin",
                                 map_location=torch.device('cpu')))
model.eval()

batch_size = 1

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

im_path_mel = "/media/carlos/FILES/INVESTIGACION/Proyectos/Boundaries Detection/Inputs/TEST/np MLS/"
#im_path_L_MFCCs = "/media/carlos/FILES/INVESTIGACION/Proyectos/Boundaries Detection/Inputs/TEST/np SSLM from Chromas cosine 2pool3/"
labels_path = "/media/carlos/FILES/INVESTIGACION/Datasets/MusicStructure/SALAMI/annotations/"

mels_dataset = SSMDataset(im_path_mel, labels_path, transforms=[padding_MLS, normalize_image, borders])
mels_trainloader = DataLoader(mels_dataset, batch_size = batch_size, num_workers=0)

#sslms_dataset = SSMDataset(im_path_L_MFCCs, labels_path, transforms=[padding_SSLM, normalize_image, borders])
#sslms_trainloader = DataLoader(sslms_dataset, batch_size = batch_size, num_workers=0)

hop_length = 1024
sr = 44100
window_size = 2024
pooling_factor = 6
padding_factor = 50 #samples
lamda = 6/pooling_factor
lamda = round(lamda*sr/hop_length)
n_songs = len(mels_trainloader)
beta = 1


window = 0.5

j_list = [0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5]
P_list, R_list, F_list = np.zeros_like((j_list)), np.zeros_like((j_list)), np.zeros_like((j_list))
for k in range(len(j_list)):
    F = 0.0
    R = 0.0
    P = 0.0   
    for i in range(len(mels_trainloader)): #zip(mels_trainloader, sslms_trainloader):
        image_mel, label, label_sec = mels_dataset[i]
        #image_sslm, label, label_sec = sslms_dataset[i]
        image_mel = np.expand_dims(image_mel, 0) #creating dimension corresponding to batch
        #image_sslm = np.expand_dims(image_sslm, 0) 
        image_mel = torch.Tensor(image_mel)
        #image_sslm = torch.Tensor(image_sslm)
        pred = model(image_mel)
        pred = pred.view(-1, 1)
        pred = torch.sigmoid(pred)
        pred_new = pred.detach().numpy()
        pred_new = pred_new[:, 0]
        delta = j_list[k] + pred_new.min()  # threshold

        #----------------------------------------------------------------------
        label_sec = label_sec[1:]
        reference = np.array((np.copy(label_sec[:-1]), np.copy(label_sec[1:]))).T  # labels in seconds
        repeated_list = []
        for j in range(reference.shape[0]):
            if reference[j, 0] == reference[j, 1]:
                repeated_list.append(j)
        reference = np.delete(reference, repeated_list, 0)

        peak_position = signal.find_peaks(pred_new, height=delta, distance=lamda)[0]  # array of peaks
        peaks_positions = ((peak_position - padding_factor) * pooling_factor * hop_length) / sr
        for i in range(len(peaks_positions)):  # if the 1st element is <0 convert it to 0
            if peaks_positions[i] < 0:
                peaks_positions[i] = 0

        pred_positions = np.array((np.copy(peaks_positions[:-1]), np.copy(peaks_positions[1:]))).T
        repeated_list = []
        for j in range(pred_positions.shape[0]):
            if pred_positions[j, 0] == pred_positions[j, 1]:
                repeated_list.append(j)
        pred_positions = np.delete(pred_positions, repeated_list, 0)

        P_ant, R_ant, F_ant, *_ = mir_eval.segment.detection(reference, pred_positions, window=window, beta=beta, trim=False)
        P, R, F = P + P_ant, R + R_ant, F + F_ant

    P_list[k], R_list[k], F_list[k] = P/n_songs, R/n_songs, F/n_songs
    index_max = np.where(F_list == F_list.max())

plt.plot(j_list, P_list, 'c', label='P')
plt.plot(j_list, R_list, 'g', label='R')
plt.plot(j_list, F_list, 'm', label='F')
plt.axvline(j_list[int(index_max[0])], linestyle='--', color='y', label='optimum threshold')
plt.xlabel('threshold')
plt.legend()
plt.show()



