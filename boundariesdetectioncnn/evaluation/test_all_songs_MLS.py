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
import statistics

"""
This script evaluates the precision of the boundaries predictions for the 
whole dataset.
"""

epochs = "180" #load model trained this number of epochs
#Model loading
output_channels = 32 #mapas características de salida de la capa 1 de la CNN
model = CNN_Fusion(output_channels, output_channels)
model.load_state_dict(torch.load("E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Trained models\\MLS\\saved_model_" + epochs + "epochs.bin"))
model.eval()

batch_size = 1

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

im_path_mel = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\TEST\\np MLS\\"
labels_path = "E:\\UNIVERSIDAD\\MÁSTER INGENIERÍA INDUSTRIAL\\TFM\\Database\\salami-data-public\\annotations\\"

mels_dataset = SSMDataset(im_path_mel, labels_path, transforms=[padding_MLS, normalize_image, borders])
mels_trainloader = DataLoader(mels_dataset, batch_size = batch_size, num_workers=0)

hop_length = 1024
sr = 44100
window_size = 2024
pooling_factor = 6
lamda = 6/pooling_factor
padding_factor = 50
lamda = round(lamda*sr/hop_length)
n_songs = len(mels_trainloader)
delta = 0.205
beta = 1


window = 0.5
F = list()
R = list()
P = list()

for i in range(len(mels_trainloader)): #zip(mels_trainloader, sslms_trainloader):
    image_mel, label, label_sec = mels_dataset[i]
    image_mel = np.expand_dims(image_mel, 0) #creating dimension corresponding to batch
    image_mel = torch.Tensor(image_mel)
    pred = model(image_mel)
    pred = pred.view(-1, 1)
    pred = torch.sigmoid(pred)
    pred_new = pred.detach().numpy()
    pred_new = pred_new[:,0]

    #----------------------------------------------------------------------
    label_sec = label_sec[1:]
    reference = np.array((np.copy(label_sec[:-1]), np.copy(label_sec[1:]))).T #labels in seconds
    repeated_list = []
    for j in range(reference.shape[0]):
        if reference[j,0] == reference[j,1]:
            repeated_list.append(j)
    reference = np.delete(reference, repeated_list, 0)

    peak_position = signal.find_peaks(pred_new, height=delta, distance=lamda)[0] #array of peaks
    peaks_position = ((peak_position-padding_factor)*pooling_factor*hop_length)/sr
    for i in range(len(peaks_position)): #if the 1st element is <0 convert it to 0
        if peaks_position[i] < 0:
            peaks_position[i] = 0

    pred_positions = np.array((np.copy(peaks_position[:-1]), np.copy(peaks_position[1:]))).T
    repeated_list = []
    for j in range(pred_positions.shape[0]):
        if pred_positions[j,0] == pred_positions[j,1]:
            repeated_list.append(j)
    pred_positions = np.delete(pred_positions, repeated_list, 0)
    
    P_ant, R_ant, F_ant, *_ = mir_eval.segment.detection(reference, pred_positions, window=window, beta=beta, trim=False)
    P.append(P_ant)
    R.append(R_ant)
    F.append(F_ant)
    
P_total, R_total, F_total = sum(P)/n_songs, sum(R)/n_songs, sum(F)/n_songs
print("MLS")
print("Epochs:", epochs)
print("Threshold:", delta)
print("P =", P_total)
print("R =", R_total)
print("F_",beta," =", F_total)

deviation = statistics.stdev(F)
print("Deviation =", deviation)



