import matplotlib.pyplot as plt
import torch
from model_CNN_MLS_4SSLM import CNN_Fusion
import numpy as np
from extract_labels_from_txt import ReadDataFromtxt
from scipy import signal
import mir_eval
import os
from torchvision import transforms, utils
from data import SSMDataset, normalize_image, padding_MLS, padding_SSLM, borders
from torch.utils.data import DataLoader
import statistics



epochs = "100" #load model trained this number of epochs
#Model loading
output_channels = 32 #mapas características de salida de la capa 1 de la CNN
model = CNN_Fusion(output_channels, output_channels)
model.load_state_dict(torch.load("/media/carlos/FILES/INVESTIGACIÓN/Proyectos/Boundaries Detection/Trained models/MLS_4SSLM/saved_model_" + epochs + "epochs.bin", map_location=torch.device('cpu')))
model.eval()

batch_size = 1

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

im_path_mel = "/media/carlos/FILES/INVESTIGACIÓN/Proyectos/Boundaries Detection/Inputs/TEST/np MLS/"
im_path_L_MFCCs = "/media/carlos/FILES/INVESTIGACIÓN/Proyectos/Boundaries Detection/Inputs/TEST/np SSLM from MFCCs euclidean 2pool3/"
im_path_L_MFCCs2 = "/media/carlos/FILES/INVESTIGACIÓN/Proyectos/Boundaries Detection/Inputs/TEST/np SSLM from MFCCs cosine 2pool3/"
im_path_L_MFCCs3 = "/media/carlos/FILES/INVESTIGACIÓN/Proyectos/Boundaries Detection/Inputs/TEST/np SSLM from Chromas euclidean 2pool3/"
im_path_L_MFCCs4 = "/media/carlos/FILES/INVESTIGACIÓN/Proyectos/Boundaries Detection/Inputs/TEST/np SSLM from Chromas cosine 2pool3/"

labels_path = "/media/carlos/FILES/SALAMI/annotations/"


mels_dataset = SSMDataset(im_path_mel, labels_path, transforms=[padding_MLS, normalize_image, borders])
mels_trainloader = DataLoader(mels_dataset, batch_size = batch_size, num_workers=0)

sslms_dataset = SSMDataset(im_path_L_MFCCs, labels_path, transforms=[padding_SSLM, normalize_image, borders])
sslms_trainloader = DataLoader(sslms_dataset, batch_size = batch_size, num_workers=0)

sslms_dataset2 = SSMDataset(im_path_L_MFCCs2, labels_path, transforms=[padding_SSLM, normalize_image, borders])
sslms_trainloader2 = DataLoader(sslms_dataset2, batch_size = batch_size, num_workers=0)

sslms_dataset3 = SSMDataset(im_path_L_MFCCs3, labels_path, transforms=[padding_SSLM, normalize_image, borders])
sslms_trainloader3 = DataLoader(sslms_dataset3, batch_size = batch_size, num_workers=0)

sslms_dataset4 = SSMDataset(im_path_L_MFCCs4, labels_path, transforms=[padding_SSLM, normalize_image, borders])
sslms_trainloader4 = DataLoader(sslms_dataset4, batch_size = batch_size, num_workers=0)

hop_length = 1024
sr = 44100
window_size = 2024
pooling_factor = 6
lamda = 6/pooling_factor
padding_factor = 50
lamda = round(lamda*sr/hop_length)
n_songs = len(sslms_trainloader)
delta = 0.4
beta = 1


window = 0.5
F = list()
R = list()
P = list()

for i in range(len(mels_trainloader)):
    image_mel, label, label_sec = mels_dataset[i]
    image_sslm, label1, label_sec1 = sslms_dataset[i]
    image_sslm2, label2, label_sec2 = sslms_dataset2[i]
    image_sslm3, label3, label_sec3 = sslms_dataset3[i]
    image_sslm4, label4, label_sec4 = sslms_dataset4[i]

    image_mel = np.expand_dims(image_mel, 0) #creating dimension corresponding to batch
    image_sslm = np.expand_dims(image_sslm, 0)
    image_sslm2 = np.expand_dims(image_sslm2, 0)
    image_sslm3 = np.expand_dims(image_sslm3, 0)
    image_sslm4 = np.expand_dims(image_sslm4, 0)

    if image_mel.shape[3] != image_sslm.shape[3]:
        image_mel = image_mel[:, :, :, 1:]
        label = label[1:]

    image_mel = torch.Tensor(image_mel)
    image_sslm = torch.Tensor(image_sslm)
    image_sslm2 = torch.Tensor(image_sslm2)
    image_sslm3 = torch.Tensor(image_sslm3)
    image_sslm4 = torch.Tensor(image_sslm4)

    image_sslm = torch.cat((image_sslm, image_sslm2, image_sslm3, image_sslm4), 1)


    pred = model(image_mel, image_sslm)
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
    peaks_positions = ((peak_position-padding_factor)*pooling_factor*hop_length)/sr
    for i in range(len(peaks_positions)): #if the 1st element is <0 convert it to 0
        if peaks_positions[i] < 0:
            peaks_positions[i] = 0

    pred_positions = np.array((np.copy(peaks_positions[:-1]), np.copy(peaks_positions[1:]))).T
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
print("Threshold:", delta)
print("P =", P_total)
print("R =", R_total)
print("F_",beta," =", F_total)

deviation = statistics.stdev(F)
print("Deviation =", deviation)



