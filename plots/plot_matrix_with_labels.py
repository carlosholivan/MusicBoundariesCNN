import matplotlib.pyplot as plt
import torch
from model_CNN_paper import CNN_Fusion
import numpy as np
from extract_labels_from_txt import ReadDataFromtxt
from scipy import signal
import mir_eval
import os
from torchvision import transforms, utils
from data import SSMDataset, normalize_image, padding_MLS, padding_SSLM, borders
from torch.utils.data import DataLoader


batch_size = 1

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

sr = 44100
hop_length = 1024
window_size = 2048

im_path_mel = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\np MLS\\"
im_path_L_MFCCs = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\np SSLM from MFCCs\\"
labels_path = "E:\\UNIVERSIDAD\\MÁSTER INGENIERÍA INDUSTRIAL\\TFM\\Database\\salami-data-public\\annotations\\"

mels_dataset = SSMDataset(im_path_mel, labels_path, transforms=[normalize_image, padding_MLS, borders])
mels_trainloader = DataLoader(mels_dataset, batch_size = batch_size, num_workers=0)

sslms_dataset = SSMDataset(im_path_L_MFCCs, labels_path, transforms=[normalize_image, padding_SSLM, borders])
sslms_trainloader = DataLoader(sslms_dataset, batch_size = batch_size, num_workers=0)


#Show the MLS and SSLM of song nº1358 (as paper) with gaussians labels
for i in range(len(mels_trainloader)): #zip(mels_trainloader, sslms_trainloader):
    image_mel, label, label_sec = mels_dataset[i]
    image_sslm, label, label_sec = sslms_dataset[i]
    if image_mel.shape[2] == 1234: #song 1358
        break

plt.figure(1)
vector = np.arange(sslms_dataset[i][0].shape[2])
plt.plot(vector, sslms_dataset[i][1]*100, 'r-', label='Labels')
plt.imshow(sslms_dataset[i][0][0,...], origin = 'lower')
plt.show()

plt.figure(2)
vector2 = np.arange(mels_dataset[i][0].shape[2])
plt.plot(vector2, mels_dataset[i][1]*80, 'r-', label='Labels')
plt.imshow(mels_dataset[i][0][0,...], origin = 'lower')
plt.show()
