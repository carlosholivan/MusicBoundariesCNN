import matplotlib
import matplotlib.pyplot as plt
import torch
from model_CNN_SSLM import CNN_Fusion
import numpy as np
from extract_labels_from_txt import ReadDataFromtxt
from scipy import signal
import mir_eval
import os
from torchvision import transforms, utils
from data import SSMDataset, normalize_image, padding_MLS, padding_SSLM, borders
from torch.utils.data import DataLoader
import sklearn

"""
This script evaluates the precision of the boundaries predictions for the 
whole dataset.
"""

k = 8 #k=89 is index of song 1358 (paper)
epochs = "120" #load model trained this number of epochs
#Model loading
output_channels = 32 #mapas características de salida de la capa 1 de la CNN
model = CNN_Fusion(output_channels, output_channels)
model.load_state_dict(torch.load("E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Trained models\\SSLM_MFCCs_euclidean_2pool3\\saved_model_" + epochs + "epochs.bin"))
model.eval()

batch_size = 1
transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

im_path_L_MFCCs = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\TEST\\np SSLM from MFCCs euclidean 2pool3\\"
labels_path = "E:\\UNIVERSIDAD\\MÁSTER INGENIERÍA INDUSTRIAL\\TFM\\Database\\salami-data-public\\annotations\\"

sslms_dataset = SSMDataset(im_path_L_MFCCs, labels_path, transforms=[padding_SSLM, normalize_image, borders])
sslms_trainloader = DataLoader(sslms_dataset, batch_size = batch_size, num_workers=0)


hop_length = 1024
sr = 44100
window_size = 2024
pooling_factor = 6
padding_factor = 50
lamda = 6/pooling_factor
lamda = round(lamda*sr/hop_length) #window length 1 second
n_songs = len(sslms_trainloader)
delta = 0.205
beta = 1
window = 0.5


sslm = np.expand_dims(sslms_dataset[k][0], 0)
pred = model(torch.Tensor(sslm))
pred = pred.view(-1,1)
pred = torch.sigmoid(pred)
pred_new = pred.detach().numpy()
pred_new = pred_new[:,0]


#------------------------------------------------------------------------------
label = sslms_dataset[k][2]
label = label[1:]
reference = np.array((np.copy(label[:-1]), np.copy(label[1:]))).T

peak_position = signal.find_peaks(pred_new, height=delta, distance=lamda)[0] #array of peaks
peaks_positions = ((peak_position-padding_factor)*pooling_factor*hop_length)/sr
for i in range(len(peaks_positions)):
    if peaks_positions[i] < 0:
        peaks_positions[i] = 0

pred_positions = np.array((np.copy(peaks_positions[:-1]), np.copy(peaks_positions[1:]))).T
repeated_list = []
for j in range(pred_positions.shape[0]):
        if pred_positions[j,0] == pred_positions[j,1]:
            repeated_list.append(j)
pred_positions = np.delete(pred_positions, repeated_list, 0)


P, R, F, TP = mir_eval.segment.detection(reference, pred_positions, window=window, beta=beta, trim=False)
print("Threshold", delta)
print('P =',P,'R =',R,'F =',F)

TP = len(TP)
FP = ((1 - P)*TP) / P
FN = ((1 - R)*TP) / R

print("True Positives:", TP)
print("False Positives:", FP)
print("False Negatives:", FN)


delta_array = np.zeros_like(sslms_dataset[k][1])
vector = np.arange(sslms_dataset[k][0].shape[2])
#------------------------------------------------------------------------------
#Plot out vs labels
for i in range(len(delta_array)):
    delta_array[i] = delta
plt.plot(vector, delta_array*300, color='aqua') #*300 for 2pool3
plt.plot(vector, sslms_dataset[k][1]*300, 'r-', label='Labels') #*300 for 2pool3
plt.plot(vector, pred[:,0].detach().numpy()*300, 'w-', label='Output') #*300 for 2pool3
plt.imshow(sslms_dataset[k][0][0,...], origin = 'lower', aspect=1) #aspect=2 for 6pool and =1
plt.ylabel("lag bins")
matplotlib.rcParams.update({'font.size': 10})
plt.show()
#plt.savefig("E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Results\\sslm.png", dpi=10000)

#------------------------------------------------------------------------------
#Plot out vs labels
for i in range(len(delta_array)):
    delta_array[i] = delta
import plotly.graph_objs as go
from plotly.offline import plot
trace1 = go.Scatter(x = vector,
                    y = sslms_dataset[k][1],
                    mode = 'lines',
                    name = 'labels',
                    marker = dict(color = 'rgba(72, 141, 244, 1)') #blue
                    )
trace2 = go.Scatter(x = vector,
                    y = pred_new,
                    mode = 'lines',
                    name = 'predictions',
                    marker = dict(color = 'rgba(15, 194, 129, 1)') #green
                    )
trace3 = go.Scatter(x = vector,
                    y = delta_array,
                    mode = 'lines',
                    name = 'delta',
                    marker = dict(color = 'rgba(229, 183, 31, 1)') #yellow
                    )
trace4 = go.Scatter(x = peak_position,
                    y = [pred_new[j] for j in peak_position],
                    mode = 'markers',
                    name = 'estimated peaks',
                    marker = dict(color = 'rgba(240, 87, 57, 1)') #red
                    )
data = [trace1, trace2, trace3, trace4]
layout = dict(title = 'Song 1358 SALAMI 2.0  ' + epochs + ' epochs',
              xaxis= dict(title= 'Time (seconds)',ticklen= 5,zeroline= False))
fig = dict(data = data, layout = layout)
plot(fig)

"""
image = np.load(im_path_mel + "1358.npy")
for i in range(len(sslms_trainloader)):
    if sslms_dataset[i][0].shape[2] == image.shape[1]+100:
        print(i, sslms_dataset[i][0].shape)
"""