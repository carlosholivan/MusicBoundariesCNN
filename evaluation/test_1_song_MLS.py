import matplotlib
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
import sklearn

"""
This script evaluates the precision of the boundaries predictions for the 
whole dataset.
"""

k = 8 #k=89 is index of song 1358 (paper) k=148 bien k=4 k=8 k=62 k=73 k=199 k=190
epochs = "180" #load model trained this number of epochs
#Model loading
output_channels = 32 #mapas características de salida de la capa 1 de la CNN
model = CNN_Fusion(output_channels, output_channels)
model.load_state_dict(torch.load("/media/carlos/FILES/INVESTIGACIÓN/Proyectos/Boundaries Detection/Trained models/MLS/saved_model_" + epochs + "epochs.bin"))
model.eval()

batch_size = 1
transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

im_path_MLS = "/media/carlos/FILES/INVESTIGACIÓN/Proyectos/Boundaries Detection/Inputs/TEST/np MLS/"
labels_path = "/media/carlos/FILES/SALAMI/annotations/"

mls_dataset = SSMDataset(im_path_MLS, labels_path, transforms=[padding_MLS, normalize_image, borders])
mls_trainloader = DataLoader(mls_dataset, batch_size = batch_size, num_workers=0)

"""
for i in range(len(mels_trainloader)):
    image_mel, label, label_sec = mels_dataset[i]
    if image_mel.shape[2] == 2136+100: #search image n701
        print(i)
"""

hop_length = 1024
sr = 44100
window_size = 2024
pooling_factor = 6
padding_factor = 50
lamda = 6/pooling_factor
lamda = round(lamda*sr/hop_length) #window length 1 second
n_songs = len(mls_trainloader)
delta = 0.205
beta = 1
window = 0.5


mls = np.expand_dims(mls_dataset[k][0], 0)
pred = model(torch.Tensor(mls))
pred = pred.view(-1,1)
pred = torch.sigmoid(pred)
pred_new = pred.detach().numpy()
pred_new = pred_new[:,0]


#------------------------------------------------------------------------------
label = mls_dataset[k][2]
label = label[1:]
label_padded = [label[i] + padding_factor*hop_length/sr for i in range(label.shape[0])]

reference = np.array((np.copy(label[:-1]), np.copy(label[1:]))).T
repeated_list = []
for j in range(reference.shape[0]):
    if reference[j,0] == reference[j,1]:
        repeated_list.append(j)
reference = np.delete(reference, repeated_list, 0)

peak_position = signal.find_peaks(pred_new, height=delta, distance=lamda)[0] #array of peaks
#for candidate in peak_position:
    #new_candidate = np.mean(pred_new[int(candidate-12*padding_factor*sr/hop_length) : int(candidate+6*padding_factor*sr/hop_length)])


peaks_position = ((peak_position-padding_factor)*pooling_factor*hop_length)/sr
for i in range(len(peaks_position)):
    if peaks_position[i] < 0:
        peaks_position[i] = 0

pred_positions = np.array((np.copy(peaks_position[:-1]), np.copy(peaks_position[1:]))).T
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


delta_array = np.zeros_like(mls_dataset[k][1])
vector = np.arange(mls_dataset[k][0].shape[2])
#------------------------------------------------------------------------------
#Plot out vs labels
plt.figure(1)
for i in range(len(delta_array)):
    delta_array[i] = delta
plt.plot(vector, delta_array*80, color='aqua')
plt.plot(vector, mls_dataset[k][1]*80, 'r-', label='Labels')
plt.plot(vector, pred[:,0].detach().numpy()*80, 'w-', label='Output')
plt.imshow(mls_dataset[k][0][0,...], origin = 'lower', aspect=3)
plt.ylabel("mel bands")
matplotlib.rcParams.update({'font.size': 10})
plt.show()

plt.figure(4)
plt.plot(vector, pred[:,0].detach().numpy()*80, 'w-', label='Output')
plt.show()

#------------------------------------------------------------------------------
#Plot out vs labels
for i in range(len(delta_array)):
    delta_array[i] = delta
import plotly.graph_objs as go
from plotly.offline import plot
trace1 = go.Scatter(x = vector,
                    y = mls_dataset[k][1],
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
for i in range(len(mls_trainloader)):
    if mls_dataset[i][0].shape[2] == image.shape[1]+100:
        print(i, mls_dataset[i][0].shape)
"""