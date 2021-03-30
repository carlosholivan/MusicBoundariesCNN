"""
SCRIPT NÚMERO: 5
Este script contiene los bucles de entrenamiento y validación de la RN.
"""

#terminal: tensorboard --logdir="E:\INVESTIGACIÓN\Proyectos\Boundaries Detection\graphs"
#browser: http://LAPTOP-A05GF0EK:6006

import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter, FileWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms, utils
from data import SSMDataset, normalize_image, padding_MLS, padding_SSLM, borders
from model_CNN_MLS_SSLM_SSLM import CNN_Fusion
import numpy as np
from scipy import signal
import mir_eval

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
start_time = time.time()


batch_size = 1
labels_path = "H:\\INVESTIGACION\\Datasets\\MusicStructure\\SALAMI\\annotations\\"
im_path_mel_near_train = "H:\\INVESTIGACION\\Proyectos\\Boundaries Detection\\Inputs\\TRAIN\\np MLS\\"
im_path_L_MFCCs_near_train = "H:\\INVESTIGACION\\Proyectos\\Boundaries Detection\\Inputs\\TRAIN\\np SSLM from MFCCs euclidean 2pool3\\"
im_path_L_MFCCs_near_train2 = "H:\\INVESTIGACION\\Proyectos\\Boundaries Detection\\Inputs\\TRAIN\\np SSLM from MFCCs cosine 2pool3\\"

im_path_mel_near_val = "H:\\INVESTIGACION\\Proyectos\\Boundaries Detection\\Inputs\\VALIDATION\\np MLS\\"
im_path_L_MFCCs_near_val = "H:\\INVESTIGACION\\Proyectos\\Boundaries Detection\\Inputs\\VALIDATION\\np SSLM from MFCCs euclidean 2pool3\\"
im_path_L_MFCCs_near_val2 = "H:\\INVESTIGACION\\Proyectos\\Boundaries Detection\\Inputs\\VALIDATION\\np SSLM from MFCCs cosine 2pool3\\"

#Datasets initialization
dataset_mels_train = SSMDataset(im_path_mel_near_train, labels_path, transforms=[padding_MLS, normalize_image, borders])
dataset_sslms_train = SSMDataset(im_path_L_MFCCs_near_train, labels_path, transforms=[padding_SSLM, normalize_image, borders])
dataset_sslms_train2 = SSMDataset(im_path_L_MFCCs_near_train2, labels_path, transforms=[padding_SSLM, normalize_image, borders])

dataset_mels_val = SSMDataset(im_path_mel_near_val, labels_path, transforms=[padding_MLS, normalize_image, borders])
dataset_sslms_val = SSMDataset(im_path_L_MFCCs_near_val, labels_path, transforms=[padding_SSLM, normalize_image, borders])
dataset_sslms_val2 = SSMDataset(im_path_L_MFCCs_near_val2, labels_path, transforms=[padding_SSLM, normalize_image, borders])

#creating train and test
#1006 songs as dataset, so 65% for training (first 650 songs), 
#15% for validation (next 150 songs) and 
#20% for test (last 206 songs)
                                           
trainloader_mels = DataLoader(dataset_mels_train,
                               batch_size = batch_size,
                               num_workers=0,
                               shuffle=False)

valloader_mels = DataLoader(dataset_mels_val,
                             batch_size = batch_size,
                             num_workers=0)

trainloader_sslms = DataLoader(dataset_sslms_train,
                               batch_size = batch_size,
                               num_workers=0,
                               shuffle=False)

trainloader_sslms2 = DataLoader(dataset_sslms_train2,
                               batch_size = batch_size,
                               num_workers=0,
                               shuffle=False)

valloader_sslms = DataLoader(dataset_sslms_val,
                             batch_size = batch_size,
                             num_workers=0)

valloader_sslms2 = DataLoader(dataset_sslms_val2,
                             batch_size = batch_size,
                             num_workers=0)

print("Input data imported in {:.2f}s".format(time.time() - start_time))

print('============================MODEL=====================================')
print("Total MLS cargados para entrenamiento:", len(trainloader_mels)*batch_size)
print("Total MLS cargados para validacion:", len(valloader_mels)*batch_size)
print("Total SSLMs cargadas para entrenamiento:", len(trainloader_sslms)*batch_size)
print("Total SSLMs cargadas para validacion:", len(valloader_sslms)*batch_size)

output_channels = 32 #mapas características de salida de la capa 1 de la CNN
model = CNN_Fusion(output_channels, output_channels).to(device)

#print('Architecture model:\n', model)
#print('Architecture model fusion:\n', model_fusion)


print("==========================TRAINING====================================")
num_epochs = 1000
learning_rate = 0.001

criterion = nn.BCEWithLogitsLoss() #nn.MSELoss() #BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = []
train_loss = []
validation_loss = []
train_accuracy = []
validation_accuracy = []

writer_train = SummaryWriter("H:\\INVESTIGACION\\Proyectos\\Boundaries Detection\\graphs\\train")
writer_test = SummaryWriter("H:\\INVESTIGACION\\Proyectos\\Boundaries Detection\\graphs\\validation")

#Parameters
iteration = 10
hop_length = 1024
sr = 44100
samples_frame = 2048
pooling_factor = 6
padding_factor = 50
lamda = round(6*sr/hop_length)
window = 3

training_start_time = time.time()
"==================================MAIN LOOP=================================="    
for epoch in range(num_epochs):
    model.train()
    
    training_loss = 0.0
    running_loss = 0.0
    training_accuracy = 0.0
    running_accuracy = 0.0
    
    F_train = 0.0
    R_train = 0.0
    P_train = 0.0

    val_loss = 0.0
    validation_accuracy = 0.0

    examples = 0
    pbar = tqdm(total = len(trainloader_mels))
    print("Epoch:", epoch)

    "==============================TRAIN LOOP================================="
    optimizer.zero_grad()
    for batch_idx, ((images_mels, labels_mels, labels_sec_mels), (images_sslms, labels_sslms, labels_sec_sslms), (images_sslms2, labels_sslms, labels_sec_sslms)) in enumerate(zip(trainloader_mels,trainloader_sslms,trainloader_sslms2)):
        #train in first convolution MLS and SSLMs separately
        if images_mels.shape[3] != images_sslms.shape[3]:
            images_mels = images_mels[:,:,:,1:]
            labels_mels = labels_mels[:,1:]
        """
        plt.imshow(images_sslms[0,0,...].detach().numpy())
        plt.show()
        """
        images_sslms = torch.cat((images_sslms,images_sslms2), axis = 1)
        images_mels = images_mels.to(device)
        images_sslms = images_sslms.to(device)
        
        outputs_combined = model(images_mels.float(), images_sslms.float())
            
        outputs = outputs_combined.view(-1) #2º valor son las clases de salida
        labels = labels_mels.view(-1).float().to(device) *0.98 + 0.01
        
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        
        if batch_idx % iteration == 0:
            optimizer.step()
            optimizer.zero_grad()
               
        #Loss
        training_loss += train_loss
        running_loss = 0.99 * running_loss + (1 - 0.99) * train_loss
    
        #Accuracy
        accuracy = ((outputs > 0.5).float() == labels).float().mean()
        training_accuracy += accuracy
        running_accuracy = 0.99 * running_accuracy + (1 - 0.99) * accuracy
        pbar.set_postfix(loss=running_loss.item()/(1-0.99**(batch_idx+1)), accuracy=running_accuracy.item()/(1-0.99**(batch_idx+1)))
        pbar.update()
        writer_train.add_scalar('Output Max', torch.sigmoid(outputs.max()), batch_idx)
        
        """This is only to graph results in tensorboard"""
        labels_sec_mels = labels_sec_mels[0,:]
        reference = np.array((np.copy(labels_sec_mels[:-1]), np.copy(labels_sec_mels[1:]))).T
        repeated_list = []
        for j in range(reference.shape[0]):
            if reference[j,0] == reference[j,1]:
                repeated_list.append(j)
        reference = np.delete(reference, repeated_list, 0)
    
        pred_new = torch.sigmoid(outputs).cpu().detach().numpy()
                
        delta = (pred_new.max() - pred_new.min())*0.2 + pred_new.min() #threshold
        
        peak_position = signal.find_peaks(pred_new, height=delta, distance=lamda)[0] #array of peaks
        peaks_positions = ((peak_position-padding_factor)*pooling_factor*hop_length)/sr
        for i in range(len(peaks_positions)):
            if peaks_positions[i] < 0:
                peaks_positions[i] = 0

        peaks_position = np.insert(peaks_positions, 0, 0)
        #peaks_position = np.append(peaks_position, len(pred_new))
        pred_positions = np.array((np.copy(peaks_position[:-1]), np.copy(peaks_position[1:]))).T
        repeated_list = []
        for j in range(pred_positions.shape[0]):
            if pred_positions[j,0] == pred_positions[j,1]:
                repeated_list.append(j)
        pred_positions = np.delete(pred_positions, repeated_list, 0)
        
        P_ant_train, R_ant_train, F_ant_train, *_ = mir_eval.segment.detection(reference, pred_positions, window=window, beta=1.0, trim=False)
        P_train, R_train, F_train = P_train + P_ant_train, R_train + R_ant_train, F_train + F_ant_train
        """"""

    pbar.close()
    print("training_accuracy={:.2f}  training_loss={:.2f}\n".format(training_accuracy / len(trainloader_mels), training_loss / len(trainloader_mels)))

    writer_train.add_scalar('Loss', training_loss / len(trainloader_mels), epoch)
    writer_train.add_scalar('Accuracy', training_accuracy / len(trainloader_mels), epoch)
    
    
    n_songs_train = len(trainloader_mels)
    P_total_train, R_total_train, F_total_train = P_train/n_songs_train, R_train/n_songs_train, F_train/n_songs_train
    writer_train.add_scalar('R', R_total_train, epoch)
    writer_train.add_scalar('P', P_total_train, epoch)
    writer_train.add_scalar('F', F_total_train, epoch)

    model.eval()
    F = 0.0
    R = 0.0
    P = 0.0

    "============================VALIDATION LOOP=============================="
    with torch.no_grad(): #para que no calcule gradientes en validación
        for batch_idx, ((images_mels, labels_mels, labels_sec_mels), (images_sslms, labels_sslms, labels_sec_sslms), (images_sslms2, labels_sslms, labels_sec_sslms)) in enumerate(zip(valloader_mels,valloader_sslms,valloader_sslms2)):
            #Forward pass
            if images_mels.shape[3] != images_sslms.shape[3]:
                images_mels = images_mels[:,:,:,1:]
                labels_mels = labels_mels[:,1:]
            
            images_sslms = torch.cat((images_sslms,images_sslms2), axis = 1)

            images_mels = images_mels.to(device)
            images_sslms = images_sslms.to(device)

            val_outputs_combined = model(images_mels.float(), images_sslms.float())
            val_outputs = val_outputs_combined.view(-1) #2º valor son las clases de salida
            labels = labels_mels.view(-1).float().to(device) *0.98 + 0.01
            
            labels_sec_mels = labels_sec_mels[0,:]
            reference = np.array((np.copy(labels_sec_mels[:-1]), np.copy(labels_sec_mels[1:]))).T
            repeated_list = []
            for j in range(reference.shape[0]):
                if reference[j,0] == reference[j,1]:
                    repeated_list.append(j)
            reference = np.delete(reference, repeated_list, 0)
            
            pred_new = torch.sigmoid(val_outputs).cpu().detach().numpy()
                
            delta = (pred_new.max() - pred_new.min())*0.2 + pred_new.min() #threshold
            
            peak_position = signal.find_peaks(pred_new, height=delta, distance=lamda)[0] #array of peaks
            peaks_positions = ((peak_position-padding_factor)*pooling_factor*hop_length)/sr
            for i in range(len(peaks_positions)):
                if peaks_positions[i] < 0:
                    peaks_positions[i] = 0

            peaks_position = np.insert(peaks_positions, 0, 0)
            #peaks_position = np.append(peaks_position, len(pred_new))
            pred_positions = np.array((np.copy(peaks_position[:-1]), np.copy(peaks_position[1:]))).T
            repeated_list = []
            for j in range(pred_positions.shape[0]):
                if pred_positions[j,0] == pred_positions[j,1]:
                    repeated_list.append(j)
            pred_positions = np.delete(pred_positions, repeated_list, 0)

            P_ant, R_ant, F_ant, *_ = mir_eval.segment.detection(reference, pred_positions, window=window, beta=1.0, trim=False)
            P, R, F = P + P_ant, R + R_ant, F + F_ant
            
            val_loss_size = criterion(val_outputs, labels)

            #Loss
            val_loss += val_loss_size

            #Accuracy
            val_accuracy = ((val_outputs > 0.5).float() == labels).float().mean()
            validation_accuracy += val_accuracy
            
        print("validation_accuracy={:.2f}  validation_loss={:.2f}\n".format(validation_accuracy / len(valloader_mels), val_loss / len(valloader_mels)))

        writer_test.add_scalar('Loss', val_loss / len(valloader_mels), epoch) #epoch* len(valloader) + batch_idx
        writer_test.add_scalar('Accuracy', validation_accuracy / len(valloader_mels), epoch)
        
        n_songs = len(valloader_mels)
        P_total, R_total, F_total = P/n_songs, R/n_songs, F/n_songs
        writer_test.add_scalar('R', R_total, epoch)
        writer_test.add_scalar('P', P_total, epoch)
        writer_test.add_scalar('F', F_total, epoch)

        
    #save trained model every 5 epochs
    if epoch % 5 == 0:
        torch.save(model.state_dict(), "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Trained_models\\saved_model_" + str(epoch) + "epochs.bin")

writer_train.close()
writer_test.close()


print("Training finished in {:.2f}s".format(time.time() - training_start_time))
#Se guardan los pesos de entrenamiento
torch.save(model.state_dict(), "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Trained_models\\saved_model_" + str(epoch) + "epochs.bin")

"""
for n,p in model.named_parameters():
    print(n, p.shape)
"""
#model.cnn1_mel.conv1.weight.grad