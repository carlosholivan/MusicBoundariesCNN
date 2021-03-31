import time
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter, FileWriter
from tqdm import tqdm
from torchvision import transforms, utils

import numpy as np
from scipy import signal
import mir_eval
import os

# Our modules
from boundariesdetectioncnn.data import dataloaders
from boundariesdetectioncnn.models import model_CNN_MLS
from boundariesdetectioncnn import configs


def run_training(model, 
                 batch_size=configs.ParamsConfig.BATCH_SIZE, 
                  epochs=configs.ParamsConfig.NUM_EPOCHS, 
                  save_epoch=5,
                  lr=configs.ParamsConfig.LEARNING_RATE, 
                  iterations=configs.ParamsConfig.ITERATIONS,
                  lamda=configs.InputsConfig.LAMBDA,
                  output_channels=configs.ParamsConfig.OUT_CHANNELS, 
                  padding_factor=configs.InputsConfig.PADDING_FACTOR,
                  pooling_factor=configs.InputsConfig.POOLING_FACTOR,
                  hop_length=configs.InputsConfig.HOP_LENGTH,
                  sr=configs.InputsConfig.SAMPLING_RATE,
                  window=configs.InputsConfig.WINDOW,
                  labels_path=configs.PathsConfig.LABELS_PATH):
    
    start_time = time.time()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if model == "mels":
        input_train_path = configs.PathsConfig.MELS_TRAIN_PATH
        input_val_path = configs.PathsConfig.MELS_VAL_PATH
        
        mels_train_dataset, mels_trainloader = dataloaders.build_dataloader(batch_size, input_train_path, labels_path)
        mels_val_dataset, mels_valloader = dataloaders.build_dataloader(batch_size, input_val_path, labels_path)
        
        assert len(mels_train_dataset) > 0
        assert len(mels_val_dataset) > 0
        
        model = model_CNN_MLS.CNN_Fusion(output_channels, output_channels).to(device)

    print("Input data imported in {:.2f}s".format(time.time() - start_time))

    print('============================MODEL=====================================')
    print("Total SSLMs cargadas para entrenamiento:", len(mels_trainloader)*batch_size)
    print("Total SSLMs cargadas para validacion:", len(mels_valloader)*batch_size)
    
    print("==========================TRAINING====================================")
            
    train_loop(model=model, 
               trainloader=mels_trainloader, 
               valloader=mels_valloader, 
               device=device, 
               save_epoch=save_epoch, 
               epochs=epochs)

    return


def train_loop(model, 
               trainloader,
               valloader,
               device,
               save_epoch=5,
               epochs=configs.ParamsConfig.NUM_EPOCHS, 
               lr=configs.ParamsConfig.LEARNING_RATE, 
               iterations=configs.ParamsConfig.ITERATIONS,
               lamda=configs.InputsConfig.LAMBDA,
               padding_factor=configs.InputsConfig.PADDING_FACTOR,
               pooling_factor=configs.InputsConfig.POOLING_FACTOR,
               hop_length=configs.InputsConfig.HOP_LENGTH,
               sr=configs.InputsConfig.SAMPLING_RATE,
               window=configs.InputsConfig.WINDOW):
    
    

    criterion = nn.BCEWithLogitsLoss() #nn.MSELoss() #BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = []
    validation_loss = []
    train_accuracy = []
    validation_accuracy = []
    writer_train = SummaryWriter("../graphs/train")
    writer_test = SummaryWriter("../graphs/validation")

    training_start_time = time.time()
    "==================================MAIN LOOP=================================="    
    for epoch in range(1, epochs):
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
        pbar = tqdm(total = len(trainloader))
        print("Epoch:", epoch)
        "==============================TRAIN LOOP================================="
        optimizer.zero_grad()
        for batch_idx, (images_mls, labels_mls, labels_sec_mls) in enumerate(trainloader):

            images_mls = images_mls.to(device)

            outputs_combined = model(images_mls.float())

            outputs = outputs_combined.view(-1) #2º valor son las clases de salida
            labels = labels_mls.view(-1).float().to(device) *0.98 + 0.01

            train_loss = criterion(outputs, labels)
            train_loss.backward()

            if batch_idx % iterations == 0:
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
            labels_sec_mls = labels_sec_mls[0,:]
            labels_sec_mls = labels_sec_mls[1:]
            reference = np.array((np.copy(labels_sec_mls[:-1]), np.copy(labels_sec_mls[1:]))).T
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

            pred_positions = np.array((np.copy(peaks_positions[:-1]), np.copy(peaks_positions[1:]))).T
            repeated_list = []
            for j in range(pred_positions.shape[0]):
                if pred_positions[j,0] == pred_positions[j,1]:
                    repeated_list.append(j)
            pred_positions = np.delete(pred_positions, repeated_list, 0)

            P_ant_train, R_ant_train, F_ant_train, *_ = mir_eval.segment.detection(reference, pred_positions, window=window, beta=1.0, trim=False)
            P_train, R_train, F_train = P_train + P_ant_train, R_train + R_ant_train, F_train + F_ant_train
           

        pbar.close()
        print("training_accuracy={:.2f}  training_loss={:.2f}\n".format(training_accuracy / len(trainloader), training_loss / len(trainloader)))

        writer_train.add_scalar('Loss', training_loss / len(trainloader), epoch)
        writer_train.add_scalar('Accuracy', training_accuracy / len(trainloader), epoch)
        
        model.eval()
        F = 0.0
        R = 0.0
        P = 0.0

        "============================VALIDATION LOOP=============================="
        with torch.no_grad(): #para que no calcule gradientes en validación
            for batch_idx, (images_mls, labels_mls, labels_sec_mls) in enumerate(valloader):
                #Forward pass
                images_mls = images_mls.to(device)

                val_outputs_combined = model(images_mls.float())
                val_outputs = val_outputs_combined.view(-1) #2º valor son las clases de salida
                labels = labels_mls.view(-1).float().to(device) *0.98 + 0.01

                labels_sec_mls = labels_sec_mls[0,:]
                reference = np.array((np.copy(labels_sec_mls[:-1]), np.copy(labels_sec_mls[1:]))).T
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
                peaks_position = np.append(peaks_position, len(pred_new))
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

            print("validation_accuracy={:.2f}  validation_loss={:.2f}\n".format(validation_accuracy / len(valloader), val_loss / len(valloader)))

            writer_test.add_scalar('Loss', val_loss / len(valloader), epoch) #epoch* len(valloader) + batch_idx
            writer_test.add_scalar('Accuracy', validation_accuracy / len(valloader), epoch)

            n_songs = len(valloader)
            P_total, R_total, F_total = P/n_songs, R/n_songs, F/n_songs
            writer_test.add_scalar('R', R_total, epoch)
            writer_test.add_scalar('P', P_total, epoch)
            writer_test.add_scalar('F', F_total, epoch)


        #save trained model every 5 epochs
        if not os.path.exists(configs.ParamsConfig.WEIGHTS_PATH):
            os.mkdir(configs.ParamsConfig.WEIGHTS_PATH)

        if epoch % save_epoch == 0:
            torch.save(model.state_dict(), configs.ParamsConfig.WEIGHTS_PATH + "saved_model_" + str(epoch) + "epochs.bin")

    writer_train.close()
    writer_test.close()


    print("Training finished in {:.2f}s".format(time.time() - training_start_time))
    #Se guardan los pesos de entrenamiento
    torch.save(model.state_dict(), configs.ParamsConfig.WEIGHTS_PATH + "saved_model_" + str(epoch) + "epochs.bin")


