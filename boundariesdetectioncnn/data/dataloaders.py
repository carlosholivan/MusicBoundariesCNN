"""
SCRIPT NÚMERO: 4
Este script calcula formaliza la entrada de datos a la RN mediante la creación
del dataloader de los espetrogramas de Mel y para las matrices SSLM
"""

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import librosa

# Our modules
from boundariesdetectioncnn.data import pink_noise
from boundariesdetectioncnn.data.normalize_image import normalize
from boundariesdetectioncnn.data.extract_labels_from_txt import ReadDataFromtxt

"Create array of labels"
def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu)/sig, 2.)/2)

def borders(image, label):
    """This function transforms labels in sc to gaussians in frames"""       
    pooling_factor = 6
    num_frames = image.shape[2]
    repeated_label = []
    for i in range(len(label)-1):
        if label[i] == label[i+1]:
            repeated_label.append(i)
    label = np.delete(label, repeated_label, 0) #labels in seconds
    label = label/pooling_factor #labels in frames
    
    #Pad frames we padded in images also in labels but in seconds
    sr = 44100
    hop_length = 1024
    window_size = 2048
    padding_factor = 50
    label_padded = [label[i] + padding_factor*hop_length/sr for i in range(label.shape[0])]
    vector = np.arange(num_frames)
    new_vector = (vector*hop_length + window_size/2)/sr 
    sigma = 0.1
    gauss_array = 0
    for mu in (label_padded[1:]):
        gauss_array += gaussian(new_vector, mu, sigma)
    for i in range(len(gauss_array)):
        if gauss_array[i] > 1:
            gauss_array[i] = 1
    return image, gauss_array

def padding_MLS(image, label):
    """This function pads 30frames at the begining and end of an image"""
    sr = 44100
    hop_length = 1024
    window_size = 2048
    padding_factor = 50
    y = pink_noise.voss(padding_factor*hop_length-1)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=window_size, hop_length=hop_length, n_mels=80, fmin=80, fmax=16000)
    S_to_dB = librosa.power_to_db(S,ref=np.max)
    pad_image = S_to_dB[np.newaxis,:,:]

    #Pad MLS
    S_padded = np.concatenate((pad_image, image), axis=-1)
    S_padded = np.concatenate((S_padded, pad_image), axis=-1)
    return S_padded, label

def padding_SSLM(image, label):
    """This function pads 30frames at the begining and end of an image"""
    padding_factor = 50
    
    #Pad SSLM
    pad_image = np.full((image.shape[1], padding_factor), 1) 
    pad_image = pad_image[np.newaxis,:,:]
    S_padded = np.concatenate((pad_image, image), axis=-1)
    S_padded = np.concatenate((S_padded, pad_image), axis=-1)   
    return S_padded, label


def normalize_image(image, label):
    """This function normalies an image"""
    image = np.squeeze(image) #remove
    image = normalize(image)
    #image = (image-np.min(image))/(np.max(image)-np.min(image))
    image = np.expand_dims(image, axis=0)
    return image, label
 
    
class SSMDataset(Dataset):
    songs_list = []

    def __init__(self, images_path, labels_path, transforms=None):
        """lista de nombre de canciones"""
        self.images_path = images_path
        self.labels_path = labels_path        
        self.images_list = []
        self.labels_list = []
        self.labels_sec_list = []
        
        for (im_dirpath, im_dirnames, im_filenames) in os.walk(self.images_path): #images files of images path
            for (lab_dirpath, lab_dirnames, lab_filenames) in os.walk(self.labels_path): #labels files fo labels path
                for f in im_filenames: #loop in each images png name files (songs_IDs)
                    if f[:-4] in lab_dirnames: #if image name is annotated:
                        #images path 
                        if f.endswith('.npy'):
                            img_path = im_dirpath + f
                            image = np.load(img_path) #plt.imread si queremos abrir imagen
                            self.images_list.append(image)
                        #labels path
                        path = os.path.join(lab_dirpath, f[:-4] + "/parsed/")
                        txt1 = "textfile1_functions.txt"
                        txt2 = "textfile2_functions.txt"
                        if os.path.isfile(path + txt1):
                            txt = "textfile1_functions.txt"
                        elif os.path.isfile(path + txt2):
                            txt = "textfile2_functions.txt"
                        label_path = path + txt
                        label = np.asarray(ReadDataFromtxt(path, txt), dtype=np.float32)
                        labels_sec = np.asarray(ReadDataFromtxt(path, txt), dtype=np.float32)
           
                        self.labels_list.append(label)
                        self.labels_sec_list.append(labels_sec)
        self.transforms = transforms

    # [proc.open_files() for proc in psutil.process_iter() if proc.pid == os.getpid()]
    def __len__(self):
        'contar canciones en lista'
        return len(self.images_list)

    def __getitem__(self, index):
        'coger cancion de lista'
        image = self.images_list[index]
        image = image[np.newaxis, :, :]
        label = self.labels_list[index]
        labels_sec = self.labels_sec_list[index]
        # From numpy to Torch Tensors
        # image = torch.from_numpy(image)
        # label = torch.from_numpy(label)
        if self.transforms is not None:
            for t in self.transforms:
                image, label = t(image, label)
        return image, label, labels_sec


def build_dataloader(batch_size, input_train_path, labels_path):

    dataset_train = SSMDataset(input_train_path, labels_path, transforms=[normalize_image, padding_MLS, borders])
    trainloader = DataLoader(dataset_train, batch_size=batch_size, num_workers=0)

    return dataset_train, trainloader


"""
batch_size = 1
im_path_mel = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\TEST\\np MLS\\"
im_path_L_MFCCs = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\TEST\\np SSLM from MFCCs euclidean 2pool3\\"
labels_path = "E:\\UNIVERSIDAD\\MÁSTER INGENIERÍA INDUSTRIAL\\TFM\\Database\\salami-data-public\\annotations\\"

mels_dataset = SSMDataset(im_path_mel, labels_path, transforms=[normalize_image, padding_MLS, borders])
mels_trainloader = DataLoader(mels_dataset, batch_size = batch_size, num_workers=0)

sslms_dataset = SSMDataset(im_path_L_MFCCs, labels_path, transforms=[normalize_image, padding_SSLM, borders])
sslms_trainloader = DataLoader(sslms_dataset, batch_size = batch_size, num_workers=0)
"""
"""
image = np.load(im_path_mel + "1358.npy")
plt.imshow(image, origin='lower')
plt.show()

mls = mels_dataset[80][0][0,...]
plt.imshow(mls, origin='lower')
plt.show()

sslm = sslms_dataset[80][0][0,...]
plt.imshow(sslm, origin='lower')
plt.show()
"""

"""
#Check if all MLSs and SSLMs and their respective labels have the same time values
print("==================Dimensions of MLS, SSLM and Labels==================")
for i in range(len(mels_trainloader)): #zip(mels_trainloader, sslms_trainloader):
    image_mel, label, label_sec = mels_dataset[i]
    image_sslm, label, label_sec = sslms_dataset[i]
    #plt.imshow(image_mel[0,...])
    #plt.show()
    #print("Tamaño im No:", i, "MLS:", image_mel.shape, "SSLM:", image_sslm.shape, "label:", label.shape)
    #if image.shape[1] != label.shape[0]:
        #print(i)
"""