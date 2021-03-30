import os
import shutil

#Paths fijos
train = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\TRAIN\\np MLS"
val = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\VALIDATION\\np MLS"
test = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\TEST\\np MLS"

#Paths origen - destino
p = "np SSLM from Chromas euclidean 2pool3\\"

path = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\" + p

dest_train = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\TRAIN\\" + p
dest_val = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\VALIDATION\\" + p
dest_test = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\TEST\\" + p

if not os.path.exists(dest_train):
    os.makedirs(dest_train)
if not os.path.exists(dest_val):
    os.makedirs(dest_val)
if not os.path.exists(dest_test):
    os.makedirs(dest_test)
   
#Copy in train path
for mels_train in os.listdir(path):
    for matrices in os.listdir(train): 
        if mels_train == matrices:
            shutil.copy(path + mels_train, dest_train)

#Copy in validation path
for mels_val in os.listdir(path):
    for matrices_val in os.listdir(val): 
        if mels_val == matrices_val:
            shutil.copy(path + mels_val, dest_val)

#Copy in test path
for mels_test in os.listdir(path):
    for matrices_test in os.listdir(test): 
        if mels_test == matrices_test:
            shutil.copy(path + mels_test, dest_test)
