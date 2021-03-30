"""
SCRIPT NÚMERO: 2
Este script comprueba que todas las imágenes creadas no poseen ningún elemento
que sea nan. En caso de que un elemento de una de las matrices contenga un nan
lo convierte a 0.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

im_path_L_MFCCs = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\SSLM from MFCCs\\"
for i in range(len(os.listdir(im_path_L_MFCCs))):
    image = plt.imread(im_path_L_MFCCs + os.listdir(im_path_L_MFCCs)[i])
    if (image == float('NaN')).any():
        print("Image:",os.listdir(im_path_L_MFCCs)[i], "has nans")
    
"""
#Para una sola canción si es 
im_path_SSLM = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\np SSLM from MFCCs\\10.npy"
im_path_MLS = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\np MLS\\10.npy"
image_sslm = np.load(im_path_SSLM)
image_mls = np.load(im_path_MLS)

if (image_sslm.any() == float('NaN')) or (image_mls.any() == float('NaN')):
    print("Image:",image, "has nans.")
plt.imshow(image_sslm[:,:])
plt.show()
plt.imshow(image_mls[:,:])
plt.show()
"""


