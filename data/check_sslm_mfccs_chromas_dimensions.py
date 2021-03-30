"""
Este script comprueba que la dimension temporal (eje x) de las lag matrices
obtenidas a partir de los chromas y de los MFCCs sean iguales
"""

import os
import numpy as np

sslms_chromas_path = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\np SSLM from Chromas\\"
sslms_mfccs_path = "E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\np SSLM from MFCCs\\"

for b, d, chromas in os.walk(sslms_chromas_path):
    for b, d, mfccs in os.walk(sslms_mfccs_path):
        for i in range(len(chromas)):
            sslms_chromas = np.load(sslms_chromas_path + chromas[i])
            sslms_mfccs = np.load(sslms_mfccs_path + mfccs[i])
            if sslms_chromas.shape[1] != sslms_mfccs.shape[1]:
                print("Song:",chromas[i],"has SSLM Chroma:", sslms_chromas.shape, "SSLM MFCCs:", sslms_mfccs.shape)
                print("sslms matrices have not the same dimenion")

#%%
"""
Esta parte del script comprueba que las SSM obtenidas a partir de los chromas
y de los MFCCs tienen las mismas dimensiones
"""


#%%
"""
Esta parte del script comprueba que tanto los MLS como las SSMs y SSLMs 
obtenidas a partir de los chromas y MFCCs tengan la misma dimensión temporal
"""
