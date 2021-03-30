import matplotlib.pyplot as plt
import numpy as np

ID_song = 10
song = str(ID_song) + ".npy"

mls = np.load("E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\np MLS\\" + song)
sslm_mfcc_cosine = np.load("E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\np SSLM from MFCCs 2pool3\\" + song)
sslm_chroma_cosine = np.load("E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\np SSLM from Chromas cosine 2pool3\\" + song)
sslm_mfcc_eucl = np.load("E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\np SSLM from MFCCs euclidean 2pool3\\" + song)
sslm_chroma_eucl = np.load("E:\\INVESTIGACIÓN\\Proyectos\\Boundaries Detection\\Inputs\\np SSLM from Chromas euclidean 2pool3\\" + song)

plt.subplot(5,1,1)
plt.imshow(mls, origin='lower', aspect=4)
plt.title("Mel Spectrogram (MLS)")
plt.subplot(5,1,2)
plt.title("Self-Similarity Lag Matrix from MFCCs and cosine distance (SSLM)")
plt.imshow(sslm_mfcc_cosine, origin='lower')
plt.subplot(5,1,3)
plt.title("Self-Similarity Lag Matrix from Chromas and cosine distance (SSLM)")
plt.imshow(sslm_chroma_cosine, origin='lower')
plt.subplot(5,1,4)
plt.title("Self-Similarity Lag Matrix from MFCCs and euclidean distance (SSLM)")
plt.imshow(sslm_mfcc_eucl, origin='lower')
plt.subplot(5,1,5)
plt.title("Self-Similarity Lag Matrix from Chromas and euclidean distance (SSLM)")
plt.imshow(sslm_chroma_eucl, origin='lower')
plt.show()

if sslm_mfcc_cosine.shape[0] == sslm_chroma_cosine.shape[1] and sslm_mfcc_cosine.shape[0] == sslm_mfcc_eucl.shape[0] and sslm_mfcc_cosine.shape[0] == sslm_chroma_eucl.shape[0]:
    print("Lag dimension of all the SSLMs are the same")
