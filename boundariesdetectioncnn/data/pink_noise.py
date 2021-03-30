import numpy as np
import thinkdsp
import pandas as pd
import thinkplot
import librosa
import librosa.display
import matplotlib.pyplot as plt

def voss(nrows, ncols=16):
    """Generates pink noise using the Voss-McCartney algorithm.
    
    nrows: number of values to generate
    rcols: number of random sources to add
    
    returns: NumPy array
    """
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)
    
    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return total.values

"""
window_size = 2048 #(samples/frame)
hop_length = 1024 #overlap 50% (samples/frame) 
sr = 44100
padding_factor = 50
y = voss(padding_factor*hop_length-1)
 
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=window_size, hop_length=hop_length, n_mels=80, fmin=80, fmax=16000)
S_to_dB = librosa.power_to_db(S,ref=np.max)

#Plot Spectrogram
plt.figure(1, figsize=(10, 4))
librosa.display.specshow(S_to_dB, y_axis='mel', sr=sr, fmax=16000,x_axis='frames')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()


#Audio
wave = thinkdsp.Wave(y)
wave.unbias()
wave.normalize()
wave.plot()
wave.make_audio()
"""
