import numpy as np

def normalize(array):
    """This function normalizes a matrix along x axis (frequency)"""
    normalized = np.zeros((array.shape[0], array.shape[1]))
    for i in range(array.shape[0]):
        normalized[i,:] = (array[i,:] - np.mean(array[i,:])) / np.std(array[i,:])
    return normalized

#a = np.array(([31,24,32,41],[54,65,73,82],[19,12,1,52],[14446,74,35,46],[1,0,0,3]))
#normalized = normalize_image(a)
