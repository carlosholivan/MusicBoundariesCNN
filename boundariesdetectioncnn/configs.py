from pathlib import Path

class InputsConfig:
    HOP_LENGTH          =       1024
    SAMPLING_RATE       =       44100
    SAMPLES_FRAME       =       2048
    POOLING_FACTOR      =       6   
    PADDING_FACTOR      =       50
    LAMBDA              =       round(6*SAMPLING_RATE/HOP_LENGTH)
    WINDOW              =       3


class ParamsConfig:
    BATCH_SIZE          =       1
    NUM_EPOCHS          =       1000
    LEARNING_RATE       =       0.001
    OUT_CHANNELS        =       32   
    ITERATIONS          =       10
    WEIGHTS_PATH        =       "../weights/"


class PathsConfig:

    MELS_TRAIN_PATH                         =       "../Inputs/TRAIN/np MLS/"
    MELS_VAL_PATH                           =       "../Inputs/VALIDATION/np MLS/"
    MELS_TEST_PATH                          =       "../Inputs/TEST/np MLS/"
   
    SSLM_COS_CHROMAS_SMALL_TRAIN_PATH       =       "../Inputs/TRAIN/np SSLM from Chromas cosine 2pool3/"
    SSLM_COS_CHROMAS_SMALL_VAL_PATH         =       "../Inputs/VALIDATION/np SSLM from Chromas cosine 2pool3/"
    SSLM_COS_CHROMAS_SMALL_TEST_PATH        =       "../Inputs/TEST/np SSLM from Chromas cosine 2pool3/"

    SSLM_COS_CHROMAS_LARGE_TRAIN_PATH       =       "../Inputs/TRAIN/np SSLM from Chromas cosine/"
    SSLM_COS_CHROMAS_LARGE_VAL_PATH         =       "../Inputs/VALIDATION/np SSLM from Chromas cosine/"
    SSLM_COS_CHROMAS_LARGE_TEST_PATH        =       "../Inputs/TEST/np SSLM from Chromas cosine/"

    SSLM_EUCL_CHROMAS_SMALL_TRAIN_PATH      =       "../Inputs/TRAIN/np SSLM from Chromas euclidean 2pool3/"
    SSLM_EUCL_CHROMAS_SMALL_VAL_PATH        =       "../Inputs/VALIDATION/np SSLM from Chromas euclidean 2pool3/"
    SSLM_EUCL_CHROMAS_SMALL_TEST_PATH       =       "../Inputs/TEST/np SSLM from Chromas euclidean 2pool3/"

    SSLM_EUCL_CHROMAS_LARGE_TRAIN_PATH      =       "../Inputs/TRAIN/np SSLM from Chromas euclidean/"
    SSLM_EUCL_CHROMAS_LARGE_VAL_PATH        =       "../Inputs/VALIDATION/np SSLM from Chromas euclidean/"
    SSLM_EUCL_CHROMAS_LARGE_TEST_PATH       =       "../Inputs/TEST/np SSLM from Chromas euclidea/n"


    SSLM_COS_MFCCS_SMALL_TRAIN_PATH         =       "../Inputs/TRAIN/np SSLM from MFCCs cosine 2pool3/"
    SSLM_COS_MFCCS_SMALL_VAL_PATH           =       "../Inputs/VALIDATION/np SSLM from MFCCs cosine 2pool3/"
    SSLM_COS_MFCCS_SMALL_TEST_PATH          =       "../Inputs/TEST/np SSLM from MFCCs cosine 2pool3/"

    SSLM_COS_MFCCS_LARGE_TRAIN_PATH         =       "../Inputs/TRAIN/np SSLM from MFCCs cosine/"
    SSLM_COS_MFCCS_LARGE_VAL_PATH           =       "../Inputs/VALIDATION/np SSLM from MFCCs cosine/"
    SSLM_COS_MFCCS_LARGE_TEST_PATH          =       "../Inputs/TEST/np SSLM from MFCCs cosine/"

    SSLM_EUCL_MFCCS_SMALL_TRAIN_PATH        =       "../Inputs/TRAIN/np SSLM from MFCCs euclidean 2pool3/"
    SSLM_EUCL_MFCCS_SMALL_VAL_PATH          =       "../Inputs/VALIDATION/np SSLM from MFCCs euclidean 2pool3/"
    SSLM_EUCL_MFCCS_SMALL_TEST_PATH         =       "../Inputs/TEST/np SSLM from MFCCs euclidean 2pool3/"

    SSLM_EUCL_MFCCS_LARGE_TRAIN_PATH        =       "../Inputs/TRAIN/np SSLM from MFCCs euclidean/"
    SSLM_EUCL_MFCCS_LARGE_VAL_PATH          =       "../Inputs/VALIDATION/np SSLM from MFCCs euclidean/"
    SSLM_EUCL_MFCCS_LARGE_TEST_PATH         =       "../Inputs/TEST/np SSLM from MFCCs euclidean/"

    LABELS_PATH                             =       Path("E:\\UNIVERSIDAD\\MÁSTER INGENIERÍA INDUSTRIAL\\TFM\\Database\\salami-data-public\\annotations\\")
