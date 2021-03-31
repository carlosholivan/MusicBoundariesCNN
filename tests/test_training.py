import os

# Our modules
from boundariesdetectioncnn.train import train_model


def test_train_model():

    model = "mels"

    #Train 1 epoch and save weights
    train_model.run_training(model=model, epochs=2, save_epoch=1)

    assert os.path.exists("../weights/saved_model_1epochs.bin")
