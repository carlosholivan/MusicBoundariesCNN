
# Our modules
from boundariesdetectioncnn.data import dataloaders
from boundariesdetectioncnn.configs import PathsConfig, ParamsConfig

def test_dataloader_mels():

    input_path = PathsConfig.MELS_PATH
    labels_path = PathsConfig.LABELS_PATH
    batch_size = ParamsConfig.BATCH_SIZE

    mels_dataset, mels_trainloader = dataloaders.build_dataloader(batch_size, im_path_mel, labels_path)

    assert len(mels_dataset) > 0
