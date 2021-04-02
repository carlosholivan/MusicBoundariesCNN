
# Our modules
from boundariesdetectioncnn.data import dataloaders
from boundariesdetectioncnn.configs import PathsConfig, ParamsConfig

def test_dataloader_mels():

    batch_size = ParamsConfig.BATCH_SIZE
    input = 'mel'
    run = 'train'
    mels_dataset, mels_trainloader = dataloaders.build_dataloader(batch_size, input, run)

    assert len(mels_dataset) > 0
