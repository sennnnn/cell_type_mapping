from Dataloader.visium import VisiumDataset

from torch.utils.data import DataLoader


def construct_dataloader(params):
    if params["DATASET_SELECTION"] == "visium":
        dataset = VisiumDataset(params)
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = params["BATCH_SIZE"],
            shuffle = True,
            num_workers = 0
        )
        return dataloader