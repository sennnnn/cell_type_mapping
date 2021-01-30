from visium import VisiumDataset

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


if __name__ == "__main__":
    params = {}
    params["DATASET_SELECTION"] = "visium"
    params["BATCH_SIZE"] = 4
    params["SP_DATASET_PATH"] = "../Data/visium_dataset.h5ad"
    params["TASK"] = "train"
    params["CUDA"] = True

    dataloader = construct_dataloader(params)
    for batch in dataloader:
        print(batch.shape)