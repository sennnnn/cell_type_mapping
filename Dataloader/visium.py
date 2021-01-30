import os
import torch

import scanpy as sc
import numpy as np

from torch.utils.data import Dataset


class VisiumDataset(Dataset):
    def __init__(self, params):
        self.params = params
        self.task = params["TASK"]
        self.if_cuda = params["CUDA"]
        self.adata_sp = sc.read_h5ad(params["SP_DATASET_PATH"])
        self.adata_sp_array = self.adata_sp.X.toarray()

    def __len__(self):
        return self.adata_sp_array.shape[0]

    def __getitem__(self, index):
        data = self.adata_sp_array[index]

        data = torch.tensor(data)
        # data = self._pre_process(data)
        # data = torch.unsqueeze(torch.tensor(data), dim=-1)

        if self.if_cuda:
            data = data.cuda()

        return data

    def _pre_process(self, data):
        params = self.params

        if self.task == "test" or self.task == "valid":
            tr_list = [
            ]
        elif self.task == "train":
            tr_list = [
            ]
        else:
            assert False, "{} underdefined task type.".format(self.task)

        for tr in tr_list:
            data = tr(data)

        return data
            

