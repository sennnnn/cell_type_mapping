import torch
import torch.nn as nn

import pandas as pd
import matplotlib.pyplot as plt

import argparse

from Dataloader import *
from Modeling import *

from Tools.loss_utils import *
from Tools.metric_utils import *
from Tools.lr_utils import *
from Tools.ckpt_utils import *
from Tools.record_utils import *
from Tools.utils import *


def train(params):
    # Data Loader
    dataloader           = construct_dataloader(params)
    total_step_per_epoch = len(dataloader)
    # Inference Related Things
    model        = construct_model(params)
    loss         = construct_loss(params)
    optimizer    = torch.optim.Adam([{"params": model.parameters(), "lr": params["LR"]}])
    lr_scheduler = LrScheduler("poly", params["LR"], params["TOTAL_EPOCHES"], total_step_per_epoch, params["WARM_UP_EPOCHES"])
    # Ckpt Tools
    save_tool    = Saver(params["BACKUP_PATH"], params["CUSTOM_KEY"], params["LOSS_TYPE"], "best")    
    zip_file     = save_tool.restore()
    if zip_file != None:
        load_state            = model.load_state_dict(zip_file["model_params"])
        params["START_EPOCH"] = zip_file["epoch"] + 1
    # Record Tools
    record_tool = TrainRecorder(params["BACKUP_PATH"], params["CUSTOM_KEY"], params["LOSS_TYPE"], params["START_EPOCH"])
    # Signature Load
    signature = pd.read_csv(params["SIGNATURE_PATH"], index_col=0)
    signature_array = np.array(signature)
    signature_array_t = signature_array.T
    signature_tensor = torch.tensor(signature_array_t).float()
    if params["CUDA"]:
        signature_tensor = signature_tensor.cuda()
    # Main Loop
    for epoch in range(params["START_EPOCH"] - 1, params["TOTAL_EPOCHES"]):
        loss_epoch_avg = 0
        DSC_epoch_avg  = 0
        for step, batch in enumerate(dataloader):
            # Inference
            input_data = batch
            predict = model(input_data)
            predict = torch.matmul(predict, signature_tensor)
            loss_variable = loss(predict, input_data)
            # Learning Rate Schedule
            lr_scheduler(optimizer, step + 1, epoch + 1)
            # Gradient Back Propogation
            optimizer.zero_grad()
            loss_variable.backward()
            optimizer.step()
            # To CPU
            logit  = predict.data.cpu().numpy()
            target = input_data.cpu().numpy()
            # Loss Value and Metric Value
            loss_value          = loss_variable.item()
            loss_epoch_avg      += loss_value
            # Log Information Print
            loss_string   = "Loss: {:+.3f}".format(loss_value)
            log_string    = "Epoch: {:<3} Step: {:>4}/{:<4} {}".format(epoch + 1, step + 1, total_step_per_epoch, loss_string)
            print(log_string)
            # Record Middle Result
            record_tool.write([epoch + 1, loss_value], log_string)

        loss_epoch_avg /= total_step_per_epoch
        # params["TASK"] = "test"
        # total_result = test(params, model, DSC_epoch_avg, epoch + 1)
        # params["TASK"] = "train"
        # save_tool.save(epoch + 1, total_result["DSC"][0], model)
        save_tool.save(epoch + 1, loss_epoch_avg, model)


def test(params, model=None, train_performance=None, epoch=None):
   pass


if __name__ == "__main__":
    params = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="autoencoder")
    parser.add_argument("--dataset_name", type=str, default="visium")
    parser.add_argument("--GPU_number", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    # Data load settings
    params["BATCH_SIZE"] = args.batch_size
    params["IF_SHUFFLE"] = True
    params["DATASET_SELECTION"] = args.dataset_name
    params["SP_DATASET_PATH"] = "Data/visium_dataset.h5ad"
    params["SIGNATURE_PATH"] = "Data/signature.csv"
    # Data augmentation settings
    # Model settins
    params["IN_CHANNELS"] = 7801
    params["OUT_CHANNELS"] = 59
    params["HIDDEN_CHANNELS"] = 1024
    # Task settings
    params["TASK"] = "train"
    params["LR"] = 0.0001
    params["CUDA"] = True
    params["START_EPOCH"] = 1
    params["WARM_UP_EPOCHES"] = 1
    params["TOTAL_EPOCHES"] = 100
    params["LOSS_TYPE"] = "MSE"
    params["LOSS_ARGS"] = (0.5, 2)
    params["MODEL_SELECTION"] = args.model_name
    # Other settings
    params["GPU_VALID_NUMBER"] = str(args.GPU_number)
    params["CUSTOM_KEY"] = f"{params['MODEL_SELECTION']}"
    params["BACKUP_PATH"] = f"Backup/{params['CUSTOM_KEY']}"

    os.environ["CUDA_VISIBLE_DEVICES"] = params["GPU_VALID_NUMBER"]

    if params["TASK"]   == "train":
        train(params)
    elif params["TASK"] == "test":
        test(params)