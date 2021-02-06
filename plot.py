import matplotlib as mpl

import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

from Tools.utils import select_slide
from cell2location.plt.mapping_video import plot_spatial


def train_val_curve(train_csv_path, val_csv_path, save_path="train_val_curve.png"):
    train_csv = pd.read_csv(train_csv_path)
    val_csv = pd.read_csv(val_csv_path)

    train_collect = []
    for i in range(1, 101):
        item = np.average(np.array(train_csv["Loss"][train_csv["Epoch"] == i]))
        train_collect.append(item)

    index = np.arange(1, 101)
    train_collect = np.array(train_collect)
    val_collect = np.array(val_csv["Cos"])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    p1 = ax1.plot(index, train_collect, color="#FF02FF", label="Loss")
    p2 = ax2.plot(index, val_collect, color="#02FFFF", label="Cos")
    
    ax1.set_ylim((0, 300))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    ax2.set_ylim((0, 1))
    ax2.set_ylabel("Cos")
    ax2.legend(loc="upper right")
    
    plt.title("Train-Val Curve")
    plt.savefig(save_path)


def cell_type_annotation(adata, slide_name, cell_types, save_path="cell_type_annotation.png"):
    if isinstance(adata, str):
        adata = sc.read(adata)
    slide = select_slide(adata, slide_name)
    with mpl.rc_context({'figure.figsize': (15, 15)}):
        fig = plot_spatial(slide.obs[cell_types], labels=cell_types,
                    coords=slide.obsm['spatial'] \
                            * list(slide.uns['spatial'].values())[0]['scalefactors']['tissue_hires_scalef'],
                    show_img=True, img_alpha=0.8,
                    style='fast', # fast or dark_background
                    img=list(slide.uns['spatial'].values())[0]['images']['hires'],
                    circle_diameter=6, colorbar_position='right')
        plt.savefig(save_path)

# train_val_curve("autoencoder@CMSE_train.csv", "autoencoder@CMSE_val.csv")

cell_type_annotation("Data/result.h5ad", "ST8059048", ["Ext_L56"])