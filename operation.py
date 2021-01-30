import os
import gc
import sys
import anndata
import cell2location

import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from re import sub
from matplotlib import rcParams
from cell2location import run_regression
from cell2location.plt.mapping_video import plot_spatial

from Tools.utils import *


if __name__ == "__main__":
    # sc
    sc_data_folder = "Data/sc"
    sc_data_name = "all_cells_20200625.h5ad"
    cell_types_name = "snRNA_annotation_astro_subtypes_refined59_20200823.csv"
    sc_results_folder = "Backup/sc"
    reg_results_folder = os.path.join(sc_results_folder, "regression_model", "RegressionGeneBackgroundCoverageTorch_65covariates_40532cells_7809genes")

    # sp 
    sp_data_folder = "Data/sp"
    data_list_name = "Visium_mouse.csv"
    data_block_name = "rawdata/"
    data_list_path = os.path.join(sp_data_folder, "Visium_mouse.csv")
    data_block_path = os.path.join(sp_data_folder, "rawdata/")

    # Load sc data and sp data
    adata_sc = preprocess_scdata(sc_data_folder, sc_data_name, cell_types_name)
    adata_sp = preprocess_spdata(sp_data_folder, data_list_name, data_block_name)

    adata_sp_vis = adata_sp.copy()
    adata_sp_vis.raw = adata_sp_vis

    inf_aver, adata_snrna_raw = sc_expression_signature(f"{reg_results_folder}/sc.h5ad", True)
    inf_aver_t = np.array(inf_aver).transpose((1, 0))
    inf_aver_t = np.log10(inf_aver_t+1)
    inf_aver_t = (inf_aver_t-np.min(inf_aver_t)) / np.max(inf_aver_t) - np.min(inf_aver_t)
    inf_aver_t_sort = inf_aver_t[:, (-1*np.sum(inf_aver_t, axis=1)).argsort()]
    inf_aver_t_sort_part = inf_aver_t_sort[:, :50]

    # Visualization part of reference gene expression signature
    im = plt.imshow(inf_aver_t_sort_part, cmap="cool")
    plt.colorbar(cmap="cool")
    plt.title("The Reference Expression Signature of Highest Expression Gene (TOP50)")
    plt.xlabel("Highest Expression Gene (Top50)")
    plt.ylabel("Cell Type Index")
    plt.savefig("1.png")

    # Get selected spatial transcription and singel cell transcription data.
    judge = adata_sp_vis.var_names.isin(inf_aver.index)
    selected_adata_sp_vis = adata_sp_vis[:, judge]
    selected_inf_aver = inf_aver[inf_aver.index.isin(selected_adata_sp_vis.var_names)]
    selected_inf_aver.to_csv("signature.csv")
    selected_adata_sp_vis.write_h5ad("visium.h5ad")

    # Visualization top 50 highest expression gene.
    selected_adata_sc = adata_sc[:, adata_sc.var_names.isin(selected_adata_sp_vis.var_names)]
    selected_adata_sc.var_names = selected_adata_sc.var["SYMBOL"]
    sc.pl.highest_expr_genes(adata_sc, n_top=50, palette="hls")
    plt.subplots_adjust(left=0.2, top=0.98, bottom=0.05)
    plt.savefig("3.png")

    