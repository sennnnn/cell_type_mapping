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

from Tools.utils import *

data_type = 'float32'

# this line forces theano to use the GPU and should go before importing cell2location
# os.environ["THEANO_FLAGS"] = 'device=cuda0,floatX=' + data_type + ',force_device=True'
# if using the CPU uncomment this:
#os.environ["THEANO_FLAGS"] = 'device=cpu,floatX=float32,openmp=True,force_device=True'

# silence scanpy that prints a lot of warnings
import warnings
warnings.filterwarnings('ignore')


results_folder = "Backup"

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

adata = preprocess_spdata(sp_data_folder, data_list_name, data_block_name)

slide = select_slide(adata, "ST8059048")

adata_vis = adata.copy()
adata_vis.raw = adata_vis

s = ["ST8059048", "ST8059052"]
adata_vis = adata_vis[adata_vis.obs["sample"].isin(s),:]

inf_aver = sc_expression_signature(f"{reg_results_folder}/sc.h5ad")

# now we don't need to keep the scRNA-seq data set and a list with slides in memory
del adata_snrna_raw, slides
gc.collect()