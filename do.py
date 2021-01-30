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

adata = sc.read_h5ad("sp.h5ad")

from cell2location.plt.mapping_video import plot_spatial

# select up to 6 clusters
sel_clust = ['Oligo_2', 'Inh_Meis2_3', 'Inh_4', 'Ext_Thal_1', 'Ext_L23', 'Ext_L56']
sel_clust_col = ['q05_spot_factors' + str(i) for i in sel_clust]

slide = select_slide(adata, 'ST8059048')

# keys = slide.obs.keys()
# collect = []
# for key in keys:
#     if "spot" in key and "q95" in key:
#         print(key)
#         collect.append(key)
#     continue
#     flag = False
#     for sel in sel_clust:
#         if sel in key:
#             flag = True
#             break
#     if flag:
#         print(key)


# print(len(collect))
# exit(0)
selected_slide = slide.obs[sel_clust_col]
print(selected_slide)

with mpl.rc_context({'figure.figsize': (15, 15)}):
    fig = plot_spatial(slide.obs[sel_clust_col], labels=sel_clust,
                  coords=slide.obsm['spatial'] \
                          * list(slide.uns['spatial'].values())[0]['scalefactors']['tissue_hires_scalef'],
                  show_img=True, img_alpha=0.8,
                  style='fast', # fast or dark_background
                  img=list(slide.uns['spatial'].values())[0]['images']['hires'],
                  circle_diameter=6, colorbar_position='right')

plt.savefig("4.png")

exit(0)

adata_sc = preprocess_scdata(sc_data_folder, sc_data_name, cell_types_name)
adata = preprocess_spdata(sp_data_folder, data_list_name, data_block_name)
# print(adata)
# print(adata.obs)
# print(adata.obsm["spatial"])
# exit(0)
adata_vis = adata.copy()
adata_vis.raw = adata_vis

spatial_information = adata_vis.uns["spatial"]
keys = list(spatial_information.keys())

adata = adata[adata.obs["sample"].isin(keys[0].split("_")), :]

print(list(adata.uns['spatial'].values())[0]) # list(adata.uns['spatial'].values())[0]['scalefactors']['tissue_hires_scalef']

coords=adata.obsm['spatial'] \
                          * list(adata.uns['spatial'].values())[0]['scalefactors']['tissue_hires_scalef']
hires_image = spatial_information[keys[0]]["images"]["hires"]

print(hires_image.shape)
print(coords.shape)
exit(0)

inf_aver, adata_snrna_raw = sc_expression_signature(f"{reg_results_folder}/sc.h5ad", True)
# string = ""
# temp = [f"{x}, " for x in inf_aver.keys()]
# print(string.join(temp))
# exit(0)
inf_aver_t = np.array(inf_aver).transpose((1, 0))
print(np.max(inf_aver_t), np.min(inf_aver_t))
inf_aver_t = np.log10(inf_aver_t+1)
print(np.max(inf_aver_t), np.min(inf_aver_t))
inf_aver_t = (inf_aver_t-np.min(inf_aver_t)) / np.max(inf_aver_t) - np.min(inf_aver_t)
summation = np.sum(inf_aver, axis=1)
inf_aver_t_sort = inf_aver_t[:, (-1*summation).argsort()]

inf_aver_t_sort_part = inf_aver_t_sort[:, :50]

# red_min = 2; red_max = 255
# green_min = 255; green_max = 255
# blue_min = 255; blue_max = 2

# red_factor = (red_max - red_min) / 49
# green_factor = (green_max - green_min) / 49
# blue_factor = (blue_max - blue_min) / 49

# r = red_min - inf_aver_t_sort_part*(red_min - red_max)
# g = green_min - inf_aver_t_sort_part*(green_min - green_max)
# b = blue_min - inf_aver_t_sort_part*(blue_min - blue_max)

# print(r)
# print(g)
# print(b)

# rgb = np.stack([r, g, b], axis=-1)
# print(rgb.shape)

im = plt.imshow(inf_aver_t_sort_part, cmap="cool")
plt.colorbar(cmap="cool")
plt.title("The Reference Expression Signature of Highest Expression Gene (TOP50)")
plt.xlabel("Highest Expression Gene (Top50)")
plt.ylabel("Cell Type Index")
plt.savefig("1.png")

# print(dir(inf_aver))
# print(type(inf_aver))
# print(inf_aver)
# signature_array = np.array(inf_aver)
# print(adata_vis.var)
# print(signature_array.shape)
# print(inf_aver.index)
judge = adata_vis.var_names.isin(inf_aver.index)
# print([x for x in judge if x])
# print(np.sum(judge))
# print(adata_vis[:, ~judge].var_names)
selected_adata_vis = adata_vis[:, judge]
print(inf_aver[~inf_aver.index.isin(selected_adata_vis.var_names)])
selected_inf_aver = inf_aver[inf_aver.index.isin(selected_adata_vis.var_names)]
selected_inf_aver.to_csv("signature.csv")
selected_adata_vis.write_h5ad("visium.h5ad")

adata_sc = adata_sc[:, adata_sc.var_names.isin(selected_adata_vis.var_names)]
print(adata_sc)
adata_sc.var_names = adata_sc.var["SYMBOL"]
plt.clf()
# plt.figure(figsize=(2048, 2048))
sc.pl.highest_expr_genes(adata_sc, n_top=50, palette="hls")
plt.subplots_adjust(left=0.2, top=0.98, bottom=0.05)
plt.savefig("3.png")
exit(0)