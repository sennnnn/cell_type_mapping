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

# mapping 
mapping_results_folder = os.path.join(results_folder, "mapping")

# Get cell type expression signature
# adata_snrna_raw = preprocess_scdata(sc_data_folder, sc_data_name, types_name)
# r, adata_snrna_raw = sc_expression_regression(adata_snrna_raw, "Backup")

# PLOT QC FOR EACH SAMPLE
# fig, axs = plt.subplots(len(slides), 4, figsize=(15, 4*len(slides)-4))
# for i, s in enumerate(adata.obs['sample'].unique()):
#     #fig.suptitle('Covariates for filtering')

#     slide = select_slide(adata, s)
#     sns.distplot(slide.obs['total_counts'],
#                  kde=False, ax = axs[i, 0])
#     axs[i, 0].set_xlim(0, adata.obs['total_counts'].max())
#     axs[i, 0].set_xlabel(f'total_counts | {s}')

#     sns.distplot(slide.obs['total_counts']\
#                  [slide.obs['total_counts']<20000],
#                  kde=False, bins=40, ax = axs[i, 1])
#     axs[i, 1].set_xlim(0, 20000)
#     axs[i, 1].set_xlabel(f'total_counts | {s}')

#     sns.distplot(slide.obs['n_genes_by_counts'],
#                  kde=False, bins=60, ax = axs[i, 2])
#     axs[i, 2].set_xlim(0, adata.obs['n_genes_by_counts'].max())
#     axs[i, 2].set_xlabel(f'n_genes_by_counts | {s}')

#     sns.distplot(slide.obs['n_genes_by_counts']\
#                  [slide.obs['n_genes_by_counts']<6000],
#                  kde=False, bins=60, ax = axs[i, 3])
#     axs[i, 3].set_xlim(0, 6000)
#     axs[i, 3].set_xlabel(f'n_genes_by_counts | {s}')

# plt.tight_layout()
# plt.savefig("3.png")

adata = preprocess_spdata(sp_data_folder, data_list_name, data_block_name)

slide = select_slide(adata, "ST8059048")

# print(slide)
# print("--------------------------")
# print(slide.X.shape)
# print("--------------------------")
# # print(np.sum(np.array(adata.obs["in_tissue"] == 0)))
# # print(np.max(np.array(adata.obs["array_col"])))
# # print(np.max(np.array(adata.obs["array_row"])))
# # print("---------------------------")
# # print(adata.var.keys())
# print(slide.uns["spatial"])
# exit(0)

# with mpl.rc_context({'figure.figsize': [6,7],
#                      'axes.facecolor': 'white'}):
#    sc.pl.spatial(slide, img_key = "hires", cmap='magma',
#                   library_id=list(slide.uns['spatial'].keys())[0],
#                   color=['total_counts', 'n_genes_by_counts'], size=1,
#                   gene_symbols='SYMBOL', show=False, return_fig=True)
      
#    plt.savefig("4.png")

# with mpl.rc_context({'figure.figsize': [6,7],
#                      'axes.facecolor': 'black'}):
#    sc.pl.spatial(slide,
#                color=["Rorb", "Vip"], img_key=None, size=1,
#                vmin=0, cmap='magma', vmax='p99.0',
#                gene_symbols='SYMBOL'
#                )
   
#    plt.savefig("5.png")

adata_vis = adata.copy()
adata_vis.raw = adata_vis

s = ["ST8059048", "ST8059052"]
adata_vis = adata_vis[adata_vis.obs["sample"].isin(s),:]

## snRNAseq reference (raw counts)
adata_snrna_raw = sc.read(f"{reg_results_folder}/sc.h5ad")

# Column name containing cell type annotations
covariate_col_names = 'annotation_1'

# Extract a pd.DataFrame with signatures from anndata object
inf_aver = adata_snrna_raw.raw.var.copy()
inf_aver = inf_aver.loc[:, [f'means_cov_effect_{covariate_col_names}_{i}' for i in adata_snrna_raw.obs[covariate_col_names].unique()]]
inf_aver.columns = [sub(f'means_cov_effect_{covariate_col_names}_{i}', '', i) for i in adata_snrna_raw.obs[covariate_col_names].unique()]
inf_aver = inf_aver.iloc[:, inf_aver.columns.argsort()]

# normalise by average experiment scaling factor (corrects for sequencing depth)
inf_aver = inf_aver * adata_snrna_raw.uns['regression_mod']['post_sample_means']['sample_scaling'].mean()

print(inf_aver)
exit(0)

# with mpl.rc_context({'figure.figsize': [10, 10],
#                      'axes.facecolor': 'white'}):
#    sc.pl.umap(adata_snrna_raw, color=['annotation_1'], size=15,
#             color_map = 'RdPu', ncols = 1, legend_loc='on data',
#             legend_fontsize=10)

#    plt.savefig("7.png")

# now we don't need to keep the scRNA-seq data set and a list with slides in memory
# del adata_snrna_raw, slides
# gc.collect()

# selecting most informative genes based on specificity
selection_specificity = 0.07

# normalise expression signatures:
cell_state_df_norm = (inf_aver.T / inf_aver.sum(1)).T
# apply cut off:
cell_state_df_norm = (cell_state_df_norm > selection_specificity)

# check the number of markers per cell type
cell_state_df_norm.sum(0), (cell_state_df_norm.sum(1) > 0).sum(0)

sc.settings.set_figure_params(dpi = 100, color_map = 'viridis', dpi_save = 100,
                              vector_friendly = True, format = 'pdf',
                              facecolor='white')

r = cell2location.run_cell2location(

      # Single cell reference signatures as pd.DataFrame
      # (could also be data as anndata object for estimating signatures
      #  as cluster average expression - `sc_data=adata_snrna_raw`)
      sc_data=inf_aver,
      # Spatial data as anndata object
      sp_data=adata_vis,

      # the column in sc_data.obs that gives cluster idenitity of each cell
      summ_sc_data_args={'cluster_col': "annotation_1",
                         # select marker genes of cell types by specificity of their expression signatures
                         'selection': "cluster_specificity",
                         # specificity cutoff (1 = max, 0 = min)
                         'selection_specificity': 0.07
                        },

      train_args={'use_raw': True, # By default uses raw slots in both of the input datasets.
                  'n_iter': 40000, # Increase the number of iterations if needed (see QC below)

                  # Whe analysing the data that contains multiple experiments,
                  # cell2location automatically enters the mode which pools information across experiments
                  'sample_name_col': 'sample'}, # Column in sp_data.obs with experiment ID (see above)


      export_args={'path': mapping_results_folder, # path where to save results
                   'run_name_suffix': '' # optinal suffix to modify the name the run
                  },

      model_kwargs={ # Prior on the number of cells, cell types and co-located groups

                    'cell_number_prior': {
                        # - N - the expected number of cells per location:
                        'cells_per_spot': 8,
                        # - A - the expected number of cell types per location:
                        'factors_per_spot': 9,
                        # - Y - the expected number of co-located cell type groups per location
                        'combs_per_spot': 5
                    },

                     # Prior beliefs on the sensitivity of spatial technology:
                    'gene_level_prior':{
                        # Prior on the mean
                        'mean': 1/2,
                        # Prior on standard deviation,
                        # a good choice of this value should be at least 2 times lower that the mean
                        'sd': 1/4
                    }
      }
      
)

exit(0)
# adata_snrna_raw = anndata.read_h5ad(os.path.join(sc_data_folder, "all_cells_20200625.h5ad"))
# attribute = dir(adata_snrna_raw.var)
# print([x for x in attribute if "index" in x])
# print(adata_snrna_raw.var.keys())
# print(adata_snrna_raw.var.index)
# print(type(adata_snrna_raw.var))
# print(adata_snrna_raw.var.first_valid_index)
# print(adata_snrna_raw.var.SYMBOL)
# exit(0)
labels = pd.read_csv(os.path.join(sc_data_folder, "snRNA_annotation_astro_subtypes_refined59_20200823.csv"), index_col=0)
# print(labels)
# print(len(adata_snrna_raw.obs_names))
labels = labels.reindex(index=adata_snrna_raw.obs_names)
adata_snrna_raw.obs[labels.columns] = labels
adata_snrna_raw = adata_snrna_raw[~adata_snrna_raw.obs['annotation_1'].isna(), :]
# print(adata_snrna_raw[1].obs)
# print(adata_snrna_raw[1, 1].var)
# print(~labels.isna())
# print(labels[labels['annotation_1'].isna()])
# print(~adata_snrna_raw.obs['annotation_1'].isna())
# print(dir(adata_snrna_raw.obs))
# print(adata_snrna_raw.obs.keys())
# print(adata_snrna_raw.obs[n_genes_by_counts])
# print(adata_snrna_raw)
# print(dir(labels))
# print(adata_snrna_raw)
# print(dir(adata_snrna_raw))
# print(adata_snrna_raw.obs)
# print(adata_snrna_raw.var)

# remove cells and genes with 0 counts everywhere
# print(adata_snrna_raw)
# print(dir(adata_snrna_raw))
# print(adata_snrna_raw.var[adata_snrna_raw.var["n_cells_by_counts"] == 0])
# print(adata_snrna_raw.obs[adata_snrna_raw.obs["n_genes_by_counts"] == 0])
sc.pp.filter_cells(adata_snrna_raw, min_genes=1)
sc.pp.filter_genes(adata_snrna_raw, min_cells=1)
# print(dir(adata_snrna_raw))
# print(adata_snrna_raw.X)

cell_type = adata_snrna_raw.obs["annotation_1"]
cell_type = set(list(cell_type))
cell_type_num = len(cell_type)
print(adata_snrna_raw.obs.keys())
exit(0)

# calculate the mean of each gene across non-zero cells
# .X 为 cellxgene 的表达表
adata_snrna_raw.var['n_cells'] = (adata_snrna_raw.X.toarray() > 0).sum(0)
adata_snrna_raw.var['nonz_mean'] = adata_snrna_raw.X.toarray().sum(0) / adata_snrna_raw.var['n_cells']

plt.hist2d(np.log10(adata_snrna_raw.var['nonz_mean']),
           np.log10(adata_snrna_raw.var['n_cells']), bins=200,
           norm=mpl.colors.LogNorm(),
           range=[[0, 0.5], [1, 4.5]]);
nonz_mean_cutoff = 0.05
cell_count_cutoff = np.log10(adata_snrna_raw.shape[0] * 0.0005)
cell_count_cutoff2 = np.log10(adata_snrna_raw.shape[0] * 0.03)
# print(adata_snrna_raw.X.shape)
# exit(0)
# plt.vlines(nonz_mean_cutoff, cell_count_cutoff, cell_count_cutoff2);
# plt.hlines(cell_count_cutoff, nonz_mean_cutoff, 1);
# plt.hlines(cell_count_cutoff2, 0, nonz_mean_cutoff);
# plt.show()

# adata_snrna_raw[:,(np.array(np.log10(adata_snrna_raw.var['nonz_mean']) > nonz_mean_cutoff)
#          | np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff2))
#       & np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff)].shape

# select genes based on mean expression in non-zero cells
adata_snrna_raw = adata_snrna_raw[:,(np.array(np.log10(adata_snrna_raw.var['nonz_mean']) > nonz_mean_cutoff)
         | np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff2))
      & np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff)
              & np.array(~adata_snrna_raw.var['SYMBOL'].isna())]

adata_snrna_raw.raw = adata_snrna_raw

temp = adata_snrna_raw.X.toarray()
# temp = np.log10(temp + 1)
# temp = (temp - np.min(temp))*255 / np.max(temp) - np.min(temp)
temp_ = temp.copy()
temp_[temp != 0] = 0
temp_[temp == 0] = 255
print(np.max(temp_), np.min(temp_), np.sum(temp_ == 0), np.sum(temp_ != 0))
print(temp_.shape)
plt.close()
plt.imshow(temp_, cmap="gray")
plt.savefig("2.png")
exit(0)

#########################
adata_snrna_raw.X = adata_snrna_raw.raw.X.copy()
sc.pp.log1p(adata_snrna_raw)

sc.pp.scale(adata_snrna_raw, max_value=10)
sc.tl.pca(adata_snrna_raw, svd_solver='arpack', n_comps=80, use_highly_variable=False)

# Plot total counts over PC to check whether PC is indeed associated with total counts
#sc.pl.pca_variance_ratio(adata_snrna_raw, log=True)
#sc.pl.pca(adata_snrna_raw, color=['total_counts'],
#          components=['0,1', '2,3', '4,5', '6,7', '8,9', '10,11', '12,13'],
#          color_map = 'RdPu', ncols = 3, legend_loc='on data',
#          legend_fontsize=10, gene_symbols='SYMBOL')

# remove the first PC which explains large amount of variance in total UMI count (likely technical variation)
adata_snrna_raw.obsm['X_pca'] = adata_snrna_raw.obsm['X_pca'][:, 1:]
adata_snrna_raw.varm['PCs'] = adata_snrna_raw.varm['PCs'][:, 1:]
#########################

import bbknn
bbknn.bbknn(adata_snrna_raw, neighbors_within_batch = 3, batch_key = 'sample', n_pcs = 79)
sc.tl.umap(adata_snrna_raw, min_dist = 0.8, spread = 1.5)

#########################

adata_snrna_raw = adata_snrna_raw[adata_snrna_raw.obs['annotation_1'].argsort(),:]

with mpl.rc_context({'figure.figsize': [10, 10],
                     'axes.facecolor': 'white'}):
   sc.pl.umap(adata_snrna_raw, color=['annotation_1'], size=15,
               color_map = 'RdPu', ncols = 1, legend_loc='on data',
               legend_fontsize=10)
   plt.savefig("1.png")

results_folder = "temp/"

from cell2location import run_regression
r, adata_snrna_raw = run_regression(adata_snrna_raw, # input data object]

                   verbose=True, return_all=True,

                   train_args={
                    'covariate_col_names': ['annotation_1'], # column listing cell type annotation
                    'sample_name_col': 'sample', # column listing sample ID for each cell

                    # column listing technology, e.g. 3' vs 5',
                    # when integrating multiple single cell technologies corresponding
                    # model is automatically selected
                    'tech_name_col': None,

                    'stratify_cv': 'annotation_1', # stratify cross-validation by cell type annotation

                    'n_epochs': 100, 'minibatch_size': 1024, 'learning_rate': 0.01,

                    'use_cuda': True, # use GPU?

                    'train_proportion': 0.9, # proportion of cells in the training set (for cross-validation)
                    'l2_weight': True,  # uses defaults for the model

                    'readable_var_name_col': 'SYMBOL', 'use_raw': True},

                   model_kwargs={}, # keep defaults
                   posterior_args={}, # keep defaults

                   export_args={'path': results_folder, # where to save results
                                'save_model': True, # save pytorch model?
                                'run_name_suffix': ''})

reg_mod = r['mod']

reg_mod_name = 'RegressionGeneBackgroundCoverageTorch_65covariates_40532cells_12819genes'
reg_path = f'{results_folder}/{reg_mod_name}/'

## snRNAseq reference (raw counts)
adata_snrna_raw = sc.read(f'{reg_path}sc.h5ad')
## model
r = pickle.load(file = open(f'{reg_path}model_.p', "rb"))
reg_mod = r['mod']

# Export cell type expression signatures:
covariate_col_names = 'annotation_1'

inf_aver = adata_snrna_raw.raw.var.copy()
inf_aver = inf_aver.loc[:, [f'means_cov_effect_{covariate_col_names}_{i}' for i in adata_snrna_raw.obs[covariate_col_names].unique()]]
from re import sub
inf_aver.columns = [sub(f'means_cov_effect_{covariate_col_names}_{i}', '', i) for i in adata_snrna_raw.obs[covariate_col_names].unique()]
inf_aver = inf_aver.iloc[:, inf_aver.columns.argsort()]

# scale up by average sample scaling factor
inf_aver = inf_aver * adata_snrna_raw.uns['regression_mod']['post_sample_means']['sample_scaling'].mean()
