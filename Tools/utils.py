import os

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl

from re import sub
from cell2location import run_regression, run_cell2location


def select_slide(adata, s, s_col="sample"):
    r""" This function selects the data for one slide from the spatial anndata object.

    :param adata: Anndata object with multiple spatial experiments
    :param s: name of selected experiment
    :param s_col: column in adata.obs listing experiment name for each location
    """

    slide = adata[adata.obs[s_col].isin([s]), :]
    s_keys = list(slide.uns['spatial'].keys())
    s_spatial = np.array(s_keys)[[s in k for k in s_keys]][0]

    slide.uns['spatial'] = {s_spatial: slide.uns['spatial'][s_spatial]}

    return slide


def read_and_qc(sample_name, path):
    r""" This function reads the data for one 10X spatial experiment into the anndata object.
    It also calculates QC metrics. Modify this function if required by your workflow.

    :param sample_name: Name of the sample
    :param path: path to data
    """

    adata = sc.read_visium(path + str(sample_name),
                           count_file='filtered_feature_bc_matrix.h5', load_images=True)

    adata.obs['sample'] = sample_name
    adata.var['SYMBOL'] = adata.var_names
    adata.var.rename(columns={'gene_ids': 'ENSEMBL'}, inplace=True)
    adata.var_names = adata.var['ENSEMBL']
    adata.var.drop(columns='ENSEMBL', inplace=True)

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata.var['mt'] = [gene.startswith('mt-') for gene in adata.var['SYMBOL']]
    adata.obs['mt_frac'] = adata[:, adata.var['mt'].tolist()].X.sum(1).A.squeeze()/adata.obs['total_counts']

    # add sample name to obs names
    adata.obs["sample"] = [str(i) for i in adata.obs['sample']]
    adata.obs_names = adata.obs["sample"] \
                          + '_' + adata.obs_names
    adata.obs.index.name = 'spot_id'

    return adata


def umap_representation(adata_snrna_raw):
    sc.pp.log1p(adata_snrna_raw)
    sc.pp.scale(adata_snrna_raw, max_value=10)
    sc.tl.pca(adata_snrna_raw, svd_solver='arpack', n_comps=80, use_highly_variable=False)


def sc_expression_regression(adata_snrna_raw, results_folder):
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

                   export_args={'path': os.path.join(results_folder, "regression_model"), # where to save results
                                'save_model': True, # save pytorch model?
                                'run_name_suffix': ''})

    return r, adata_snrna_raw


def preprocess_scdata(sc_data_folder, sc_data_name, cell_types_name):
    sc_data_path = os.path.join(sc_data_folder, sc_data_name)
    types_path = os.path.join(sc_data_folder, cell_types_name)
    # Data Reading
    ## Single Cell Gene Experission Data
    adata_snrna_raw = sc.read_h5ad(sc_data_path)
    ## Label: Sample Index -- Cell Type
    ### The zero index col is the sample name with UMI.
    labels = pd.read_csv(types_path, index_col=0)
    # Preprocessing
    ## Store the cell type annotation into anndata object.
    labels = labels.reindex(index=adata_snrna_raw.obs_names)
    adata_snrna_raw.obs[labels.columns] = labels
    adata_snrna_raw = adata_snrna_raw[~adata_snrna_raw.obs["annotation_1"].isna(), :]
    ## Remove invalid item.
    sc.pp.filter_cells(adata_snrna_raw, min_genes=1)
    sc.pp.filter_genes(adata_snrna_raw, min_cells=1)
    ## .X is the CellxGene expression table.
    ## The first dimension is cell and the second dimension is gene.
    expression_table = adata_snrna_raw.X
    expression_array = expression_table.toarray()
    ## Calculate non-zero expression cell number of each gene.
    adata_snrna_raw.var['n_cells'] = (expression_array > 0).sum(0)
    ## Calculate average expression amount per cell of each gene.
    adata_snrna_raw.var['nonz_mean'] = expression_array.sum(0) / adata_snrna_raw.var['n_cells']
    ## Cut off lowly-expressed gene.
    cell_total_count = expression_array.shape[0]
    nonz_mean_cutoff = 0.05
    cell_count_cutoff = np.log10(cell_total_count * 0.0005)
    cell_count_cutoff2 = np.log10(cell_total_count * 0.03)
    ## First cutoff: Include all genes expressed by at least 3% of cells.
    adata_snrna_raw = adata_snrna_raw[:, np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff2)]
    ## Second cutoff: Include genes expressed by at least 0.05% of cells when they have high counts in non-zero cells.
    adata_snrna_raw = adata_snrna_raw[:, (np.array(np.log10(adata_snrna_raw.var['nonz_mean']) > nonz_mean_cutoff)
                                        & np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff))]
    ## Remove genes which has invalid gene name.
    adata_snrna_raw = adata_snrna_raw[:, np.array(~adata_snrna_raw.var["SYMBOL"].isna())]
    adata_snrna_raw.raw = adata_snrna_raw
    adata_snrna_raw.X = adata_snrna_raw.raw.X.copy()

    return adata_snrna_raw


def preprocess_spdata_single(data_folder, sample_name):
    adata = sc.read_visium(os.path.join(data_folder, sample_name), \
                           count_file="filtered_feature_bc_matrix.h5", load_images=True)

    adata.obs["sample"] = sample_name
    adata.var["SYMBOL"] = adata.var_names
    adata.var.rename(columns={"gene_ids": "ENSEMBL"}, inplace=True)
    adata.var_names = adata.var["ENSEMBL"]
    adata.var.drop(columns="ENSEMBL", inplace=True)

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata.var["mt"] = [gene.startswith("mt-") for gene in adata.var["SYMBOL"]]
    adata.obs["mt_frac"] = adata[:, adata.var["mt"].tolist()].X.sum(1).A.squeeze()/adata.obs["total_counts"]

    # add sample name to obs names
    adata.obs["sample"] = [str(i) for i in adata.obs["sample"]]
    adata.obs_names = adata.obs["sample"] \
                          + "_" + adata.obs_names
    adata.obs.index.name = "spot_id"

    return adata


def preprocess_spdata(data_folder, data_list_name, data_block_name):
    data_list_path = os.path.join(data_folder, data_list_name)
    data_block_path = os.path.join(data_folder, data_block_name)

    # Read the list of spatial experiments
    sample_data = pd.read_csv(data_list_path)

    # Read the data into anndata objects
    slides = []
    for i in sample_data["sample_name"]:
        slides.append(preprocess_spdata_single(data_block_path, i))

    # Combine anndata objects together
    adata = slides[0].concatenate(
        slides[1:],
        batch_key="sample",
        uns_merge="unique",
        batch_categories=sample_data['sample_name'],
        index_unique=None
    )

    # Load visium spatial expression data.
    # mitochondria-encoded (MT) genes should be removed for spatial mapping
    # The "mt" coloumn of anndata.var is the judge about if this gene is the mitochondria-encoded genes.
    adata.obsm['mt'] = adata[:, adata.var['mt'].values].X.toarray()
    adata = adata[:, ~adata.var['mt'].values]

    return adata


def sc_expression_signature(path, if_raw=False):
    ## snRNAseq reference (raw counts)
    adata_snrna_raw = sc.read(path)

    # Column name containing cell type annotations
    covariate_col_names = 'annotation_1'

    # Extract a pd.DataFrame with signatures from anndata object
    inf_aver = adata_snrna_raw.raw.var.copy()
    inf_aver = inf_aver.loc[:, [f'means_cov_effect_{covariate_col_names}_{i}' for i in adata_snrna_raw.obs[covariate_col_names].unique()]]
    inf_aver.columns = [sub(f'means_cov_effect_{covariate_col_names}_{i}', '', i) for i in adata_snrna_raw.obs[covariate_col_names].unique()]
    inf_aver = inf_aver.iloc[:, inf_aver.columns.argsort()]

    # normalise by average experiment scaling factor (corrects for sequencing depth)
    inf_aver = inf_aver * adata_snrna_raw.uns['regression_mod']['post_sample_means']['sample_scaling'].mean()

    if if_raw:
        return inf_aver, adata_snrna_raw
    else:
        return inf_aver, None