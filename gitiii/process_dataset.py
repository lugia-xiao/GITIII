import os
import pandas as pd
import numpy as np
import scanpy as sc
import torch

from gitiii.find_ligands import search_ligands

def visualize_slide(df):
    adata=sc.AnnData(X=np.zeros((df.shape[0],3)))
    adata.obs["x"]=df["centerx"].values
    adata.obs["y"]=df["centery"].values
    adata.obs["cell_type"]=df["subclass"].values
    sc.pl.scatter(
        adata,
        x='x',#'position_x',
        y='y',#'position_y',
        color="cell_type"
    )

def get_spatial_relationship(df):
    x=df.loc[:,"centerx"].values
    y=df.loc[:,"centery"].values
    xx, yy = np.meshgrid(x, y)
    dx = xx - xx.T
    dy = yy - yy.T

    # Calculate pairwise Euclidean distances
    spatial = np.sqrt(dx**2 + dy**2)
    return spatial

def argsort_topk(arr, k):
    # Using argpartition to get indices of the k smallest values
    indices = np.argpartition(arr, k)[:k]

    # Optionally, sort the k indices if needed
    sorted_indices = indices[np.argsort(arr[indices])]
    return sorted_indices

def calculate_cell_type_exp(df,genes):
    cell_type_exp = {}
    cell_types = np.unique(df.loc[:, 'subclass'].values)
    for cell_typei in cell_types:
        exp_type = np.mean(df.loc[df.loc[:, 'subclass'] == cell_typei, genes].values, axis=0)
        cell_type_exp[cell_typei] = exp_type
    type_exp = []
    for i in range(df.shape[0]):
        type_exp.append(cell_type_exp[df.loc[i, 'subclass']])
    cell_exp = []
    for i in range(df.shape[0]):
        cell_exp.append(df.loc[i, genes] - cell_type_exp[df.loc[i, 'subclass']])
    type_exp = np.array(type_exp)
    cell_exp = np.array(cell_exp)

    all_var = np.mean(np.var(cell_exp + type_exp, axis=0))
    type_var = np.mean(np.var(type_exp, axis=0))
    state_var = np.mean(np.var(cell_exp, axis=0))
    print("All variance", all_var, "type_variance", type_var, type_var / all_var, "state_variance", state_var,
          state_var / all_var)
    return type_exp, cell_exp, cell_type_exp

def get_index(df, num_neighbor, threshold):
    indexes = []
    flags = []

    df.index = np.array(list(range(df.shape[0])))

    for i in df.index:
        xi = df.loc[i, "centerx"]
        yi = df.loc[i, "centery"]
        dx = np.abs(df["centerx"].values - xi)
        dy = np.abs(df["centery"].values - yi)
        dx[dx > 1e4] = 1e4
        dy[dy > 1e4] = 1e4
        distancei = np.sqrt(dx ** 2 + dy ** 2)
        indexi = argsort_topk(distancei, num_neighbor)

        flagi = (distancei[indexi[1]] < threshold)
        flagi2 = (df.loc[i, "subclass"] != 'Unlabeled')
        flagi = flagi and flagi2

        indexes.append(indexi.tolist())
        flags.append(flagi)

    print("Select", np.sum(flags), "cells from", df.shape[0], "cells")
    header = ["index_" + str(i) for i in range(num_neighbor)]
    df_meta = pd.DataFrame(data=indexes, columns=header, index=df.index)
    df = df.join(df_meta, how='inner')
    df["flag"] = flags
    df.index = np.array(list(range(df.shape[0])))
    return df

def preprocess_dataset(df_all,genes,use_log_normalize,species,use_nichenetv2=True,
                       visualize=False,num_neighbor=150,distance_threshold=80):
    '''
    Preprocess the dataset for the input of the deep learning model, save the processed data to "./data/processed/"

    :param df_all: a pandas dataframe that contains the columns of
        - all genes (more than one column), as described below, these columns form the expression matrix.
            values in these columns must be int or float
        - "centerx": x coordinates of the cells. int or float
        - "centery": y coordinates of the cells. int or float
        - "section": which tissue slide this cell belongs to, since a dataset may contain more than one slide. string
        - "subclass": the cell type annotation for this cell. string
    :param genes: list, a python list of measured gene names in the dataset
    :param use_log_normalize: bool, whether to perform log-normalization log2(x+1) here for the expression matrix
        Attention: If you have normalized your expression matrix in the dataframe, choose False
    :param species: str, only "human" and "mouse" are supported
    :param use_nichenetv2: bool, whether or not to include the ligands from nichenetv2, if not, only ligand-receptor
        pair from cellchat and neuronchat will be used
    :param visualize: bool, whether or not to visualize the ST dataset with colors indicated by cell types
    :param num_neighbor: int, how many k-nearest neighbor are needed to be calculated
    :param distance_threshold: float or int, if the distance between one cell and its nearest neighbor is
    above this threshold, then we think this cell is moved during the preparation of the tissue slide in
    the wet lab experiment, we would not include this cell in the analysis
    :return: no return, results written at "./data/processed/"
    '''
    # Check if file path exists
    if not os.path.exists(os.path.join(os.getcwd(),"data")):
        os.mkdir(os.path.join(os.getcwd(),"data"))
    if not os.path.exists(os.path.join(os.getcwd(),"data","processed")):
        os.mkdir(os.path.join(os.getcwd(),"data","processed"))

    # save gene information
    torch.save(genes,os.path.join(os.getcwd(),"data","genes.pth"))

    # search for ligands
    search_ligands(genes=genes, species=species, use_nichenetv2=use_nichenetv2, select_liangd_strict=True)

    # Reset index
    df_all.index=list(range(df_all.shape[0]))

    # Do log2(x+1) transform
    print("Doing log2(x+1) tranform")
    if use_log_normalize:
        X=df_all.loc[:,genes].values
        X=np.log(X+1)
        df_all.loc[:,genes]=X
    print("Finish doing log2(x+1) tranform")

    # save cell types dictionary
    cell_types=np.unique(df_all.loc[:,"subclass"].values).tolist()
    torch.save(cell_types,os.path.join(os.getcwd(),"data","processed","cell_types.pth"))

    # calculate cell type expression and cell state expression
    print("Splitting cell type expression and cell state expression")
    type_exp, cell_exp, type_exp_dict = calculate_cell_type_exp(df_all,genes)
    df_all.loc[:, genes] = cell_exp
    print("Finish splitting cell type expression and cell state expression")

    # Preprocess and save each tissue slide
    for sectioni in np.unique(df_all["section"].values):
        print("processing:", sectioni)
        dfi = df_all[df_all["section"] == sectioni]
        if visualize:
            visualize_slide(dfi)
        dfi = get_index(df=dfi,num_neighbor=num_neighbor,threshold=distance_threshold)

        dfi_path=os.path.join(os.path.join(os.getcwd(),"data","processed",sectioni + ".csv"))
        dfi.to_csv(dfi_path)
        type_exp_dict_path=os.path.join(os.path.join(os.getcwd(),"data","processed",sectioni + "_TypeExp.npz"))
        np.savez(type_exp_dict_path, **type_exp_dict)
