import os
import torch
import numpy as np
import pandas as pd

import scanpy as sc
from anndata import AnnData
import anndata as ad
import matplotlib.pyplot as plt


def extract_genes_and_pvals_by_group(adata, group_index, cutoff=0.05, up=True):
    '''
    Select significant differential expressed genes within an adata object
    :param adata:
    :param group_index: leiden index
    :param cutoff: adjusted p-value cutoff
    :param up: only to select up-regulated genes if True otherwise only select down-regulated genes
    :return:
    '''
    # Extracting gene names and p-values from the adata object
    gene_names = adata.uns['rank_genes_groups']['names']
    p_values = adata.uns['rank_genes_groups']['pvals']
    logfoldchanges = adata.uns['rank_genes_groups']['logfoldchanges']
    p_adj = adata.uns['rank_genes_groups']['pvals_adj']

    # Lists to hold filtered gene names and their corresponding p-values
    filtered_genes = []
    filtered_pvals = []

    # Iterate through each group in the gene names and p-values
    for gene_group, pval_group, fold_group in zip(gene_names, p_adj, logfoldchanges):
        gene = gene_group[group_index]
        pval = pval_group[group_index]
        foldchange = fold_group[group_index]

        # Check if p-value is below the cutoff and add to the lists if it is
        # print(pval,foldchange)
        if pval < cutoff and ((foldchange > 0) == up):
            filtered_genes.append(gene)
            filtered_pvals.append(pval)

class Subtyping_anlayzer():
    def __init__(self,sample,normalize_to_1=True,use_abs=False,noise_threshold=1e-5):
        '''

        :param sample: which tissue slide to analyze
        :param normalize_to_1: whether normalize the aggregated influence tensor so that their abs value sum
        up to one on the second dimension
        :param use_abs: whehter to use the absolute value for the aggregated influence tensor for downstream
        analysis
        :param noise_threshold: For values in the influence tensor, if its abs value is less than this threshold,
        we would treat it as noise, ignore it by setting it to 0
        '''
        # load data
        print("Start loading data")
        result_dir=os.path.join(os.getcwd(),"influence_tensor")
        if result_dir[-1]!="/":
            result_dir=result_dir+"/"
        data_dir=os.path.join(os.getcwd(),"data","processed")
        if data_dir[-1]!="/":
            data_dir=data_dir+"/"
        cell_types=torch.load(os.path.join(os.getcwd(),"data","processed","cell_types.pth"))
        self.cell_types=cell_types
        genes = torch.load(os.path.join(os.getcwd(),"data","genes.pth"))
        self.genes=genes
        type_exp_dict = np.load(os.path.join(data_dir, sample + "_TypeExp.npz"), allow_pickle=True)
        results = torch.load(result_dir + "edges_" + sample + ".pth", map_location=torch.device('cpu'))
        print("Finish loading data")

        feature_names = []
        for i in range(len(cell_types)):
            for j in range(len(genes)):
                feature_names.append(cell_types[i] + "--" + genes[j])

        position_x = results["position_x"][:, 0]
        position_y = results["position_y"][:, 0]
        cell_type_name = np.array(results["cell_type_name"])
        cell_type_target = cell_type_name[:, 0]

        type_exps = torch.stack(
            [torch.Tensor(type_exp_dict[cell_type_targeti]) for cell_type_targeti in cell_type_target], dim=0)
        results["y"] = results["y"] + type_exps

        attention_scores = results["attention_score"]
        cell_type_names = np.array(results["cell_type_name"])

        proportion = torch.abs(attention_scores)
        proportion = proportion / torch.sum(proportion, dim=1, keepdim=True)
        attention_scores[proportion < noise_threshold] = 0

        # Initialize a tensor to hold aggregated interaction strengths
        B, _, C = attention_scores.shape
        t = len(cell_types)
        aggregated_interactions = torch.zeros((B, t, C))

        # Map cell type names to indices
        cell_type_to_index = {ct: idx for idx, ct in enumerate(cell_types)}

        # Aggregate interaction strengths by cell type
        print("Start aggregating")
        for b in range(B):
            if b%500==0:
                print(b,"/",B)
            for n in range(1, 50):  # Skip the first element, which is the target cell type
                neighbor_type = cell_type_names[b][n]
                if neighbor_type in cell_type_to_index:
                    idx = cell_type_to_index[neighbor_type]
                    aggregated_interactions[b, idx] += attention_scores[b, n - 1]
        print("Finish aggregating")

        if normalize_to_1:
            aggregated_interactions1 = aggregated_interactions / torch.sum(torch.abs(aggregated_interactions), dim=1,
                                                                           keepdim=True)
            aggregated_interactions = torch.where(
                torch.sum(torch.abs(aggregated_interactions), dim=1, keepdim=True) == 0,
                torch.zeros_like(aggregated_interactions), aggregated_interactions1)

        if use_abs:
            aggregated_interactions=torch.abs(aggregated_interactions)

        adata = AnnData(aggregated_interactions.reshape(B, -1).numpy())
        adata.obs['cell_type'] = cell_type_target
        adata.obs['position_x'] = position_x
        adata.obs['position_y'] = position_y
        adata.var_names = feature_names
        adata.obsm["y"] = results["y"].numpy()
        self.adata=adata

        with plt.rc_context({'figure.figsize': (6, 6)}):
            sc.pl.scatter(
                adata,
                x='position_x',  # 'position_x',
                y='position_y',  # 'position_y',
                color="cell_type"
            )

        self.adata_type=None
        self.adata_type_y=None

    def subtyping(self,COI,resolution=0.2):
        '''
        Do cell subtyping analysis, plot UMAP for subgroup and plot their spatial distribution
        :param COI: Cell Of Interest
        :param resolution: resolution in leiden clustering
        :return:
        '''
        self.adata_type = self.adata[self.adata.obs["cell_type"] == COI]

        sc.tl.pca(self.adata_type, n_comps=50)
        sc.pp.neighbors(self.adata_type)  # Compute the neighborhood graph

        # Clustering
        sc.tl.leiden(self.adata_type, resolution=resolution)  # or sc.tl.louvain(adata)

        # Plot UMAP
        sc.tl.umap(self.adata_type)  # Compute UMAP
        sc.pl.umap(self.adata_type, color='leiden')

        sc.pl.scatter(
            self.adata_type,
            x='position_x',  # 'position_x',
            y='position_y',  # 'position_y',
            color="leiden",
            title=f"Spatial cluster of {COI}"
        )

    def subtyping_filter_groups(self,group_to_remain):
        '''

        :param group_to_remain: list of str, for example, if you want to just analyze or compare the
            0-the group and 1-th group, as shown on the UMAP in subtyping analysis, you can make
            group_to_remain=["1","0"], be aware that the items in the list are not int, they are str
        :return:
        '''
        if self.adata_type is None:
            raise ValueError("Please do subtyping analysis first")
        self.adata_type=self.adata_type[self.adata_type.obs["leiden"].isin(group_to_remain)].copy()

    def subtyping_DE(self,method='wilcoxon',n_gene_show=5):
        '''
        Do and visualize the differential expression analysis
        :param method: statistical method to make comparison using scanpy, default to 'wilcoxon' (rank-sum test),
            other available methods are: 'logreg', 't-test', 'wilcoxon', 't-test_overestim_var'
        :param n_gene_show: how many DE gene to plot for one subgroup
        :return:
        '''
        if self.adata_type is None:
            raise ValueError("Please do subtyping analysis first before perform DEG analysis on subtypes that do not exist")
        self.adata_type_y = ad.AnnData(X=np.abs(self.adata_type.obsm["y"]), obs=self.adata_type.obs)
        # adata_y=adata_y[adata_y.obs['leiden'].isin(["0","1"])]#,"2","2","3"
        self.adata_type_y.var_names = self.genes
        sc.tl.rank_genes_groups(self.adata_type_y, 'leiden',method=method)
        sc.pl.rank_genes_groups_heatmap(self.adata_type_y, n_genes=n_gene_show, show_gene_labels=True,
                                        standard_scale='var',cmap='viridis')

    def subtyping_get_aggregated_influence(self):
        '''
        Analyze, proportionally, how each cell of the COI influenced by other cell types
        :return:
        '''
        if self.adata_type is None:
            raise ValueError("Please do subtyping analysis first")

        x = np.zeros((self.adata_type.shape[0], len(self.cell_types)))
        for i in range(len(self.cell_types)):
            offset = i * len(self.genes)
            x[:, i] = np.mean(np.abs(self.adata_type.X[:, offset:offset + len(self.genes)]), axis=1)

        self.adata_type_aggregated = ad.AnnData(X=x, obs=self.adata_type.obs)
        self.adata_type_aggregated.var_names = self.cell_types

        sc.pl.heatmap(self.adata_type_aggregated, var_names=self.adata_type_aggregated.var_names, groupby='leiden', cmap='bwr')
        sc.tl.rank_genes_groups(self.adata_type_aggregated, 'leiden', method='wilcoxon')
        sc.pl.rank_genes_groups_heatmap(self.adata_type_aggregated, n_genes=10, show_gene_labels=True, cmap='bwr')

    def subtyping_get_aggregated_influence_target_gene(self,target_gene):
        '''
        Analyze, proportionally, how each cell of the COI's one target gene influenced by other cell types
        :return:
        '''
        if self.adata_type is None:
            raise ValueError("Please do subtyping analysis first")

        x = np.zeros((self.adata_type.shape[0], len(self.cell_types)))

        offset = self.genes.index(target_gene)
        for i in range(len(self.cell_types)):
            x[:, i] = x[:, i] + self.adata_type.X[:, i * len(self.genes) + offset]

        self.adata_type_aggregated_target_gene = ad.AnnData(X=x, obs=self.adata_type.obs)
        self.adata_type_aggregated_target_gene.var_names = self.cell_types

        sc.pl.heatmap(self.adata_type_aggregated_target_gene, var_names=self.adata_type_aggregated_target_gene.var_names, groupby='leiden',
                      cmap='bwr')
        sc.tl.rank_genes_groups(self.adata_type_aggregated_target_gene, 'leiden', method='wilcoxon')
        sc.pl.rank_genes_groups_heatmap(self.adata_type_aggregated_target_gene, n_genes=10, show_gene_labels=True, cmap='bwr')



