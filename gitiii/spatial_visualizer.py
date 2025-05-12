import os
import torch
import numpy as np
import pandas as pd

import scanpy as sc
from anndata import AnnData
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

def plot_binned_average_with_loess(x, y, bins=300, frac=0.003):
    # Bin data and compute average y for each bin
    # bin_means, bin_edges, binnumber = binned_statistic(x, y, statistic='mean', bins=bins)
    # Determine the range of x and set bin edges
    min_x = np.min(x)
    max_x = np.max(x)
    bin_edges = np.linspace(min_x, max_x, bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_width / 2

    # Compute mean y-value for each bin
    bin_means, _, binnumber = binned_statistic(x, y, statistic='mean', bins=bin_edges)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_width / 2

    # Calculate the smooth curve using Lowess
    lowess = sm.nonparametric.lowess
    smoothed_data = lowess(bin_means, bin_centers, frac=frac)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.scatter(bin_centers, bin_means, alpha=0.5, label='Binned Average')
    plt.plot(smoothed_data[:, 0], smoothed_data[:, 1], 'r-', label='LOESS Smooth')
    plt.title("Binned Average Plot with LOESS Smooth")
    plt.xlabel("x")
    plt.ylabel("y")
    if np.min(y)>=0:
        plt.ylim(bottom=0)  # Set the y-axis to start from 0
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_CCI_UMAP(adata, color, select_type_number):
    unique = np.unique(adata.obs['cell_type'], return_counts=True)
    args = np.argsort(-unique[1])[:select_type_number]
    select_type = unique[0][args]
    print("selecting edge types:", select_type)

    flag = [adata.obs['cell_type'][i] in select_type for i in range(adata.shape[0])]
    adata_filtered = adata[flag, :]

    sc.pp.scale(adata_filtered)
    sc.tl.pca(adata_filtered)
    sc.pp.neighbors(adata_filtered)  # Compute the neighborhood graph
    sc.tl.umap(adata_filtered)  # Compute UMAP
    # Plot UMAP
    sc.pl.umap(adata_filtered, color=color)
    return adata_filtered

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def calculate_pcc(adata1, adata2, gene_name):
    gene_expression1 = adata1.obs_vector(gene_name)
    gene_expression2 = adata2.obs_vector(gene_name)
    correlation_matrix = np.corrcoef(gene_expression1, gene_expression2)
    return correlation_matrix[0, 1]  # PCC between the two gene expressions

def spatial_visualize_gene_(adata, gene_name, is_pred, vmin, vmax, pcc1, pcc2,
                            point_size=1, normalize_expression=False, colorbar_size=0.8):
    if gene_name not in adata.var_names:
        raise ValueError(f"Gene '{gene_name}' not found in adata.var_names")

    title = "Predicted expression of " + gene_name if is_pred else "Expression of " + gene_name
    if is_pred:
        title += f"\nOverall PCC: {pcc1:.2f}; Cell state PCC: {pcc2:.2f}"

    x = adata.obs['to_position_x']
    y = adata.obs['to_position_y']
    gene_expression = adata.obs_vector(gene_name)
    if normalize_expression:
        gene_expression = normalize_data(gene_expression)

    x_range = max(x) - min(x)
    y_range = max(y) - min(y)
    aspect_ratio = x_range / y_range
    fig_width = 8
    fig_height = fig_width / aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    scatter = ax.scatter(x, y, c=gene_expression, cmap='jet', vmin=vmin, vmax=vmax, s=point_size)
    cbar = plt.colorbar(scatter, ax=ax, shrink=colorbar_size)
    ax.set_title(title)
    ax.set_aspect('equal')

    plt.show()

def spatial_visualize_gene(adata_pred, adata_y, gene_name, pcc1, pcc2,
                           normalize_expression=False, colorbar_size=0.8):
    ground_truth_values = adata_y.obs_vector(gene_name)
    if normalize_expression:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = ground_truth_values.min(), ground_truth_values.max()
    spatial_visualize_gene_(adata_y, gene_name, False, vmin, vmax, pcc1, pcc2,
                            normalize_expression=normalize_expression, colorbar_size=colorbar_size)
    spatial_visualize_gene_(adata_pred, gene_name, True, vmin, vmax, pcc1, pcc2,
                            normalize_expression=normalize_expression, colorbar_size=colorbar_size)

def calculate_pcc_for_all_genes(adata1, adata2):
    pcc_dict = {}
    for gene in adata1.var_names:
        pcc = calculate_pcc(adata1, adata2, gene)
        pcc_dict[gene] = pcc
    return pcc_dict

class Spatial_visualizer():
    def __init__(self,sample):
        '''
        Distance_scaler: by (rank) or (distance) for (proportional influences) or (influence's abs values)
        UMAP of CCI pairs: the pair of one sender cell influence one receiver cell is called a CCI pair,
            and the estimated influence are treated as the features of the CCI pair. This function aim to
            visualize how different cell type pair influence each other.
        Prediction visualization: visualize the predicted cell expression v.s. predicted expression,
            can be state expression or raw expression
        Information flow: select the top 5 CCI pair for each receiver cell and select top 5% strongest CCI out
            of all the top 5 CCI pairs and visualize them

        :param sample: which tissue section (sample) to demonstrate
        '''
        # load data
        print("Start loading data")
        self.sample=sample
        result_dir = os.path.join(os.getcwd(), "influence_tensor")
        if result_dir[-1] != "/":
            result_dir = result_dir + "/"
        data_dir = os.path.join(os.getcwd(), "data", "processed")
        if data_dir[-1] != "/":
            data_dir = data_dir + "/"
        self.data_dir=data_dir
        cell_types = torch.load(os.path.join(os.getcwd(), "data", "processed", "cell_types.pth"),weights_only=False)
        self.cell_types = cell_types
        genes = torch.load(os.path.join(os.getcwd(), "data", "genes.pth"),weights_only=False)
        self.genes = genes
        type_exp_dict = np.load(os.path.join(data_dir, sample + "_TypeExp.npz"), allow_pickle=True)
        self.results = torch.load(result_dir + "edges_" + sample + ".pth", map_location=torch.device('cpu'),weights_only=False)

        # convert to absolute expression
        self.results["y_state"] = self.results["y"]
        cell_type_names = np.array(self.results["cell_type_name"])
        cell_type_target = [cell_type_names[i][0] for i in range(len(cell_type_names))]
        type_exp_dict = np.load(data_dir + sample + "_TypeExp.npz", allow_pickle=True)
        type_exps = torch.Tensor(np.stack([type_exp_dict[cell_typei] for cell_typei in cell_type_target], axis=0))
        self.results["y"] = self.results["y"] + type_exps

        # load position df
        self.df_position = pd.read_csv(data_dir + sample + ".csv")
        self.df_position['cell_type_plot'] = self.df_position["subclass"].apply(
            lambda x: x.split(" ")[0] if x.startswith('L') and x[1] < '9' and x[1] > '0' else "Not_excitatory_neuron")

        # load cell state expression predictions
        self.adata_y_pred,self.adata_y=self.read_prediction_adata(state=False)
        self.adata_y_state_pred, self.adata_y_state = self.read_prediction_adata(state=True)

        # calculate the pccs for each gene
        self.pccs_dict = calculate_pcc_for_all_genes(self.adata_y, self.adata_y_pred)
        self.state_pccs_dict=calculate_pcc_for_all_genes(self.adata_y_state,self.adata_y_state_pred)

        print("There are",self.results["y"].shape[0],"cells in this sample")
        print("Finish loading data")


    def plot_distance_scaler(self,rank_or_distance="distance",proportion_or_abs="abs",target_gene=None,
                             bins=300, frac=0.003):
        '''
        Estimate the distance scaler and visualize it, x-axis is distance or the rank of nearest neighbor,
        the y-axis can be proportional influence or the abs value of influence
        :param rank_or_distance: x axis is distance or the order (rank) of nearest neighbor
        :param proportion_or_abs: use the proportional influence (for each cell's each gene, the influence are all
            positive values that sum up to 1) or the abs value of interaction
        :param target_gene: if None, calculate the distance scaler averaged over all target genes
        :param bins, frac: parameters used of calculating and plot losses.
        :return:
        '''
        assert rank_or_distance in ["rank","distance"]
        assert proportion_or_abs in ["proportion","abs"]

        attention_scores = self.results["attention_score"]
        print("loss:", torch.mean(torch.square(self.results["y_pred"] - self.results["y_state"])))
        print("random:", torch.mean(torch.square(self.results["y_state"])))
        print("first 5:", torch.mean(torch.square(torch.sum(attention_scores[:, :5, :], dim=1)/8 - self.results["y_state"])))
        print("first 10:", torch.mean(torch.square(torch.sum(attention_scores[:, :10, :], dim=1)/8 - self.results["y_state"])))
        print("first 20:", torch.mean(torch.square(torch.sum(attention_scores[:, :20, :], dim=1)/8 - self.results["y_state"])))

        proportion = torch.abs(attention_scores)  # torch.log(torch.log(-torch.log(torch.abs(attention_scores))))
        if proportion_or_abs=="proportion":
            proportion=proportion/torch.sum(proportion,dim=1,keepdim=True)

        if target_gene is None:
            proportion = torch.mean(proportion, dim=-1)
        else:
            proportion = proportion[:, :, self.genes.index(target_gene)]

        position_xs = self.results["position_x"][:, 1:]
        position_ys = self.results["position_y"][:, 1:]
        position_x0 = self.results["position_x"][:, 0:1]
        position_y0 = self.results["position_y"][:, 0:1]
        distances = torch.sqrt(torch.square(position_xs - position_x0) + torch.square(position_ys - position_y0))

        proportion = proportion.flatten().numpy()
        distance = distances.flatten().numpy()

        if rank_or_distance=="rank":
            n = attention_scores.shape[1]
            tmp = np.arange(n) + 1
            distance = np.tile(tmp, (attention_scores.shape[0], 1)).flatten()

        plot_binned_average_with_loess(distance, proportion,bins=bins,frac=frac)


    def read_topk(self,select_topk=5,target_gene=None):
        '''

        :param select_topk: For each receiver cell, how many strongest CCI pair should be selected
        :return:
        '''
        position_x = self.results["position_x"]
        position_y = self.results["position_y"]
        cell_type_name = self.results["cell_type_name"]
        B, N = position_x.shape
        cell_type_target = [cell_type_name[i][0] for i in range(len(cell_type_name))]

        if target_gene is None:
            tmp = torch.abs(self.results["attention_score"]) / torch.sum(torch.abs(self.results["attention_score"]),dim=-2,keepdim=True)
        else:
            tmp=torch.abs(self.results["attention_score"])[:,:,self.genes.index(target_gene)].unsqueeze(dim=-1)

        indices = torch.topk(torch.sum(tmp, dim=-1), k=select_topk, dim=-1)[1]
        indices_tmp = torch.arange(0, indices.shape[0], 1).unsqueeze(dim=-1).repeat(1, select_topk)
        indices = torch.stack([indices_tmp.reshape(-1), indices.reshape(-1)], dim=0)

        edges = self.results["attention_score"]
        edges = edges[indices[0], indices[1], :].reshape(-1, edges.shape[-1]).numpy()

        to_position_x = position_x[:, 0:1].repeat(1, select_topk).reshape(-1).numpy()
        to_position_y = position_y[:, 0:1].repeat(1, select_topk).reshape(-1).numpy()
        from_position_x = position_x[:, 1:][indices[0], indices[1]].reshape(-1).numpy()
        from_position_y = position_y[:, 1:][indices[0], indices[1]].reshape(-1).numpy()

        dx = position_x - position_x[:, 0:1]
        dy = position_y - position_y[:, 0:1]
        distances = torch.sqrt(torch.square(dx) + torch.square(dy))
        distances = distances[indices[0], indices[1]].reshape(-1).numpy()

        indices_np = indices.numpy()
        cell_types = np.array(cell_type_name)[:, 1:][indices_np[0], indices_np[1]].reshape(B, select_topk)
        print("number of cell types in this sample:", np.unique(np.array(cell_type_name)[:, 0], return_counts=True))
        cell_types_pair = []
        cell_types_target = []
        flags = []
        for i in range(cell_types.shape[0]):
            for j in range(cell_types.shape[1]):
                cell_types_pair.append(cell_types[i][j] + "->" + cell_type_target[i])
                flags.append(cell_types[i][j] != cell_type_target[i])
                cell_types_target.append(cell_type_target[i])

        adata = AnnData(edges)
        adata.obs['cell_type'] = cell_types_pair
        adata.obs['cell_type_target'] = cell_types_target
        adata.obs['from_position_x'] = from_position_x
        adata.obs['from_position_y'] = from_position_y
        adata.obs['to_position_x'] = to_position_x
        adata.obs['to_position_y'] = to_position_y
        adata.obs['distance'] = distances
        # modify here
        adata.obsm["y"] = self.results["y"].repeat(1, select_topk).reshape(self.results["y"].shape[0] * select_topk,
                                                                           self.results["y"].shape[1]).numpy()
        return adata


    def visualize_CCI_function(self,select_topk=5,num_type_pair=10):
        '''

        :param select_topk: For each receiver cell, how many strongest CCI pair should be selected for visualize
            one point on the UMAP is one CCI pair
        :param num_type_pair: How many most frequent CCI type pair combination to show, since we can not demonstrate
            all cell_type_number*cell_type_number cell type combinations, there are too many colors
        :return:
        '''
        adata=self.read_topk(select_topk=select_topk)
        plot_CCI_UMAP(adata=adata,color="cell_type",select_type_number=num_type_pair)
        plot_CCI_UMAP(adata=adata, color="distance", select_type_number=num_type_pair)


    def visualize_information_flow(self,target_gene,select_topk=5,use_neuron_layer=True,cutoff=0.05):
        '''

        :param target_gene:
        :param select_topk:
        :param use_neuron_layer: In the plot, whether to generalize the cell types to only excitatory neurons at
            different layers and Not_excitatory_neuron
        :param cutoff: In percentage, how many top CCI pairs out of select_topk*number_of_cell pairs to visualize.
        :return:
        '''
        title="Information flow of "+target_gene
        adata = self.read_topk(select_topk=select_topk,target_gene=target_gene)
        idx=self.genes.index(target_gene)
        from_x = adata.obs['from_position_x']
        from_y = adata.obs['from_position_y']
        to_x = adata.obs['to_position_x']
        to_y = adata.obs['to_position_y']
        values = adata.X[:, idx]

        retain_number = int(values.shape[0] * cutoff)
        args = np.argsort(-values)[:retain_number]
        flag = [i for i in range(values.shape[0]) if i in args.tolist()]

        from_x = from_x[flag]
        from_y = from_y[flag]
        to_x = to_x[flag]
        to_y = to_y[flag]
        values = values[flag]

        # Calculate differences for arrow directions
        dx = to_x - from_x
        dy = to_y - from_y

        # Normalize the values for coloring
        norm = plt.Normalize(values.min(), values.max())

        # Create a colormap
        cmap = plt.cm.jet

        # Create a figure
        width = np.max(self.df_position.loc[:, "centerx"]) - np.min(self.df_position.loc[:, "centerx"])
        height = np.max(self.df_position.loc[:, "centery"]) - np.min(self.df_position.loc[:, "centery"])
        fig, ax = plt.subplots(figsize=(10 * (width / height + 0.25), 10))

        # Plotting the arrows using quiver
        hue='cell_type_plot' if use_neuron_layer else 'subclass'
        scatter = sns.scatterplot(data=self.df_position, x="centerx", y="centery", hue=hue,
                                  legend='full', ax=ax, size=0.1)

        quiver = ax.quiver(from_x, from_y, dx, dy, values, angles='xy', scale_units='xy', scale=1, cmap=cmap, norm=norm)

        # Set the x and y axis limits
        min_x_value = min(np.min(adata.obs['from_position_x']), np.min(adata.obs['to_position_x']))
        min_y_value = min(np.min(adata.obs['from_position_y']), np.min(adata.obs['to_position_y']))
        max_x_value = max(np.max(adata.obs['from_position_x']), np.max(adata.obs['to_position_x']))
        max_y_value = max(np.max(adata.obs['from_position_y']), np.max(adata.obs['to_position_y']))
        ax.set_xlim([min_x_value, max_x_value])  # Replace with your desired limits
        ax.set_ylim([min_y_value, max_y_value])  # Replace with your desired limits

        # Show color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax)

        # Show the plot
        plt.title(title)
        plt.show()


    def read_prediction_adata(self,state=False):
        '''

        :param state: whether to return
        :return:
        '''
        position_x = self.results["position_x"]
        position_y = self.results["position_y"]
        cell_type_name = self.results["cell_type_name"]

        B, N = position_x.shape

        cell_type_target = [cell_type_name[i][0] for i in range(len(cell_type_name))]

        type_exp_dict = np.load(self.data_dir + self.sample + "_TypeExp.npz", allow_pickle=True)
        type_exps = np.stack([type_exp_dict[cell_typei] for cell_typei in cell_type_target], axis=0)

        to_position_x = position_x[:, 0].numpy()
        to_position_y = position_y[:, 0].numpy()

        if not state:
            adata = AnnData(self.results["y_pred"].numpy())  # +type_exps)#(np.log(results["y_pred"].numpy()))
        else:
            adata = AnnData(self.results["y_pred"].numpy() + type_exps)
        adata.obs['cell_type'] = cell_type_target
        adata.obs['to_position_x'] = to_position_x
        adata.obs['to_position_y'] = to_position_y
        adata.var_names = self.genes

        if not state:
            adata1 = AnnData(self.results["y_state"].numpy())  # +type_exps)#(np.log(results["y"].numpy()))
        else:
            adata1 = AnnData(self.results["y_state"].numpy() + type_exps)  # (np.log(results["y"].numpy()))
        adata1.obs['cell_type'] = cell_type_target
        adata1.obs['to_position_x'] = to_position_x
        adata1.obs['to_position_y'] = to_position_y
        adata1.var_names = self.genes
        return adata, adata1

    def visualize_prediction(self,target_gene,plot_state=False):
        if plot_state:
            print("Now plotting predicted cell state expression v.s. real cell state expression")
            spatial_visualize_gene(adata_pred=self.adata_y_state_pred, adata_y=self.adata_y_state,
                                   gene_name=target_gene, pcc1=self.state_pccs_dict[target_gene],
                                   pcc2=self.state_pccs_dict[target_gene],
                                   normalize_expression=False, colorbar_size=0.8)
        else:
            print("Now plotting predicted expression v.s. real expression")
            spatial_visualize_gene(adata_pred=self.adata_y_pred, adata_y=self.adata_y,
                                   gene_name=target_gene, pcc1=self.state_pccs_dict[target_gene],
                                   pcc2=self.state_pccs_dict[target_gene],
                                   normalize_expression=False, colorbar_size=0.8)

