# Investigation of pair-wise single cell interactions through statistically interpreting spatial cell state correlations learned by self-supervised graph inductive bias transformer

![](./Figure1_overview.png)

## Description

GITIII is an interpretable and lightweight python packaged used for analyzing cell-cell interaction (CCI) in imaging-based spatial transcriptomics with minium demand on the ligand-receptor database since many imaging-based ST datasets contain low or even no ligand-receptor pairs in the measured genes. Generally, it takes 1 hour to run on a dataset with 366,272 cells and 140 genes, the running time increase linearly with the number of cells and number of genes.

The main functionalities of GITIII includes:
1. Estimating the influence tensor that describe how each cell is influenced by its top k (default 50) nearest neighbors
2. Visualizing the relationship between the strength of interaction with the distance between two cells
3. Visualizing the funcitons of CCI pairs using UMAP: the pair of one sender cell influencing one receiver cell is called a CCI pair, and the estimated influence from the sender cell to the receiver cell (how much the gene expression in the receiver cell would change because of the existing of the sender cell) are treated as the features of the CCI pair. This function aim to visualize how different CCI pairs belonging to different cell type combinations differ from each other in terms of their functions
4. Prediction visualization: visualize the predicted cell expression v.s. predicted expression, can be state expression or raw expression
5. Information flow: Where are the strongest CCI pairs in the slide (tissue section), with arrows in the plot indicating the interaction from one sender cell to one receiver cell
6. Cell subtyping analysis: Construct interpretable CCI-informed features for each cell, (how each cell type influence each measured genes in this cell), use these features to do Leiden clustering and UMAP visualization. Then differential expressed gene (DEG) analysis can be performed on these subtypes (subgroups), and we can also visualize how this cell ('s one target gene) is influenced by other cell types via heatmap.
7. Network analysis: Using partial linear regression to make statistical test of whether one cell type significantly influence one gene in the receiver cell type, forming a significant CCI network targeting each gene.

## Installation

It is recommanded to install it in a environment with pytorch installed.

```bash
git clone https://github.com/lugia-xiao/GITIII.git
cd GITIII
pip install .
```

Or

```bash
pip install gitiii
```



## Quick start

### Data you need:

:param `df_path`: str, the path of your dataset, which should be a .csv file with columns of:
- genes (more than one column), as described below, these columns form the (normalized) expression matrix.
            values in these columns must be int or float
- "centerx": x coordinates of the cells. int or float
- "centery": y coordinates of the cells. int or float
- "section": which tissue slide this cell belongs to, since a dataset may contain more than one slide. string
- "subclass": the cell type annotation for this cell. string

:param `genes`: list of str, a python list of measured gene names in the dataset

:param `species`: str, what is the species of your dataset, must be one of "human" or "mouse"

```python
# Import necessary python packages
import gitiii

estimator=gitiii.estimator.GITIII_estimator(df_path=df_path,genes=genes,use_log_normalize=True,species="human",use_nichenetv2=True,visualize_when_preprocessing=False,distance_threshold=80,process_num_neighbors=50,num_neighbors=50,batch_size_train=256,lr=1e-4,epochs=50,node_dim=256,edge_dim=48,att_dim=8,batch_size_val=256)
```

:param `use_log_normalize`: bool, whether to perform log-normalization log2(x+1) here for the expression matrix

 - Attention: If you would like to use your own way of data normalization or have already normalized your expression matrix in the dataframe, choose False

### Preprocess dataset

Split the cell type expression and cell state expression.

```python
estimator.preprocess_dataset()
```

### Train the deep learning model

```python
estimator.train()
```

### Calculate the influence tensor

```python
estimator.calculate_influence_tensor()
```

### Visualize the spatial patterns

```python
sample='H20.33.001.CX28.MTG.02.007.1.02.03'

spatial_visualizer=gitiii.spatial_visualizer.Spatial_visualizer(sample=sample)

spatial_visualizer.plot_distance_scaler(rank_or_distance="distance",proportion_or_abs="abs",target_gene=None,bins=300, frac=0.003)
```

### Cell subtyping analysis

```python
subtyping_analyzer=gitiii.subtyping_analyzer.Subtyping_anlayzer(sample=sample,normalize_to_1=True,use_abs=True,noise_threshold=2e-2)
# Take L2/3 IT as an example for CCI informed subtyping analysis
COI="L2/3 IT" # Cell Of Interest
subtyping_analyzer.subtyping(COI=COI,resolution=0.1)
```

**For more detailed tutorial, please refer to tutorial_mouse_brain_MERFISH.nbconvert.ipynb and tutorial_human_AD.nbconvert.ipynb**









