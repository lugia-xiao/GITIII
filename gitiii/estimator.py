import pandas as pd

from gitiii.process_dataset import preprocess_dataset
from gitiii.trainer import train_GITIII
from gitiii.predictor import Calculate_influence_tensor

class GITIII_estimator():
    def __init__(self,df_path,genes,use_log_normalize,species,use_nichenetv2=True,
                 visualize_when_preprocessing=False,distance_threshold=80,
                 process_num_neighbors=50,num_neighbors=50,batch_size_train=256,
                 lr=1e-4,epochs=50,node_dim=256,edge_dim=48,att_dim=8,batch_size_val=256,
                 use_cell_type_embedding=True):
        '''
        This object is used for generating the influence tensor

        :param df_path: str, the path of your dataset, which should be a .csv file with columns of:
            - genes (more than one column), as described below, these columns form the expression matrix.
            values in these columns must be int or float
            - "centerx": x coordinates of the cells. int or float
            - "centery": y coordinates of the cells. int or float
            - "section": which tissue slide this cell belongs to, since a dataset may contain more than one slide. string
            - "subclass": the cell type annotation for this cell. string
        :param genes: list of str, a python list of measured gene names in the dataset
        :param use_log_normalize: bool, whether to perform log-normalization log2(x+1) here for the expression matrix
            Attention: If you have normalized your expression matrix in the dataframe, choose False
        :param species: str, what is the species of your dataset, must be one of "human" or "mouse"
        :param use_nichenetv2: bool, whether or not to include the ligands from nichenetv2, if not, only ligand-receptor
            pair from cellchat and neuronchat will be used
        :param visualize_when_preprocessing: bool, whether to visualize the ST dataset when preprocessing
        :param distance_threshold: float or int, if the distance between one cell and its nearest neighbor is
            above this threshold, then we think this cell is moved during the preparation of the tissue slide in
            the wet lab experiment, we would not include this cell in the analysis
        :param process_num_neighbors: int, how many k-nearest neighbor are needed to be calculated in preprocessing
        :param num_neighbors: int, number of neighboring cells used to predict the cell state of the center cell
        :param batch_size_train: int, batch size at the training step
        :param lr: learning rate, float
        :param epochs: int, number of training rounds
        :param node_dim: int, embedding dimension for node in the graph transformer
        :param edge_dim: int, embedding dimension for edge in the graph transformer
        :param att_dim: int, dimension needed to calculate for one-head attention
        :param batch_size_val: int, batch size at the evaluating step
        '''

        self.df_path=df_path
        self.genes=genes
        self.use_log_normalize=use_log_normalize
        self.species=species
        self.use_nichenetv2=use_nichenetv2
        self.visualize_when_preprocessing=visualize_when_preprocessing
        self.distance_threshold = distance_threshold
        self.process_num_neighbors = process_num_neighbors
        self.num_neighbors = num_neighbors
        self.batch_size_train = batch_size_train
        self.lr = lr
        self.epochs = epochs
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.att_dim = att_dim
        self.batch_size_val = batch_size_val
        self.use_cell_type_embedding = use_cell_type_embedding

    def preprocess_dataset(self):
        preprocess_dataset(df_all=pd.read_csv(self.df_path),genes=self.genes,
                           use_log_normalize=self.use_log_normalize,species=self.species,
                           use_nichenetv2=self.use_nichenetv2,
                           visualize=self.visualize_when_preprocessing,
                           num_neighbor=self.process_num_neighbors,distance_threshold=self.distance_threshold)

    def train(self):
        train_GITIII(num_neighbors=self.num_neighbors,batch_size=self.batch_size_train,
                     lr=self.lr,epochs=self.epochs,node_dim=self.node_dim,
                     edge_dim=self.edge_dim,att_dim=self.att_dim,
                     use_cell_type_embedding=self.use_cell_type_embedding)

    def calculate_influence_tensor(self):
        Calculate_influence_tensor(num_neighbors=self.num_neighbors,batch_size=self.batch_size_val,
                                   node_dim=self.node_dim,edge_dim=self.edge_dim,att_dim=self.att_dim)
