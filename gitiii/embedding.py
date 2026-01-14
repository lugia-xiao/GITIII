import torch
import torch.nn as nn
import torch.nn.functional as F
import gitiii.preprocess as preprocess

class FFN(nn.Module):
    def __init__(self,dim,in_dim=None,out_dim=None,bias=True):
        super().__init__()
        if in_dim is None:
            in_dim=dim
        if out_dim is None:
            out_dim=dim

        self.linear1=nn.Linear(in_dim,dim*4,bias=bias)
        self.linear2=nn.Linear(dim*4,out_dim,bias=bias)
        self.act=nn.GELU()

    def forward(self,x):
        return self.linear2(self.act(self.linear1(x)))

def group_strings_by_numbers(strings, numbers):
    """
    Group a list of strings based on unique numbers and return as two ordered lists.

    :param strings: List of strings to be grouped.
    :param numbers: List of numbers indicating the grouping.
    :return: Two lists, one with unique numbers and the other with corresponding grouped strings.
    """
    numbers=numbers.copy()[:len(strings)]
    unique_numbers = sorted(set(numbers))
    grouped_strings = [[] for _ in unique_numbers]

    for string, number in zip(strings, numbers):
        index = unique_numbers.index(number)
        grouped_strings[index].append(string)

    return unique_numbers, grouped_strings

class Embedding(nn.Module):
    def __init__(self,genes,ligands_info,node_dim,edge_dim,use_cell_type_embedding=True):
        super().__init__()
        self.use_cell_type_embedding=use_cell_type_embedding
        self.ligands_info=ligands_info

        # for scaling the data
        #self.exp_scaler=nn.Parameter(torch.ones((len(genes))))

        # for node
        self.cell_encoder1 = FFN(node_dim, in_dim=len(genes))
        self.cell_encoder2 = nn.Embedding(29,node_dim)

        # for distance
        self.distance_scaler=nn.Sequential(FFN(edge_dim,out_dim=len(ligands_info[0]),in_dim=5),nn.Sigmoid())
        self.distance_encoder=FFN(edge_dim,out_dim=len(ligands_info[0]),in_dim=5)
        self.distance_embedding = nn.Sequential(FFN(edge_dim, in_dim=5), nn.Sigmoid())

        # for edge
        self.ligands_encoder=nn.Linear(len(ligands_info[0]),edge_dim)
        self.edge_encoder=nn.Linear(len(ligands_info[0])+edge_dim,edge_dim,bias=False)
        self.scaler=edge_dim**0.5

        # calculate ligand scores
        self.ligands_index=[]
        for i in range(len(self.ligands_info[0])):
            ligandi = self.ligands_info[0][i]
            ligand_groupi = self.ligands_info[1][i]
            uniquei, groupi = group_strings_by_numbers(ligandi, ligand_groupi)
            groupi=[torch.LongTensor([genes.index(groupi[j][k]) for k in range(len(groupi[j]))]) for j in range(len(groupi))]
            self.ligands_index.append(groupi)

    def forward(self,x):
        '''
        :param x:
        x["x"]--b,n,in_node
        x["distance_matrix"]--b,5,n,n
        x["cell_types"]--long,b,n
        x["type_exp"]--b,n,in_node
        :return:
        node_features:b,n,node_dim
        edge1:b,n,n,edge_dim
        edge2:b,n,n,edge_dim+5
        rw: b,n,n,num_hops+5
        '''
        x=preprocess.process(x)

        # node features
        #scaler=torch.square(self.exp_scaler)
        exp=x["type_exp"]+x["x"]
        exp=torch.where(x["cell_types"].unsqueeze(dim=-1)==x["cell_types"][:,0:1].unsqueeze(dim=-1),x["type_exp"],exp)
        exp=exp#*scaler
        node_features1=self.cell_encoder1(exp)
        node_features2 = self.cell_encoder1(x["type_exp"])#*scaler)
        node_features=torch.where(x["cell_types"].unsqueeze(dim=-1)==x["cell_types"][:,0:1].unsqueeze(dim=-1),node_features2,node_features1)

        if self.use_cell_type_embedding:
            node_features=node_features+self.cell_encoder2(x["cell_types"])

        # calculate ligand scores
        B,N,C=node_features.shape
        data_ligand=[]
        for i in range(len(self.ligands_info[0])):
            data_ligandi=torch.ones((B,N),device=node_features.device)
            for j in range(len(self.ligands_index[i])):
                data_ligandi = data_ligandi * torch.mean(exp[:,:,self.ligands_index[i][j]],dim=-1)
            data_ligandi=torch.pow(data_ligandi,exponent=1 / len(self.ligands_index[i]))
            data_ligand.append(data_ligandi)
        data_ligand=torch.stack(data_ligand,dim=-1)

        # distance scaler
        distance_scaler = self.distance_scaler(x["distance_matrix"])

        # ligand score scaled by distance
        ligand_score=distance_scaler*data_ligand
        edge_ligand=self.ligands_encoder(ligand_score)

        # edge features
        edge_distance=self.distance_encoder(x["distance_matrix"])
        edges = self.edge_encoder(torch.concat([edge_distance,edge_ligand],dim=-1))*self.scaler

        # distance embedding
        embedding = self.distance_embedding(x["distance_matrix"])
        return [node_features,edges.unsqueeze(dim=1),embedding.unsqueeze(dim=1)]



