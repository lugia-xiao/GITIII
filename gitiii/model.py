import torch
import torch.nn as nn
import numpy as np
import random
import os

from gitiii.attention import GRIT_encoder,GRIT_encoder_last_layer
from gitiii.embedding import Embedding

class GITIII(nn.Module):
    def __init__(self,genes,ligands_info,node_dim,edge_dim,num_heads,n_layers,
                 node_dim_small=16,att_dim=8,use_cell_type_embedding=True):
        super().__init__()
        self.embeddings=Embedding(genes,ligands_info,node_dim,edge_dim,
                                  use_cell_type_embedding=use_cell_type_embedding)
        self.encoders=nn.ModuleList([
            GRIT_encoder(node_dim,edge_dim,num_heads,att_dim) for i in range(n_layers-1)
        ])
        self.last_layer=GRIT_encoder_last_layer(node_dim, len(genes), edge_dim, node_dim_small, att_dim)

    def forward(self,x):
        x=self.embeddings(x)
        for encoderi in self.encoders:
            x=encoderi(x)
        x=self.last_layer(x)
        return x

class Loss_function(nn.Module):
    def __init__(self,gene_list,interaction_gene_list1):
        super().__init__()
        interaction_gene_list=list(set(elem for sublist in interaction_gene_list1[0] for elem in sublist))

        self.interaction_gene_index = []
        self.not_interaction_gene_index = []
        for i in range(len(gene_list)):
            if gene_list[i] in interaction_gene_list:
                self.interaction_gene_index.append(i)
            else:
                self.not_interaction_gene_index.append(i)
        self.interaction_gene_index = torch.LongTensor(self.interaction_gene_index).cuda()
        self.not_interaction_gene_index = torch.LongTensor(self.not_interaction_gene_index).cuda()

        self.mse=nn.MSELoss()

    def forward(self,y_pred,y):
        output,edges=y_pred
        mse_loss=self.mse(output,y)
        return (mse_loss,
                self.mse(output[:,self.not_interaction_gene_index],y[:,self.not_interaction_gene_index]).detach().cpu())

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    # Python random module
    random.seed(seed_value)
    # Numpy
    np.random.seed(seed_value)
    # PyTorch
    torch.manual_seed(seed_value)
    # CUDA
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    # Additional configurations to enforce deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Setting environment variables to further ensure reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed_value)

set_seed(123)

if __name__=="__main__":
    import os

    torch.cuda.empty_cache()
    batch_size = 256
    lr = 1e-4
    folds = 5
    epochs = 200

    data_dir = "D:/project/tmp/st/"
    from torch.utils.data import DataLoader, random_split

    ligands_info = torch.load("/".join(data_dir.split("/")[:-2]) + "/ligands.pth",weights_only=False)
    genes = torch.load("/".join(data_dir.split("/")[:-2]) + "/genes.pth",weights_only=False)

    my_model = GITIII(genes, ligands_info, node_dim=256, edge_dim=48, num_heads=2, node_dim_small=16, att_dim=8,n_layers=1)
    my_model = my_model.cuda()
    optimizer = torch.optim.AdamW(my_model.parameters(), lr=lr, betas=(0.99, 0.999))
    loss_func = Loss_function(genes, ligands_info).cuda()
    import time

    for i in range(20):
        print("start training")
        start = time.time()
        for i in range(10):
            batch = {
                "x": torch.randn((batch_size, 50, len(genes))),
                "type_exp": torch.randn((batch_size, 50, len(genes))),
                "y": torch.randn((batch_size, len(genes))),
                "cell_types": torch.LongTensor([1]).unsqueeze(dim=0).repeat((batch_size, 50)),
                "position_x": torch.arange(1, 201, 4).unsqueeze(dim=0).repeat((batch_size, 1)),
                "position_y": torch.arange(1, 201, 4).unsqueeze(dim=0).repeat((batch_size, 1))
            }
            batch = {k: v.cuda() for k, v in batch.items()}
            '''for k, v in batch.items():
                print(k,v.shape,torch.mean(torch.abs(v.float())))'''
            output = my_model(batch)
            print("output", output[0].shape, torch.mean(torch.abs(output[0])), torch.mean(output[0]),
                  torch.mean(torch.abs(batch["y"])), torch.mean(batch["y"]))
            print(loss_func(output, batch["y"]), output[1][0].shape)
        print(time.time() - start)
