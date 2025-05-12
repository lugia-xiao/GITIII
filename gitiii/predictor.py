import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import time

from gitiii.model import GITIII,Loss_function

from gitiii.dataloader import GITIII_evaluate_dataset

def Calculate_influence_tensor(num_neighbors=50,batch_size=128,node_dim=256,
                               edge_dim=48,att_dim=8):
    print("Start calculating the influence tensor, now loading pretrained model")
    data_dir=os.path.join(os.getcwd(), "data", "processed")
    if data_dir[-1]!="/":
        data_dir=data_dir+"/"

    # Get sample names
    samples = []
    for filei in os.listdir(data_dir):
        if filei.find("_TypeExp.npz") >= 0:
            samples.append(filei.split("_TypeExp.npz")[0])
    samples = list(sorted(list(set(samples))))

    # make directory
    if not os.path.exists(os.path.join(os.getcwd(),"influence_tensor")):
        os.mkdir(os.path.join(os.getcwd(),"influence_tensor"))

    # load gene information
    ligands_info = torch.load("/".join(data_dir.split("/")[:-2]) + "/ligands.pth",weights_only=False)
    genes = torch.load("/".join(data_dir.split("/")[:-2]) + "/genes.pth",weights_only=False)

    # load cell type information
    cell_types_dict = {}
    cnt = 0
    for cell_typei in torch.load(os.path.join(data_dir,"cell_types.pth"),weights_only=False):
        cell_types_dict[cnt] = cell_typei
        cnt += 1

    # define the model and load pretrained-model
    my_model = GITIII(genes, ligands_info, node_dim=node_dim, edge_dim=edge_dim, num_heads=2,
                      node_dim_small=16, att_dim=att_dim,n_layers=1)
    my_model = my_model.cuda()
    my_model.load_state_dict(torch.load(os.path.join(os.getcwd(),"GRIT_best.pth")))

    print("Start calculating the influence tensor")
    cnt=0
    for samplei in samples:
        print(cnt, "/", len(samples))
        cnt = cnt + 1
        print("Now calculating the influence tensor of:",samplei)

        my_dataset = GITIII_evaluate_dataset(processed_dir=data_dir, sample=samplei, num_neighbors=num_neighbors)
        my_dataloader = DataLoader(my_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
        length = len(my_dataloader)

        my_model.eval()

        y_preds_all = []
        ys_all = []
        cell_type_names_all = []
        position_xs_all = []
        position_ys_all = []
        attention_scores_all = []
        NNs_all = []

        with torch.no_grad():
            for (stepi, x) in enumerate(my_dataloader, start=1):
                x = {k: v.cuda() for k, v in x.items()}

                y_preds, influences = my_model(x)
                influences=influences[0].squeeze(dim=2).permute(0,2,1).contiguous()
                ys = x["y"]
                attention_scores_all.append(influences.detach().cpu())
                y_preds_all.append(y_preds.detach().cpu())
                ys_all.append(ys.detach().cpu())

                x["cell_types"] = x["cell_types"].detach().cpu().numpy().tolist()
                cell_type_name = [[cell_types_dict[int(j)] for j in x["cell_types"][i]] for i in
                                  range(len(x["cell_types"]))]
                cell_type_name = np.array(cell_type_name)
                cell_type_names_all.append(cell_type_name)

                position_x = x["position_x"].cpu().detach()
                position_y = x["position_y"].cpu().detach()
                position_xs_all.append(position_x)
                position_ys_all.append(position_y)

                NNs = x["NN"].detach().cpu().numpy()
                NNs_all.append(NNs)

                if stepi % 2000 == 0:
                    print(stepi, "/", length)

        y_preds_all = torch.concat(y_preds_all, dim=0)
        ys_all = torch.concat(ys_all, dim=0)
        cell_type_names_all = np.concatenate(cell_type_names_all, axis=0).tolist()
        position_xs_all = torch.concat(position_xs_all, dim=0)
        position_ys_all = torch.concat(position_ys_all, dim=0)
        attention_scores_all = torch.concat(attention_scores_all, dim=0)
        NNs_all = np.concatenate(NNs_all, axis=0)

        results = {
            "attention_score": attention_scores_all,
            "position_x": position_xs_all,
            "position_y": position_ys_all,
            "cell_type_name": cell_type_names_all,
            "y_pred": y_preds_all,
            "y": ys_all,
            "NN": NNs_all
        }
        torch.save(results, os.path.join(os.getcwd(),"influence_tensor", "edges_" + samplei + ".pth"))
        print("Finish", samplei)

