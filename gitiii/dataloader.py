import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd

class GITIII_dataset(Dataset):
    def __init__(self, processed_dir, num_neighbors=50, col_centerx="centerx",
                 col_centery="centery", col_cell_type="subclass"):
        if processed_dir[-1] != "/":
            processed_dir = processed_dir + "/"

        samples = []
        for filei in os.listdir(processed_dir):
            if filei.find("_TypeExp.npz") >= 0:
                samples.append(filei.split("_TypeExp.npz")[0])
        self.samples = list(sorted(list(set(samples))))
        print("Have samples:",len(self.samples),self.samples)

        self.index_index = ["index_" + str(i) for i in range(num_neighbors)]
        self.interactions = torch.load("/".join(processed_dir.split("/")[:-2]) + "/ligands.pth",weights_only=False)
        self.genes = torch.load("/".join(processed_dir.split("/")[:-2]) + "/genes.pth",weights_only=False)

        cell_types = torch.load(os.path.join(processed_dir,"cell_types.pth"),weights_only=False)
        # build cell type dictionary
        self.cell_types_dict = {}
        cnt = 0
        for cell_typei in cell_types:
            self.cell_types_dict[cell_typei] = cnt
            cnt += 1

        self.indexes = []
        self.flags = []
        self.centerx = []
        self.centery = []
        self.exps = []
        self.cell_types = []
        self.type_exp = []
        for samplei in self.samples:
            dfi = pd.read_csv(processed_dir + samplei + ".csv")
            dfi["cell_type_number"] = np.array([self.cell_types_dict[cell_type] for cell_type in dfi[col_cell_type]])
            if "flag" not in dfi.columns.tolist():
                dfi["flag"] = [True for i in range(dfi.shape[0])]

            type_exp_dicti = np.load(processed_dir + samplei + "_TypeExp.npz", allow_pickle=True)
            type_exp_dicti = {k: torch.Tensor(np.array(v).astype(np.float64)) for k, v in type_exp_dicti.items()}
            type_exp = torch.stack([type_exp_dicti[cell_type] for cell_type in dfi[col_cell_type]], dim=0)
            self.type_exp.append(type_exp)

            self.indexes.append(torch.LongTensor(dfi.loc[:, self.index_index].values))
            self.flags.append(dfi.loc[:, "flag"].values)
            self.centerx.append(torch.Tensor(dfi.loc[:, col_centerx].values))
            self.centery.append(torch.Tensor(dfi.loc[:, col_centery].values))
            self.exps.append(torch.Tensor(dfi.loc[:, self.genes].values))
            self.cell_types.append(torch.LongTensor(dfi.loc[:, "cell_type_number"].values))

        self.meta_counts = []
        self.arg_meta = []
        for i in range(len(self.samples)):
            self.meta_counts.append(np.sum(self.flags[i]))
            self.arg_meta.append(torch.LongTensor(np.where(self.flags[i] != 0)[0]))
        self.meta_counts = np.array(self.meta_counts)
        self.cumsum_meta = np.cumsum(self.meta_counts)

        print("There are totally", self.cumsum_meta[-1], "cells in this dataset")

    def __len__(self):
        return self.cumsum_meta[-1]

    def __getitem__(self, idx):
        # start_time = time.time()
        # Know which sample does this idx come from
        sample_id = np.searchsorted(self.cumsum_meta, idx, side='right')
        idx = idx - self.cumsum_meta[sample_id]
        idx = self.arg_meta[sample_id][idx]

        indices = self.indexes[sample_id][idx]
        centerx = self.centerx[sample_id][indices]
        centery = self.centery[sample_id][indices]

        exp = self.exps[sample_id][indices]
        type_exp = self.type_exp[sample_id][indices]
        y = exp[0]
        cell_types = torch.LongTensor(self.cell_types[sample_id][indices])

        return {
            "x": exp,
            "type_exp": type_exp,
            "y": y,
            "cell_types": cell_types,
            "position_x": centerx,
            "position_y": centery
        }

class GITIII_evaluate_dataset(Dataset):
    def __init__(self, processed_dir, sample, num_neighbors=50, col_centerx="centerx",
                 col_centery="centery", col_cell_type="subclass"):
        if processed_dir[-1] != "/":
            processed_dir = processed_dir + "/"

        self.samples = [sample]
        print("Have samples:",self.samples)

        self.index_index = ["index_" + str(i) for i in range(num_neighbors)]
        self.interactions = torch.load("/".join(processed_dir.split("/")[:-2]) + "/ligands.pth",weights_only=False)
        self.genes = torch.load("/".join(processed_dir.split("/")[:-2]) + "/genes.pth",weights_only=False)

        cell_types = torch.load(os.path.join(processed_dir,"cell_types.pth"),weights_only=False)
        # build cell type dictionary
        self.cell_types_dict = {}
        cnt = 0
        for cell_typei in cell_types:
            self.cell_types_dict[cell_typei] = cnt
            cnt += 1

        self.indexes = []
        self.flags = []
        self.centerx = []
        self.centery = []
        self.exps = []
        self.cell_types = []
        self.type_exp = []
        for samplei in self.samples:
            dfi = pd.read_csv(processed_dir + samplei + ".csv")
            dfi["cell_type_number"] = np.array([self.cell_types_dict[cell_type] for cell_type in dfi[col_cell_type]])
            if "flag" not in dfi.columns.tolist():
                dfi["flag"] = [True for i in range(dfi.shape[0])]

            type_exp_dicti = np.load(processed_dir + samplei + "_TypeExp.npz", allow_pickle=True)
            type_exp_dicti = {k: torch.Tensor(np.array(v).astype(np.float64)) for k, v in type_exp_dicti.items()}
            type_exp = torch.stack([type_exp_dicti[cell_type] for cell_type in dfi[col_cell_type]], dim=0)
            self.type_exp.append(type_exp)

            self.indexes.append(torch.LongTensor(dfi.loc[:, self.index_index].values))
            self.flags.append(dfi.loc[:, "flag"].values)
            self.centerx.append(torch.Tensor(dfi.loc[:, col_centerx].values))
            self.centery.append(torch.Tensor(dfi.loc[:, col_centery].values))
            self.exps.append(torch.Tensor(dfi.loc[:, self.genes].values))
            self.cell_types.append(torch.LongTensor(dfi.loc[:, "cell_type_number"].values))

        self.meta_counts = []
        self.arg_meta = []
        for i in range(len(self.samples)):
            self.meta_counts.append(np.sum(self.flags[i]))
            self.arg_meta.append(torch.LongTensor(np.where(self.flags[i] != 0)[0]))
        self.meta_counts = np.array(self.meta_counts)
        self.cumsum_meta = np.cumsum(self.meta_counts)

        print("There are totally", self.cumsum_meta[-1], "cells in this dataset")

    def __len__(self):
        return self.cumsum_meta[-1]

    def __getitem__(self, idx):
        # start_time = time.time()
        # Know which sample does this idx come from
        sample_id = np.searchsorted(self.cumsum_meta, idx, side='right')
        idx = idx - self.cumsum_meta[sample_id]
        idx = self.arg_meta[sample_id][idx]

        indices = self.indexes[sample_id][idx]
        centerx = self.centerx[sample_id][indices]
        centery = self.centery[sample_id][indices]

        exp = self.exps[sample_id][indices]
        type_exp = self.type_exp[sample_id][indices]
        y = exp[0]
        cell_types = torch.LongTensor(self.cell_types[sample_id][indices])

        indices = torch.LongTensor(indices)
        return {
            "x": exp,
            "type_exp": type_exp,
            "y": y,
            "cell_types": cell_types,
            "position_x": centerx,
            "position_y": centery,
            "NN":indices
        }