import os
import torch
from torch.utils.data import DataLoader,random_split
import pandas as pd
import random
import numpy as np

from gitiii.dataloader import GITIII_dataset
from gitiii.model import GITIII,Loss_function
from gitiii.calculate_PCC import Calculate_PCC

def train_GITIII(num_neighbors=50,batch_size=256,lr=1e-4,data_dir=None,epochs=50,node_dim=256,
                 edge_dim=48,att_dim=8,use_cell_type_embedding=True):
    '''

    :param num_neighbors: number of neighboring cells used to predict the cell state of the center cell
    :param batch_size:
    :param lr: learning rate
    :param data_dir: directory of which the preprocessed dataset resides in
    :param epochs: number of training rounds
    :param node_dim: embedding dimension for node in the graph transformer
    :param edge_dim: embedding dimension for edge in the graph transformer
    :param att_dim: dimension needed to calculate for one-head attention
    :return:
    '''
    # Preparation
    if data_dir is None:
        data_dir=os.path.join(os.getcwd(),"data","processed")
    if data_dir[-1]!="/":
        data_dir=data_dir+"/"
    torch.cuda.empty_cache()

    # Load preprocessed dataset
    print("Start loading the dataset")
    dataset = GITIII_dataset(processed_dir=data_dir,num_neighbors=num_neighbors)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    validation_size = total_size - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    print("Finish loading the dataset")
    print("Train loader length:", len(train_loader))
    print("Validation loader length:", len(val_loader))

    # Define the model
    ligands_info = torch.load("/".join(data_dir.split("/")[:-2]) + "/ligands.pth",weights_only=False)
    genes = torch.load("/".join(data_dir.split("/")[:-2]) + "/genes.pth",weights_only=False)
    my_model = GITIII(genes, ligands_info, node_dim=node_dim, edge_dim=edge_dim, num_heads=2,
                      n_layers=1, node_dim_small=16, att_dim=att_dim,
                      use_cell_type_embedding=use_cell_type_embedding)
    my_model = my_model.cuda()
    optimizer = torch.optim.AdamW(my_model.parameters(), lr=lr, betas=(0.99, 0.999))
    loss_func = Loss_function(genes, ligands_info).cuda()
    evaluator = Calculate_PCC(genes, ligands_info)

    records = []
    best_val = 1e10

    potential_saved_name = os.path.join(os.getcwd(),"GRIT.pth")
    if sum([x.find(potential_saved_name) >= 0 for x in os.listdir(".")]) > 0:
        checkpoint = torch.load(potential_saved_name)
        my_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_val = checkpoint['best_val']
        records = checkpoint['records']

    print("Start training")
    for epochi in range(epochs):
        my_model.train()
        loss_train1 = 0
        loss_train2 = 0
        y_preds_train = []
        y_train = []
        for (stepi, x) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            x = {k: v.cuda() for k, v in x.items()}
            y_pred = my_model(x)
            y = x["y"]

            lossi1, lossi2 = loss_func(y_pred, y)

            evaluator.add_input(y_pred, y)

            lossi1.backward()
            optimizer.step()

            loss_train1 = loss_train1 + lossi1.cpu().item()
            loss_train2 = loss_train2 + lossi2.cpu().item()

            if stepi % 500 == 0:
                PCC1, PCC2 = evaluator.calculate_pcc()
                print("Training-> epoch:", epochi, "step:", stepi, "loss_all:", loss_train1 / stepi,
                      "loss_not_interaction:", loss_train2 / stepi, "median PCC all:", torch.median(PCC1),
                      "median PCC_not_interact:", torch.median(PCC2), "max PCC all:", torch.max(PCC1),
                      "max PCC_not_interact:", torch.max(PCC2))

        PCC1_train, PCC2_train = evaluator.calculate_pcc(clear=True)
        print("Finish training, epoch:", epochi)
        print("Loss_all:", loss_train1 / len(train_loader), "; Loss_not_interaction:", loss_train2 / len(train_loader),
              "median PCC all:", torch.median(PCC1_train), "median PCC_not_interact:", torch.median(PCC2_train),
              "max PCC all:", torch.max(PCC1_train), "max PCC_not_interact:", torch.max(PCC2_train))

        checkpoints = {
            'model': my_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'records': records,
            'best_val': best_val
        }
        torch.save(checkpoints, potential_saved_name)

        loss_val1 = 0
        loss_val2 = 0
        my_model.eval()
        with torch.no_grad():
            for (stepi, x) in enumerate(val_loader, start=1):
                x = {k: v.cuda() for k, v in x.items()}
                y_pred = my_model(x)
                y = x["y"]

                if stepi <= 10:
                    alphas = torch.abs(y_pred[1][0][0, :, 0, :])
                    print(torch.topk(torch.mean(alphas / torch.sum(alphas, dim=-1, keepdim=True), dim=0), k=30, dim=-1))

                evaluator.add_input(y_pred, y)

                lossi1, lossi2 = loss_func(y_pred, y)
                loss_val1 = loss_val1 + lossi1.cpu().item()
                loss_val2 = loss_val2 + lossi2.cpu().item()

                torch.cuda.empty_cache()
                if stepi % 500 == 0:
                    PCC1, PCC2 = evaluator.calculate_pcc()
                    print("Validating-> epoch:", epochi, "step:", stepi, "loss_all:", loss_val1 / stepi,
                          "loss_not_interaction:", loss_val2 / stepi, "median PCC all:", torch.median(PCC1),
                          "median PCC_not_interact:", torch.median(PCC2), "max PCC all:", torch.max(PCC1),
                          "max PCC_not_interact:", torch.max(PCC2))

        PCC1_val, PCC2_val = evaluator.calculate_pcc(clear=True)
        print("Finish validating, epoch:", epochi)
        print("Loss_all:", loss_val1 / len(val_loader), "; Loss_not_interaction:", loss_val2 / len(val_loader),
              "median PCC all:", torch.median(PCC1_val), "median PCC_not_interact:", torch.median(PCC2_val),
              "max PCC all:", torch.max(PCC1_val), "max PCC_not_interact:", torch.max(PCC2_val))

        records.append([epochi, loss_train1 / len(train_loader), loss_train2 / len(train_loader), PCC1_train,
                        PCC2_train, loss_val1 / len(val_loader), loss_val2 / len(val_loader), PCC1_val, PCC2_val])
        df = pd.DataFrame(data=records, columns=["epoch", "train_loss_interaction", "train_loss_downstream",
                                                 "PCC1_train", "PCC2_train", "val_loss_interaction",
                                                 "val_loss_downstream", "PCC1_val", "PCC2_val"])
        df.to_csv(os.path.join(os.getcwd(),"record_GRIT.csv"))
        if best_val > loss_val1 / len(val_loader):
            best_val = loss_val1 / len(val_loader)
            torch.save(my_model.state_dict(), os.path.join(os.getcwd(),'GRIT_best.pth'))
        print("Best validation loss now:", best_val)

