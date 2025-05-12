import torch

def process(x,judge_distance=28):
    '''
    :param x:
        "x": exp,
        "y": y,
        "ligand":ligand,
        "receptor":receptor,
        "cell_types": cell_types,
        "morphology": volumes,
        "position_x": centerx,
        "position_y": centery
    :return:
        "x": b,n,c1
        "interactions": b,n,n,c2
        "mask": b,n,n
        "cell_types": b,n
        "distance_matrix": b,n,n,c3
        "y": b,c
        "morphology": b,n,c4
    '''
    dx = x["position_x"][:, 0:1] - x["position_x"]
    dy = x["position_y"][:, 0:1] - x["position_y"]
    distances=torch.sqrt(torch.square(dx)+torch.square(dy))

    '''mask=torch.where(distances<judge_distance,
                     torch.ones_like(distances,device=distances.device),
                     torch.zeros_like(distances,device=distances.device))'''
    distance_matrix = torch.stack([1 / (distances + 1), 1 / (torch.sqrt(distances) + 1),
                                   1 / (1 + torch.square(distances)), torch.exp(-distances),
                                   torch.exp(-torch.square(distances))], dim=-1)
    return {
        "x":x["x"],
        "type_exp": x["type_exp"],
        "y":x["y"],
        "cell_types": x["cell_types"],
        "distance_matrix":distance_matrix
    }