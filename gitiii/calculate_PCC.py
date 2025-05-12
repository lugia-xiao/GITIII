import torch


def pearson_correlation(y_pred, y):
    """
    Calculates the Pearson Correlation Coefficient for each column in two matrices.

    Args:
    y_pred (torch.Tensor): A 2D tensor representing the predicted values.
    y (torch.Tensor): A 2D tensor representing the actual values.

    Returns:
    torch.Tensor: A 1D tensor containing the PCC for each column.
    """

    # Ensure input tensors are of float type
    y_pred = y_pred.float()
    y = y.float()

    # Calculate mean
    mean_y_pred = torch.mean(y_pred, dim=0)
    mean_y = torch.mean(y, dim=0)

    # Subtract means
    y_pred_minus_mean = y_pred - mean_y_pred
    y_minus_mean = y - mean_y

    # Calculate covariance and variances
    covariance = torch.mean(y_pred_minus_mean * y_minus_mean, dim=0)
    variance_y_pred = torch.mean(y_pred_minus_mean ** 2, dim=0)
    variance_y = torch.mean(y_minus_mean ** 2, dim=0)

    # Calculate PCC
    pcc = covariance / torch.sqrt(variance_y_pred * variance_y)

    return pcc

class Calculate_PCC:
    def __init__(self,gene_list,interaction_gene_list1):
        interaction_gene_list = list(set(elem for sublist in interaction_gene_list1[0] for elem in sublist))

        self.y_no_interact=[]
        self.y_pred_no_interact=[]
        self.y_pred=[]
        self.y=[]
        self.y_interact=[]
        self.y_pred_interact=[]

        self.interaction_gene_index = []
        self.not_interaction_gene_index = []
        for i in range(len(gene_list)):
            if gene_list[i] in interaction_gene_list:
                self.interaction_gene_index.append(i)
            else:
                self.not_interaction_gene_index.append(i)
        self.interaction_gene_index = torch.LongTensor(self.interaction_gene_index)
        self.not_interaction_gene_index = torch.LongTensor(self.not_interaction_gene_index)

    def add_input(self,y_pred,y):
        y=y.cpu().detach()
        self.y.append(y)
        self.y_no_interact.append(y[:, self.not_interaction_gene_index])
        self.y_interact.append(y[:, self.interaction_gene_index])
        if len(y_pred)==2:
            y_pred=y_pred[0].cpu().detach()
            self.y_pred.append(y_pred)
            self.y_pred_interact.append(y_pred[:, self.interaction_gene_index])
            self.y_pred_no_interact.append(y_pred[:, self.not_interaction_gene_index])
        else:
            y_pred = y_pred.cpu().detach()
            self.y_pred.append(y_pred)
            self.y_pred_interact.append(y_pred[:, self.interaction_gene_index])
            self.y_pred_no_interact.append(y_pred[:, self.not_interaction_gene_index])

    def clear(self):
        self.y_no_interact = []
        self.y_pred_no_interact = []
        self.y_pred = []
        self.y = []
        self.y_interact = []
        self.y_pred_interact = []

    def calculate_pcc(self,clear=False):
        y_interact=torch.concat(self.y_interact,dim=0)
        y_no_interact=torch.concat(self.y_no_interact,dim=0)
        y_pred_no_interact=torch.concat(self.y_pred_no_interact,dim=0)
        y_pred_interact=torch.concat(self.y_pred_interact,dim=0)
        PCC1=pearson_correlation(y_pred_interact,y_interact).cpu().detach()
        PCC2=pearson_correlation(y_pred_no_interact,y_no_interact).cpu().detach()
        if clear:
            self.clear()
        return PCC1,PCC2

    def calculate_error(self,clear=True):
        y = torch.concat(self.y, dim=0)
        y_pred = torch.concat(self.y_pred, dim=0)
        if clear:
            self.clear()
        return torch.mean(torch.square(y_pred-y),dim=0)

if __name__=="__main__":
    matrix1=torch.randn((32,100))
    matrix2 = torch.randn((32, 100))
    print(pearson_correlation(matrix1,matrix2).shape)

