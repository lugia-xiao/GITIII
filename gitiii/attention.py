import torch
import torch.nn as nn
import torch.nn.functional as F

from gitiii.embedding import FFN

def rho(x):
    return torch.sqrt(F.relu(x)) - torch.sqrt(F.relu(-x))

def get_diagonal(x):
    index=torch.arange(0,x.shape[1],1)
    return x[:,index,index,:]

class GRIT_attention(nn.Module):
    def __init__(self, node_dim, edge_dim,att_dim=8):
        super().__init__()
        self.edge_dim=edge_dim*2

        self.W_Q = nn.Linear(node_dim, self.edge_dim*att_dim)
        self.W_K = nn.Linear(node_dim, self.edge_dim*att_dim)
        self.W_V = nn.Linear(node_dim, node_dim)
        self.W_Ew = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_Eb = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_En = nn.Linear(edge_dim, node_dim)
        self.W_A = nn.Linear(edge_dim, 1, bias=False)

        self.W_No = nn.Linear(node_dim, node_dim)
        self.W_Eo = nn.Linear(edge_dim, edge_dim, bias=False)

    def forward(self, x):
        node, edge, embedding = x
        B, N, C = node.shape

        Q = self.W_Q(node).reshape(B,N,-1,self.edge_dim).permute(0,3,1,2)#.unsqueeze(dim=1).repeat(1,N,1,1)#b,n,h*c->b,n,c,h->b,h,n,c
        K = self.W_K(node).reshape(B,N,-1,self.edge_dim).permute(0,3,2,1)#.unsqueeze(dim=2).repeat(1,1,N,1)#b,n,h*c->b,n,c,h->b,h,c,n
        QK=(Q@K).permute(0,2,3,1)
        #print("abs", torch.mean(torch.abs(rho((QK[:,:,:,:self.edge_dim//2])*self.W_Ew(edge)))), torch.mean(torch.abs(self.W_Eb(edge))),torch.mean(torch.abs(QK[:,:,:,self.edge_dim//2:]*embedding)))
        edge = F.gelu(rho((QK[:,:,:,:self.edge_dim//2])*self.W_Ew(edge))+self.W_Eb(edge)+QK[:,:,:,self.edge_dim//2:]+embedding)  # b,n,n,c_e

        alpha=self.W_A(edge).squeeze(dim=-1)
        alpha = F.softmax(alpha,dim=-1)

        V = self.W_V(node)
        node = alpha @ V + self.W_En(torch.sum(edge*alpha.unsqueeze(dim=-1),dim=-2))

        node = self.W_No(node)
        edge = self.W_Eo(edge)
        return [node, edge]

class DegScaler(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        scaler = (2 / node_dim) ** 0.5  # He Initialization
        self.theta1 = nn.Parameter(torch.randn(node_dim) * scaler)
        self.theta2 = nn.Parameter(torch.randn(node_dim) * scaler)

    def forward(self, node, degree):
        node = self.theta1 * node + torch.log(degree + 1).unsqueeze(dim=-1) * node * self.theta2
        return node

class Multi_Head_Attention(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads, att_dim=8):
        super().__init__()
        self.attentions = nn.ModuleList(
            [GRIT_attention(node_dim, edge_dim, att_dim) for i in range(num_heads)]
        )

        self.W_hn = nn.Parameter(torch.ones(num_heads, 1)/num_heads)
        self.W_he = nn.Parameter(torch.ones(num_heads, 1) / num_heads)

    def forward(self, x):
        results = []
        for attentioni in self.attentions:
            results.append(attentioni(x))

        node = (torch.stack([tmp[0] for tmp in results], dim=-1)@self.W_hn).squeeze(dim=-1)

        edge = [tmp[1] for tmp in results]
        edge = torch.stack(edge, dim=-1)@self.W_he
        edge = edge.squeeze(dim=-1)

        return [node, edge, x[2]]

class GRIT_encoder(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads, att_dim=8):
        super().__init__()
        self.attentions = Multi_Head_Attention(node_dim, edge_dim, num_heads, att_dim)
        self.FFN = FFN(node_dim)

        self.ln1 = nn.LayerNorm(node_dim)
        self.ln2 = nn.LayerNorm(node_dim)
        self.ln_edge=nn.LayerNorm(edge_dim)

    def forward(self, x):
        node, edge, distance = x

        x = self.attentions(x)

        edge = self.ln_edge(edge + x[1])

        node = self.ln1(node+ x[0])
        node = self.ln2(node + self.FFN(node))
        return [node, edge, distance]

class GRIT_encoder_last_layer(nn.Module):
    def __init__(self, node_dim, in_node, edge_dim, node_dim_small=16, att_dim=8):
        super().__init__()
        self.node_dim_small = node_dim_small
        self.in_node = in_node
        self.edge_dim=edge_dim*2
        self.att_dim=att_dim

        self.W_Q = nn.Linear(node_dim, self.edge_dim * att_dim)
        self.W_K = nn.Linear(node_dim, self.edge_dim * att_dim)
        self.W_Ew = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_Eb = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_En = nn.Linear(edge_dim, node_dim)

        self.W_A=nn.Linear(edge_dim,in_node,bias=False)

        self.node_transform = nn.Linear(node_dim, in_node)
        self.edge_transform = nn.Linear(edge_dim, in_node)
        self.head = FFN(in_node,in_dim=in_node*2, out_dim=in_node)
        self.head2=nn.Sequential(nn.LayerNorm(in_node),FFN(in_node))

        self.scaler=2/(node_dim**0.5)

    def forward(self, x):
        node, edge, embedding = x
        B, N, C = node.shape

        Q = self.W_Q(node[:,0:1,:]).reshape(B, 1, self.att_dim, self.edge_dim).permute(0,3,1,2)#b,n,h*c->b,n,c,h->b,h,n,c
        K = self.W_K(node).reshape(B, N, self.att_dim, self.edge_dim).permute(0,3,2,1)#b,n,h*c->b,n,c,h->b,h,c,n
        QK = (Q @ K).permute(0, 2, 3, 1)

        edge=edge[:,0:1,:,:]
        #print("abs",torch.mean(torch.abs(QK)),torch.mean(torch.abs(edge)))
        edge = F.gelu(rho((QK[:, :, :, :self.edge_dim // 2]) * self.W_Ew(edge)) + self.W_Eb(edge) + QK[:, :, :, self.edge_dim // 2:] + embedding[:,0:1,:,:])  # b,n,n,c_e
        edge = edge[:, 0, 1:, :]

        alphas = F.softmax(self.W_A(edge).permute(0, 2, 1), dim=-1).permute(0, 2, 1)

        edge = self.edge_transform(edge)*alphas
        node = self.node_transform(node[:, 1:, :])*alphas
        tmp = self.head(torch.concat([edge, node], dim=-1))

        node = torch.sum(tmp,dim=-2)*self.scaler  # b,in_node,99->1
        return [node, [tmp.permute(0,2,1).unsqueeze(dim=-2), edge.permute(0,2,1).unsqueeze(dim=-2)]]
