import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads, dropout=0.1, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Initialize parameters efficiently
        self.W = nn.Parameter(torch.empty(in_features, out_features * n_heads))
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        
    def forward(self, x, adj):
        # Compute linear transformation and reshape in one step
        h = torch.mm(x, self.W).view(x.size(0), self.n_heads, self.out_features)
        
        # Compute attention scores efficiently using broadcasting
        h_i = h.unsqueeze(1)  # [N, 1, n_heads, out_features] [3, 1, 4, 64]
        h_j = h.unsqueeze(0)  # [1, N, n_heads, out_features] [1, 3, 4, 64]
        
        # Expand the tensors to match dimensions
        h_i = h_i.expand(-1, h.size(0), -1, -1)  # [3, 3, 4, 64]
        h_j = h_j.expand(h.size(0), -1, -1, -1)  # [3, 3, 4, 64]
        
        # Transpose h_j to align dimensions for matrix multiplication
        h_j = h_j.transpose(-1, -2)  # [3, 3, 64, 4]
        
        # Now matmul will work: [3, 3, 4, 64] x [3, 3, 64, 4] -> [3, 3, 4, 4]
        e = self.leakyrelu(torch.matmul(h_i, h_j))
        
        # Ensure e has the correct shape before applying attention
        e = e.mean(dim=-1)  # Average over the last dimension to get [3, 3, 4]
        
        # Now apply the attention mask
        attention = torch.where(
            adj.unsqueeze(2) > 0,  # [3, 3, 1] -> broadcast to [3, 3, 4]
            e,
            float('-inf') * torch.ones_like(e)
        )
        
        # Apply attention mechanism efficiently
        attention = F.dropout(
            F.softmax(attention, dim=1),
            self.dropout,
            training=self.training
        )
        
        # Compute final output efficiently
        out = torch.bmm(
            attention.permute(2, 0, 1),
            h.permute(1, 0, 2)
        ).permute(1, 0, 2).mean(dim=1)
        
        return F.elu(out)


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Linear transformation
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
            
    def forward(self, x, adj):
        # Normalize adjacency matrix
        deg = torch.sum(adj, dim=1)
        deg = torch.clamp(deg, min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt_mat = torch.diag(deg_inv_sqrt)
        norm_adj = torch.mm(torch.mm(deg_inv_sqrt_mat, adj), deg_inv_sqrt_mat)
        
        # GCN propagation rule
        support = self.linear(x)
        output = torch.mm(norm_adj, support)
        return F.relu(output)

class SAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Linear transformations for self and neighbor features
        self.linear_self = nn.Linear(in_features, out_features)
        self.linear_neigh = nn.Linear(in_features, out_features)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_self.weight)
        nn.init.xavier_uniform_(self.linear_neigh.weight)
        if self.linear_self.bias is not None:
            nn.init.zeros_(self.linear_self.bias)
        if self.linear_neigh.bias is not None:
            nn.init.zeros_(self.linear_neigh.bias)
            
    def forward(self, x, adj):
        # Aggregate neighbor features
        neigh_mean = torch.mm(adj, x) / (torch.sum(adj, dim=1, keepdim=True) + 1e-6)
        
        # Transform self and neighbor features
        from_self = self.linear_self(x)
        from_neighs = self.linear_neigh(neigh_mean)
        
        # Combine and apply non-linearity
        output = F.relu(from_self + from_neighs)
        return output

