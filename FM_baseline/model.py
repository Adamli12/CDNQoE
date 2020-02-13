import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorizationMachine(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.
    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.assistmatrix=nn.parameter(torch.zeros(field_dims,embed_dim))
        torch.nn.init.xavier_uniform_(self.assistmatrix.data)
        self.linear = nn.Linear(field_dims,1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        square_of_sum = torch.sum(torch.mm(x,self.assistmatrix), dim=1) ** 2
        sum_of_square = torch.sum(torch.mm(x,self.assistmatrix) ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        x = self.linear(x) + 0.5 * ix
        return x