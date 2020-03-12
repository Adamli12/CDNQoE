import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='AE M-protain')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5000, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

class FMmodule(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.
    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.assistmatrix=nn.Parameter(torch.zeros(field_dims,embed_dim))
        #torch.nn.init.xavier_uniform_(self.assistmatrix.data)
        self.linear = nn.Linear(field_dims,1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        square_of_sum = torch.sum(torch.mm(x,self.assistmatrix), dim=1) ** 2
        sum_of_square = torch.sum(torch.mm(x,self.assistmatrix) ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        x = self.linear(x)+ + 0.5 * ix.view((-1,1))
        #x=F.softmax(x,dim=1)
        return x

class FactorizationMachine:
    def __init__(self, field_dims, embed_dim, lr, results_dir, args=args):
        self.epoch_num = args.epochs
        self.module=FMmodule(field_dims,embed_dim).to(device)
        self.optimizer = optim.Adam(self.module.parameters(), lr=lr,weight_decay=0)
        self.criterion = nn.MSELoss(reduction="sum")
        self.results_dir=os.path.join(results_dir,str(field_dims)+"_"+str(embed_dim)+"_"+str(lr))
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def train(self,train_loader,test_loader):
        self.module.train()
        loss_curve=[]
        test_loss_curve=[]
        for epoch in range(1,self.epoch_num+1):
            train_loss = 0
            for batch_idx, (data,label) in enumerate(train_loader):
                data = data.to(device)
                label = label.to(device)
                data=data.float()
                label = label.float()
                self.optimizer.zero_grad()
                pred= self.module(data)
                loss = self.criterion(pred,label.view(len(data),1))
                loss.backward()
                train_loss += loss.item()
                loss_curve.append(loss.item() / len(data))
                self.optimizer.step()
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx * len(data)/ len(train_loader.dataset),
                        loss.item() / len(data)))
            print('====> Train Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(train_loader.dataset)))
            test_loss_curve.append(self.test(test_loader))
        plt.clf()
        plt.plot(loss_curve)
        plt.yscale('log')
        plt.title('training loss curve')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir,"train_loss"))
        plt.clf()
        plt.plot(test_loss_curve)
        plt.yscale('log')
        plt.title('testing loss curve')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir,"test_loss"))

    def test(self, testloader):
        self.module.eval()
        eval_loss = 0.
        for data,label in testloader:
            data = data.to(device)
            label = label.to(device)
            data=data.float()
            label = label.float()
            pred= self.module(data)
            loss = self.criterion(pred,label.view(len(data),1))
            eval_loss += loss.item()
        print('====> Test Average loss: {:.4f}'.format(eval_loss / len(testloader.dataset)))
        return eval_loss / len(testloader.dataset)


    def save(self,path="trained_model.pth"):
    #save
        torch.save(self.module.state_dict(), os.path.join(self.results_dir,path))
        return 0

    def load(self,path="trained_model.pth"):
        self.module.load_state_dict(torch.load(os.path.join(self.results_dir,path)))