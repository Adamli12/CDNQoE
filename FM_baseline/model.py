import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='live data FM model')
parser.add_argument('--epochs', type=int, default=7, metavar='N',
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

class Linearmodule(torch.nn.Module):

    def __init__(self, field_dims):
        super().__init__()
        self.linear = nn.Linear(field_dims,1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x)
        #x=F.softmax(x,dim=1)
        return x

class MLPmodule(torch.nn.Module):

    def __init__(self, field_dims):
        super().__init__()
        self.linear0 = nn.Linear(field_dims,32)
        self.linear1 = nn.Linear(32,16)
        self.linear2 = nn.Linear(16,8)
        self.linear3 = nn.Linear(8,1)
        self.net = nn.Sequential(self.linear0,nn.ReLU(),self.linear1,nn.ReLU(),self.linear2,nn.ReLU(),self.linear3)
        

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.net(x)
        #x=F.softmax(x,dim=1)
        return x

class tagmean_module():
    def __init__(self, n_tags, tag_begin):#how many tags are there and where does it begin in the history vector
        self.record=np.zeros((n_tags,2))#first entry is sum, second is total time
        self.n_tags=n_tags
        self.tag_begin=tag_begin
    def add(self,x,y):#x: N*len(vector), y: (N,)
        tagx=x[:,self.tag_begin:self.tag_begin+self.n_tags]
        tags=tagx.argmax(1)
        for i in range(len(tags)):
            self.record[tags[i]][0]+=y[i]
            self.record[tags[i]][1]+=1
        
    def forward(self,x):
        tagx=x[:,self.tag_begin:self.tag_begin+self.n_tags]
        tags=tagx.argmax(1)
        preds=self.record[tags,0]/self.record[tags,1]
        preds[np.isnan(preds)]=0.5
        return preds

class FactorizationMachine:
    def __init__(self, field_dims, embed_dim, lr, n_classes, results_dir, model="FM", n_tags=57, tags_begin=39, args=args):
        self.epoch_num = args.epochs
        self.model=model
        if model=="FM":
            self.module=FMmodule(field_dims,embed_dim).to(device)
            self.criterion = nn.MSELoss(reduction="sum")
            self.optimizer = optim.Adam(self.module.parameters(), lr=lr,weight_decay=0)
        elif model=="Linear":
            self.module=Linearmodule(field_dims).to(device)
            self.criterion = nn.MSELoss(reduction="sum")
            self.optimizer = optim.Adam(self.module.parameters(), lr=lr,weight_decay=0)
        elif model=="MLP":
            self.module=MLPmodule(field_dims).to(device)
            self.criterion = nn.MSELoss(reduction="sum")
            self.optimizer = optim.Adam(self.module.parameters(), lr=lr,weight_decay=0)
        elif model=="SVR":
            self.module=SVR(max_iter=1000)
        elif model=="tag_mean":
            self.module=tagmean_module(n_tags,tags_begin)
        self.results_dir=os.path.join(results_dir,str(field_dims)+"_"+str(embed_dim)+"_"+str(lr)+"_"+model)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.n_classes=n_classes
        self.logfile=str()
        self.logfile=str(args)+"\n"+"field_dim:"+str(field_dims)+", embed_dim:"+str(embed_dim)+", n_classes:"+str(n_classes)+", lr:"+str(lr)+", model:"+str(model)+"\n"

    def same_class(self,a,b):
        return int(a*self.n_classes)==int(b*self.n_classes)

    def count_same(self,a,b):
        s=0
        for i in range(len(a)):
            if self.same_class(a[i][0],b[i][0]):
                s+=1
        return s

    def MSE(self,x,y):
        return np.sum(np.square(x-y))/len(x)


    def train(self,train_loader,test_loader):
        if self.model=="FM" or self.model=="Linear" or self.model=="MLP":
            self.module.train()
            loss_curve=[]
            test_loss_curve=[]
            acc_curve=[]
            test_acc_curve=[]
            for epoch in range(1,self.epoch_num+1):
                train_loss = 0
                acc = 0
                for batch_idx, (data,label) in enumerate(train_loader):
                    data = data.to(device)
                    label = label.to(device)
                    data = data.float()
                    label = label.float().view(len(data),1)
                    self.optimizer.zero_grad()
                    pred= self.module(data)
                    loss = self.criterion(pred,label)
                    loss.backward()
                    ac = self.count_same(label.detach(),pred.detach())
                    train_loss += loss.item()
                    acc += ac
                    loss_curve.append(loss.item() / len(data))
                    acc_curve.append(ac/len(data))
                    self.optimizer.step()
                    if batch_idx % args.log_interval == 0:
                        log='Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.3f}'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx * len(data)/ len(train_loader.dataset),
                            loss.item() / len(data),
                            ac/len(data))
                        print(log)
                        self.logfile+=log+"\n"
                log='====> Train Epoch: {} Average loss: {:.4f} accuracy: {:.3f}'.format(
                    epoch, train_loss / len(train_loader.dataset),acc/len(train_loader.dataset))
                print(log)
                self.logfile+=log+"\n"
                tloss,tacc=self.test(test_loader)
                test_loss_curve.append(tloss)
                test_acc_curve.append(tacc)
                if tacc==max(test_acc_curve):
                    self.save
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
            plt.clf()
            plt.plot(test_acc_curve)
            plt.title('testing acc curve')
            plt.savefig(os.path.join(self.results_dir,"test_acc"))
        elif self.model=="SVR":
            self.module.fit(train_loader[0],train_loader[1])
            pred=self.module.predict(train_loader[0])
            MSE_score=self.MSE(pred,train_loader[1])
            log=self.model+"model training MSE score: "+str(MSE_score)
            print(log)
            self.logfile+=log+"\n"
            ac = self.count_same(pred.reshape((-1,1)),train_loader[1].reshape((-1,1)))/pred.shape[0]
            log=self.model+"model training class acc: "+str(ac)
            print(log)
            self.logfile+=log+"\n"
            self.test(test_loader)
        elif self.model=="tag_mean":
            self.module.add(train_loader[0],train_loader[1])
            pred=self.module.forward(train_loader[0])
            MSE_score=self.MSE(pred,train_loader[1])
            log=self.model+"model training MSE score: "+str(MSE_score)
            print(log)
            self.logfile+=log+"\n"
            ac = self.count_same(pred.reshape((-1,1)),train_loader[1].reshape((-1,1)))/pred.shape[0]
            log=self.model+"model training class acc: "+str(ac)
            print(log)
            self.logfile+=log+"\n"
            self.test(test_loader)
        with open(os.path.join(self.results_dir,"log.txt"),"a") as f:
            f.write(self.logfile)


    def test(self, test_loader):
        if self.model=="FM" or self.model=="Linear" or self.model=="MLP":
            self.module.eval()
            eval_loss = 0.
            acc=0
            for data,label in test_loader:
                data = data.to(device)
                label = label.to(device)
                data=data.float()
                label = label.float().view(len(data),1)
                pred= self.module(data)
                acc+=self.count_same(label,pred)
                loss = self.criterion(pred,label.view(len(data),1))
                eval_loss += loss.item()
            log='====> Test Average loss: {:.4f} accuracy:{:.3f}'.format(eval_loss / len(test_loader.dataset),acc / len(test_loader.dataset))
            print(log)
            self.logfile+=log+"\n"
            return eval_loss / len(test_loader.dataset),acc / len(test_loader.dataset)
        elif self.model=="SVR":
            pred=self.module.predict(test_loader[0])
            MSE_score=self.MSE(pred,test_loader[1])
            log=self.model+"model testing MSE score: "+str(MSE_score)
            print(log)
            self.logfile+=log+"\n"
            ac = self.count_same(pred.reshape((-1,1)),test_loader[1].reshape((-1,1)))/pred.shape[0]
            log=self.model+"model testing class acc: "+str(ac)
            print(log)
            self.logfile+=log+"\n"
        elif self.model=="tag_mean":
            pred=self.module.forward(test_loader[0])
            MSE_score=self.MSE(pred,test_loader[1])
            log=self.model+"model testing MSE score: "+str(MSE_score)
            print(log)
            self.logfile+=log+"\n"
            ac = self.count_same(pred.reshape((-1,1)),test_loader[1].reshape((-1,1)))/pred.shape[0]
            log=self.model+"model testing class acc: "+str(ac)
            print(log)
            self.logfile+=log+"\n"



    def save(self,path="trained_model.pth"):
    #save
        if self.model=="Linear" or self.model=="FM" or self.model=="MLP":
            torch.save(self.module.state_dict(), os.path.join(self.results_dir,path))
        elif self.model=="SVR":
            with open(os.path.join(self.results_dir,path),"wb") as f:
                pickle.dump(self.module,f)
        elif self.model=="tag_mean":
            with open(os.path.join(self.results_dir,path),"wb") as f:
                pickle.dump(self.module,f)
        return 0

    def load(self,path="trained_model.pth"):
        if self.model=="Linear" or self.model=="FM" or self.model=="MLP":
            self.module.load_state_dict(torch.load(os.path.join(self.results_dir,path)))
        elif self.model=="SVR":
            with open(os.path.join(self.results_dir,path),"rb") as f:
                self.module=pickle.load(f)
        elif self.model=="tag_mean":
            with open(os.path.join(self.results_dir,path),"rb") as f:
                self.module=pickle.load(f)