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
from time import time

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

    def __init__(self, feature_num, embed_dim):
        super().__init__()
        self.assistmatrix=nn.Parameter(torch.zeros(feature_num,embed_dim))
        #torch.nn.init.xavier_uniform_(self.assistmatrix.data)
        self.linear = nn.Linear(feature_num,1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        square_of_sum = torch.sum(torch.mm(x,self.assistmatrix), dim=1) ** 2
        sum_of_square = torch.sum(torch.mm(x,self.assistmatrix) ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        x = self.linear(x) + 0.5 * ix.view((-1,1))
        #x=F.sigmoid(x,dim=1)
        return x

class DeepFMmodule(nn.Module):
    """
    code from: https://github.com/chenxijun1029/DeepFM_with_PyTorch/blob/master/model/DeepFM.py

    A DeepFM network with RMSE loss for rates prediction problem.
    There are two parts in the architecture of this network: fm part for low
    order interactions of features and deep part for higher order. In this 
    network, we use bachnorm and dropout technology for all hidden layers,
    and "Adam" method for optimazation.
    You may find more details in this paper:
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
    """

    def __init__(self, feature_sizes, embedding_size=4,
                 hidden_dims=[32, 32], dropout=[0.5, 0.5]):
        """
        Initialize a new network
        Inputs: 
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - verbose: Bool
        """
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.dtype = torch.long
        self.bias = torch.nn.Parameter(torch.randn(1))
        """
            init fm part
        """
        self.fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
        self.fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        """
            init deep part
        """
        all_dims = [self.field_size * self.embedding_size] + \
            self.hidden_dims + [1]
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_'+str(i),
                    nn.Linear(all_dims[i-1], all_dims[i]))
            # nn.init.kaiming_normal_(self.fc1.weight)
            setattr(self, 'batchNorm_' + str(i),
                    nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_'+str(i),
                    nn.Dropout(dropout[i-1]))

    def forward(self, Xi, Xv):
        """
        Forward process of network. 
        Inputs:
        - Xi: A tensor of input's index, shape of (N, field_size, 1)
        - Xv: A tensor of input's value, shape of (N, field_size, 1)
        """
        """
            fm part
        """

        fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_first_order_embeddings)]
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        fm_second_order_emb_arr = [(emb(Xi[:, i, 0]) * Xv[:, i, 0].view(len(Xi),-1)) for i, emb in enumerate(self.fm_second_order_embeddings)]
        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * \
            fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [
            item*item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(
            fm_second_order_emb_square)  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square -
                           fm_second_order_emb_square_sum) * 0.5
        """
            deep part
        """
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)
        """
            sum
        """
        total_sum = torch.sum(fm_first_order, 1) + \
                    torch.sum(fm_second_order, 1) + torch.sum(deep_out, 1) + self.bias
        return total_sum

class Linearmodule(torch.nn.Module):

    def __init__(self, feature_num):
        super().__init__()
        self.linear = nn.Linear(feature_num,1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x)
        #x=F.softmax(x,dim=1)
        return x

class MLPmodule(torch.nn.Module):

    def __init__(self, feature_num):
        super().__init__()
        self.linear0 = nn.Linear(feature_num,32)
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
    def __init__(self, feature_num, embed_dim, lr, n_classes, results_dir, model="FM", n_tags=57, tags_begin=39, args=args):
        self.epoch_num = args.epochs
        self.model=model
        if model=="FM":
            self.module=FMmodule(feature_num,embed_dim).to(device)
            self.criterion = nn.MSELoss(reduction="sum")
            self.optimizer = optim.Adam(self.module.parameters(), lr=lr,weight_decay=0)
        elif model=="DeepFM":
            self.feature_sizes=[32,3,1,1,1,1,57,3]
            self.module=DeepFMmodule(feature_sizes=self.feature_sizes,embedding_size=embed_dim).to(device)
            self.criterion = nn.MSELoss(reduction="sum")
            self.optimizer = optim.Adam(self.module.parameters(), lr=lr,weight_decay=0)
        elif model=="Linear":
            self.module=Linearmodule(feature_num).to(device)
            self.criterion = nn.MSELoss(reduction="sum")
            self.optimizer = optim.Adam(self.module.parameters(), lr=lr,weight_decay=0)
        elif model=="MLP":
            self.module=MLPmodule(feature_num).to(device)
            self.criterion = nn.MSELoss(reduction="sum")
            self.optimizer = optim.Adam(self.module.parameters(), lr=lr,weight_decay=0)
        elif model=="SVR":
            self.module=SVR(max_iter=1000)
        elif model=="tag_mean":
            self.module=tagmean_module(n_tags,tags_begin)
        self.results_dir=os.path.join(results_dir,str(feature_num)+"_"+str(embed_dim)+"_"+str(lr)+"_"+model)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.n_classes=n_classes
        self.logfile=str()
        self.logfile=str(args)+"\n"+"field_dim:"+str(feature_num)+", embed_dim:"+str(embed_dim)+", n_classes:"+str(n_classes)+", lr:"+str(lr)+", model:"+str(model)+"\n"

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


    def train_DeepFM(self, loader_train, loader_val):
        """
        Training a model and valid accuracy.
        Inputs:
        - loader_train: I
        - loader_val: .
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - verbose: Bool, if print.
        - print_every: Integer, print after every number of iterations. 
        """
        """
            load input data
        """
        feature_sizes=self.feature_sizes
        model = self.module.train().to(device=device)
        criterion = F.binary_cross_entropy_with_logits
        optimizer=self.optimizer
        epochs=self.epoch_num
        print_every=args.log_interval
        field_size=len(feature_sizes)
        loss_curve=[]
        test_loss_curve=[]
        acc_curve=[]
        test_acc_curve=[]
        for epoch in range(epochs):
            train_loss=0
            acc=0
            for t, (x, y) in enumerate(loader_train):
                xi = torch.zeros((len(x),field_size,1))
                xv = torch.zeros((len(x),field_size,1))
                for i, feature_size in enumerate(feature_sizes):
                    begin=sum(feature_sizes[:i])
                    xi[:,i,0] = torch.argmax(x[:,begin:begin+feature_size],dim=1)
                    xv[:,i,0] = torch.max(x[:,begin:begin+feature_size],dim=1)[0]
                xi = xi.to(device=device, dtype=torch.long)
                xv = xv.to(device=device, dtype=torch.float)
                y = y.to(device=device, dtype=torch.float)
                
                total = model(xi, xv)
                loss = criterion(total, y)
                ac = self.count_same(y.view(len(y),-1).detach(),total.view(len(total),-1).detach())
                train_loss += loss.item()
                acc += ac
                loss_curve.append(loss.item() / len(xi))
                acc_curve.append(ac/len(xi))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                    self.test_DeepFM(loader_val, feature_sizes)
                    print()
                if t % print_every == 0:
                    log='Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.3f}'.format(
                        epoch, t * len(xi), len(loader_train.dataset),
                        100. * t * len(xi)/ len(loader_train.dataset),
                        loss.item() / len(x),
                        ac/len(x))
                    print(log)
                    self.logfile+=log+"\n"
            log='====> Train Epoch: {} Average loss: {:.4f} accuracy: {:.3f}'.format(
                epoch, train_loss / len(loader_train.dataset),acc/len(loader_train.dataset))
            print(log)
            self.logfile+=log+"\n"
            tloss,tacc=self.test_DeepFM(loader_val,feature_sizes)
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
    
    def test_DeepFM(self, loader, feature_sizes):
        field_size=len(feature_sizes)
        model = self.module.to(device)
        eval_loss=0
        acc=0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for t, (x, y) in enumerate(loader):
                xi = torch.zeros((len(x),field_size,1))
                xv = torch.zeros((len(x),field_size,1))
                for i, feature_size in enumerate(feature_sizes):
                    begin=sum(feature_sizes[:i])
                    xi[:,i,0] = torch.argmax(x[:,begin:begin+feature_size],dim=1)
                    xv[:,i,0] = torch.max(x[:,begin:begin+feature_size],dim=1)[0]
                xi = xi.to(device=device, dtype=torch.long)
                xv = xv.to(device=device, dtype=torch.float)
                y = y.to(device=device, dtype=torch.float)
                total = model(xi, xv)
                acc+=self.count_same(y.view(len(y),-1),total.view(len(y),-1))
                loss = self.criterion(total,y.view(len(x),1))
                eval_loss += loss.item()
        log='====> Test Average loss: {:.4f} accuracy:{:.3f}'.format(eval_loss / len(loader.dataset),acc / len(loader.dataset))
        print(log)
        self.logfile+=log+"\n"
        return eval_loss / len(loader.dataset),acc / len(loader.dataset)

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