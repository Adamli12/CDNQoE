import numpy as np
import torch
from model import FactorizationMachine
import os
import torch.utils.data as Data
import pickle
from sklearn.model_selection import train_test_split
import scipy.sparse

BATCH_SIZE = 16
LEARNING_RATE = 1e-5

featurename="../data/live_tagged_data.txt/fmfeature.pkl"
labelname="../data/live_tagged_data.txt/label.pkl"
results_dir="../data/live_tagged_data.txt/results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with open(labelname,"rb") as labelfile:
    label = pickle.load(labelfile)
with open(featurename,"rb") as featurefile:
    feature = pickle.load(featurefile)
    feature=feature.toarray()

label=label/np.percentile(label,80)
label=label.clip(0,1)

feature=torch.from_numpy(feature)
label=torch.from_numpy(label)

dataset = Data.TensorDataset(feature,label)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,test_size])

train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

fm10=FactorizationMachine(feature.shape[1],10,LEARNING_RATE,results_dir)
fm10.train(train_loader,test_loader)
fm10.save()

fm20=FactorizationMachine(feature.shape[1],20,LEARNING_RATE,results_dir)
fm20.train(train_loader,test_loader)
fm20.save()

fm64=FactorizationMachine(feature.shape[1],64,LEARNING_RATE,results_dir)
fm64.train(train_loader,test_loader)
fm64.save()