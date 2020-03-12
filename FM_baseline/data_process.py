import pandas as pd
import pickle
import torch
import numpy as np

filename="../data/dat.csv"
savefeaturename="../data/feature.pkl"
savelabelname="../data/label.pkl"
savedictname="../data/user_cdn_dics.pkl"

df=pd.read_csv(filename,index_col=0)
"""cdn_dict={}
user_dict={}
num_sessions=df.shape[0]
for index, row in df.iterrows():
    cdn=row["audience_stat_live_stream_ip"]
    user=row["did"]
    if cdn not in cdn_dict.keys():
        cdnn=len(cdn_dict)
        cdn_dict[cdn]=cdnn
        #df.iloc[index,0]=cdnn
    else:
        #df.iloc[index,0]=cdn_dict[cdn]
        pass
    if user not in user_dict.keys():
        usern=len(user_dict)
        user_dict[user]=usern
        #df.iloc[index,1]=usern
    else:
        #df.iloc[index,1]=user_dict[user]
        pass
    if index%10000==0:
        print(index,"/",num_sessions)

with open(savedictname,"wb") as f:
    pickle.dump([user_dict,cdn_dict],f)"""

with open(savedictname,"rb") as f:
    dictset=pickle.load(f)
    cdn_dict=dictset[1]
    user_dict=dictset[0]

for k in list(cdn_dict.keys()):
    if not k==k:
        del cdn_dict[k]

for k in list(user_dict.keys()):
    if not k==k:
        del user_dict[k]

df.dropna(axis=0,how='any',inplace=True)
df["audience_stat_live_stream_ip"]=df["audience_stat_live_stream_ip"].apply(lambda x:cdn_dict[x])
df["did"]=df["did"].apply(lambda x:user_dict[x])

num_sessions=df.shape[0]
num_users=len(user_dict)
num_cdns=len(cdn_dict)
indextuples=[]
session_times=torch.from_numpy(df.iloc[:,2].values)

index=0
for k,row in df.iterrows():#这个地方改为用apply
    cdn=int(row["audience_stat_live_stream_ip"])
    user=int(row["did"])
    indextuples.append([index,cdn])
    indextuples.append([index,user+num_cdns])
    index+=1

i = torch.LongTensor(indextuples)
v = torch.ones(2*num_sessions)
pre_sparse=[i.t(), v, torch.Size([num_sessions,num_cdns+num_users])]

with open(savefeaturename,"wb") as f:
    pickle.dump(pre_sparse,f)
with open(savelabelname,"wb") as f:
    pickle.dump(session_times,f)
    