import pandas as pd
import pickle
import torch
import numpy as np
import scipy.sparse
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("use_devicem", default=True, help="use the device model or not", type=bool)
args = parser.parse_args()

filename="../data/live_tagged_data.txt/live_tagged_data.txt"
savefeaturename="../data/live_tagged_data.txt/feature.pkl"
savefmfeaturename="../data/live_tagged_data.txt/whole_fmfeature.pkl"
savelabelname="../data/live_tagged_data.txt/label.pkl"
savedictname="../data/live_tagged_data.txt/user_cdn_dics.pkl"

class Featureprocessor:
    def __init__(self,num_data,feature_len):
        self.cdn_dict={}
        self.device_dict={}
        self.devicemodel_dict={}
        self.tag_dict={}
        self.isp_dict={}
        self.featurearray=np.zeros((num_data,feature_len))
        self.labelarray=np.zeros(num_data)
        self.arrayindex=0

    def findvalue(self,dic,key):
        leng=len(dic)
        if key not in dic.keys():
            dic[key]=leng
        return dic[key]
            
    def process_line(self,line):
        linelist=line.split(",")
        feature=np.zeros(9)
        cdn=linelist[2][:linelist[2].find(".")]
        if cdn.isdigit():
            return 0
        feature[0]=self.findvalue(self.cdn_dict,cdn)
        device=linelist[4]
        feature[1]=self.findvalue(self.device_dict,device)
        if device=="IPHONE" or device=="IPAD":
            devicemodel=linelist[5]
            feature[2]=self.findvalue(self.devicemodel_dict,devicemodel)
            QoS=linelist[8:11]+[linelist[12]]
            for i,qos in enumerate(QoS):
                feature[3+i]=qos
            duration=linelist[11]
            tag=linelist[14:-1].sort()
            feature[7]=self.findvalue(self.tag_dict,tag)
            isp=linelist[-1]
            feature[8]=self.findvalue(self.isp_dict,isp)
        else:
            devicemodel=linelist[5]
            feature[2]=self.findvalue(self.devicemodel_dict,devicemodel)
            QoS=linelist[7:10]+[linelist[11]]
            for i,qos in enumerate(QoS):
                feature[3+i]=qos
            duration=linelist[10]
            tag="".join(sorted(linelist[13:-1]))
            feature[7]=self.findvalue(self.tag_dict,tag)
            isp=linelist[-1]
            feature[8]=self.findvalue(self.isp_dict,isp)
        self.featurearray[self.arrayindex]=feature
        self.labelarray[self.arrayindex]=duration
        self.arrayindex+=1
        return 0
    
"""with open(filename,encoding="utf-8") as f:
    a=f.readlines()
processor=Featureprocessor(len(a),9)
for index,line in enumerate(a):
    if index%1000==0:
        print(index,"/",len(a))
    processor.process_line(line)
processor.featurearray=processor.featurearray[:processor.arrayindex]
processor.labelarray=processor.labelarray[:processor.arrayindex]
with open(savedictname,"wb") as f:
    pickle.dump([processor.cdn_dict,processor.device_dict,processor.devicemodel_dict,processor.tag_dict,processor.isp_dict],f)
#要处理NULL吗？？
with open(savefeaturename,"wb") as f:
    pickle.dump(processor.featurearray,f)
with open(savelabelname,"wb") as f:
    pickle.dump(processor.labelarray,f)"""
    
with open(savedictname,"rb") as f:
    dictset=pickle.load(f)
with open(savefeaturename,"rb") as f:
    featurearray=pickle.load(f)
with open(savelabelname,"rb") as f:
    labelarray=pickle.load(f)

cdnn=len(dictset[0])#32
devicen=len(dictset[1])#3
devicemodeln=len(dictset[2])#1593
tagn=len(dictset[3])#57, in feature vector 39:96(96 not included)
ispn=len(dictset[4])#3
sumn=cdnn+devicen+devicemodeln+tagn+ispn+4

if args.use_devicem==True:
    FMfeature=np.zeros((len(labelarray),sumn))
    for i in range(len(labelarray)):
        featurerow=np.zeros(sumn)
        row=featurearray[i]
        cdn=int(row[0])
        device=int(row[1])
        devicemodel=int(row[2])
        tag=int(row[7])
        isp=int(row[8])
        featurerow[cdn]=1
        featurerow[cdnn+device]=1
        featurerow[cdnn+devicen+devicemodel]=1
        for j in range(3,7):
            featurerow[cdnn+devicen+devicemodeln+j-3]=row[j]
        featurerow[cdnn+devicen+devicemodeln+4+tag]=1#
        featurerow[cdnn+devicen+devicemodeln+4+tagn+isp]=1
        FMfeature[i]=featurerow
        if i%1000==0:
            print(i,"/",len(labelarray))
else:
    sumn=sumn-devicemodeln
    FMfeature=np.zeros((len(labelarray),sumn))
    for i in range(len(labelarray)):
        featurerow=np.zeros(sumn)
        row=featurearray[i]
        cdn=int(row[0])
        device=int(row[1])
        tag=int(row[7])
        isp=int(row[8])
        featurerow[cdn]=1
        featurerow[cdnn+device]=1
        for j in range(3,7):
            featurerow[cdnn+devicen+j-3]=row[j]
        featurerow[cdnn+devicen+4+tag]=1
        featurerow[cdnn+devicen+4+tagn+isp]=1
        FMfeature[i]=featurerow
        if i%1000==0:
            print(i,"/",len(labelarray))


cscmatrix=scipy.sparse.csc_matrix(FMfeature)

with open(savefmfeaturename,"wb") as f:
    pickle.dump(cscmatrix,f)