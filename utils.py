import pandas as pd
import numpy as np
import logging
import sys
import torch
from config import batch_size,gap,feature_dim
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
# 根据原始数据转化成输入到深度模型中的数据
# dataset:原始数据集
# dump_path:处理后的数据集的存储位置
# part标明是训练集还是测试集
# gap:采样间隔，默认为４

def generateInput(dataset,dump_path,part, gap):
    features, labels = [], []
    cnt = 0
    features=[]
    labels=[]
    feature=[]

    for i in range(len(dataset)-gap-1):
        tmp=dataset.iloc[i:i+gap]
        for _,item in tmp.iterrows():
            feature += list(item[2:])  # 剔除时间信息
        features.append(feature[:])
        labels.append(dataset.iloc[i+gap][3])
        feature.clear() 
        
    df=pd.DataFrame()
    df['feature']=features
    df['label']=labels
    df.to_pickle(dump_path)

# 创建训练日志
def buildLogger(log_file, part):
    logger = logging.getLogger(part)
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    # FileHandler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def getBatch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    features, labels = [], []
    for _, item in tmp.iterrows():
        features.append(item[0])
        labels.append(item[1])
    features=torch.from_numpy(np.asarray(features,dtype='float32'))
    labels=torch.FloatTensor(labels)
    features=features.view(batch_size,gap,feature_dim)
    return features,labels

# 计算度量指标
def calMetric(predict,ground_truth):
    print(f'rmse:{np.sqrt(mean_squared_error(y_pred=predict,y_true=ground_truth))}')
    print(f'mse:{mean_squared_error(y_pred=predict,y_true=ground_truth)}')
    print(f'mae:{mean_absolute_error(y_pred=predict,y_true=ground_truth)}')

# 绘制折线图
def drawPicture(predict,ground_truth):
    predict_x=list(range(len(predict)))
    ground_truth_x=list(range(len(ground_truth)))

    plt.plot(predict_x,predict)
    plt.plot(ground_truth_x,ground_truth)

    plt.show()

    

