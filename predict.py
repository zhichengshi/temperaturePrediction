import pandas as pd 
from model import rnnModel
from config import batch_size, hidden_dim, feature_dim, use_gpu, epochs,gap
from tqdm import tqdm
from utils import getBatch,calMetric,drawPicture
import torch


if __name__ == "__main__":
    test_dataset=pd.read_pickle('dataset/test.pkl')
    model_path='log/birnn.pt'

    model=rnnModel(hidden_dim, feature_dim, batch_size)
    model.load_state_dict(torch.load(model_path))
    if use_gpu:
        model.cuda()
    
    predict=[]
    ground_truth=[]
    for i in tqdm(range(0, len(test_dataset)-batch_size,batch_size)):
        features, labels = getBatch(test_dataset, i, batch_size)

        if use_gpu:
            features, labels = features.cuda(), labels.cuda()

        model.zero_grad()

        output = model(features)

        predict.extend(output.data.cpu().numpy())
        ground_truth.extend(labels.data.cpu().numpy())

    df=pd.DataFrame()
    df['predict']=predict
    df['ground_truth']=ground_truth
    df.to_csv("log/result.csv")
    calMetric(predict,ground_truth)

    drawPicture(predict,ground_truth)

  

    
    


    