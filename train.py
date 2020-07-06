import pandas as pd
import random
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys
from config import batch_size, hidden_dim, feature_dim, use_gpu, epochs,gap
from model import rnnModel
from tqdm import tqdm
from utils import buildLogger,getBatch



if __name__ == '__main__':
    train_dataset = pd.read_pickle('dataset/train.pkl')
    model = rnnModel(hidden_dim, feature_dim, batch_size)
    logger = buildLogger("log/train.log", "train")

    if use_gpu:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.MSELoss()
    total_len = len(train_dataset)
    print("start training...")

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0

        for i in tqdm(range(0, len(train_dataset)-batch_size,batch_size)):
            features, labels = getBatch(train_dataset, i, batch_size)

            if use_gpu:
                features, labels = features.cuda(), labels.cuda()

            model.zero_grad()

            output = model(features)

            loss = loss_function(output, Variable(labels))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(features)

        logger.info(f'epoch:{epoch},train_loss:{total_loss/total_len}')

    torch.save(model.state_dict(), f"log/birnn.pt")
    

