import sys 
sys.path.append('.')
import pandas as pd 
from utils import generateInput
class Pileline:
    def __init__(self,dataset,ratio):
        self.ratio=ratio
        self.dataset=dataset
    
    def generateInputDataset(self):
        dataset=pd.read_csv(self.dataset)
        dataset_num=len(dataset)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_num=int(ratios[0]/sum(ratios)*dataset_num)

        train_dataset=dataset.iloc[:train_num]
        test_dataset=dataset.iloc[train_num:]

        generateInput(train_dataset,"dataset/train.pkl","train",4)
        generateInput(test_dataset,"dataset/test.pkl","test",4)


if __name__ == "__main__":
    ratio='8:2'
    dataset='dataset/20180103.csv'
    ppl=Pileline(dataset,ratio)
    ppl.generateInputDataset()
        



