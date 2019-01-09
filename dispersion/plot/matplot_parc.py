import matplotlib.pyplot as plt 
import pandas as pd

class Plot:
    def __init__(self, filename):
        df=pd.read_csv(filename)
        self.id = df['id'].values.tolist()
        self.ending = df['doc2vec_right'].values.tolist()
        self.noending = df['bow_right'].values.tolist()
    
    def plotDatas(self):
        xlst1 = []

        for i in range(0,50):
            xlst1.append(i)
        
        
        
        plt.bar(xlst1, self.ending, alpha = 0.5, label = "Doc2Vec", color='c')
        plt.bar(xlst1, self.noending, alpha = 0.5, label = "BoW", color='r')
        
        plt.xlabel('id')
        plt.ylabel('top500 legnagyobb cos hasonlosagu befejezesek osszege')
        plt.title('50 Doc2Vec és BoW befejezes')
        plt.legend()
        plt.show()
        


obj = Plot('bow_doc2vec_right_merge.csv')
obj.plotDatas()