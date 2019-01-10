import matplotlib.pyplot as plt 
import pandas as pd

class Plot:
    def __init__(self, filename):
        df=pd.read_csv(filename)
        self.id = df['id'].values.tolist()
        self.allApp = df['appearance'].values.tolist()
        self.isEnding = df['isEnding'].values.tolist()

        self.ending = []
        self.noending = []
        self.endingID = []
        self.noendingID = []
        end_i = 0
        noend_j = 0
        for x in range(0,len(self.allApp)):
            if self.isEnding[x] == 1 and self.allApp[x] > 4:
                self.ending.append(self.allApp[x])
                self.endingID.append(end_i)
                end_i += 1
            elif self.isEnding[x] == 0 and self.allApp[x] > 4:
                self.noending.append(self.allApp[x])
                self.noendingID.append(noend_j)
                noend_j += 1
    
    def plotDatas(self):
        xlst1 = []

        for i in range(0,50):
            xlst1.append(i)
        
        
        
        plt.bar(self.endingID, self.ending, alpha = 0.5, label = "Befejezes", color='c')
        plt.bar(self.noendingID, self.noending, alpha = 0.5, label = "Nem befejezes", color='r')
        
        plt.xlabel('id')
        plt.ylabel('Validacios mondatok top100 trainsethez mert hasonlosaga (n>4)')
        plt.title('Mondatok hasonlosagi halmazban mert gyakorisaga\nBag of Words')
        plt.legend()
        plt.show()
        


obj = Plot('appearances_bow.csv')
obj.plotDatas()