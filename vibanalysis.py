import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

class VibCharts:
    def __init__(self, 
            coleta = 14, 
            test = 5, 
            Sensores = [1],
            frequence = 100000,
            path = "/media/eduardo/SARA/Cláudio - Projeto VW/Coleta"
            ):

        self.test = test
        self.coleta = coleta
        self.Sensores = Sensores
        self.frequence = frequence
        self.path = path
        self.const_g = 0.109172
        
        self.Vibrations = [[],[],[]]
        self.numVibrations = 0
        self.parts = []
        self.numParts = 0
        self.Time = []
        self.duration = 0


    def getFirstSensorIndex(self):
        return self.Sensores[0]-1

    def getVibrations(self):
        # Lear dados
        text_file = open(self.path + str(self.coleta) + "/teste_" + str(self.test) + ".csv", "r" )
        #skip the first two lines
        if self.coleta != 12 or self.coleta != 11:
            next( text_file )
            next( text_file )

        self.numVibrations = 0;
        for line in text_file:
            row_text = line.split(";")
            for s in self.Sensores:
                if self.coleta != 12 or self.coleta != 11:
                    self.Vibrations[s-1].append( float(row_text[s]))
                else:
                    self.Vibrations[s-1].append( float(row_text[s]) / self.const_g)
            self.numVibrations += 1
        text_file.close()

        # Duration in seconds 
        self.duration = self.numVibrations / self.frequence

        # time
        self.Time = np.linspace(0.0, self.duration, self.numVibrations, endpoint=False)


    def plot_histogram(self):
        
        vib = [abs(v) for v in self.Vibrations[0]]

        n, bins, patches = plt.hist(vib, 1000, density=False, facecolor='g')
        #plt.xlabel('Smarts')
        #plt.ylabel('Probability')
        plt.title('Histogram')
        #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        #plt.xlim(40, 160)
        #plt.ylim(0, 0.03)
        plt.grid(True)
        plt.show()

    def count_parts(self, clusters):
        
        count = 0
        flip = clusters[0]
        for v in clusters[1:]:
            if v != flip:
                count += 1
                flip = v
        return count

    def stitcher(self, lista, k):
        
        flip = lista[0]
        count_k = [0, 0]
        count_k[flip] = 1
        count_k[1-flip] = 0
        for i in range(1,len(lista)):
            if lista[i] == flip:
                count_k[flip] += 1
            else:
                if count_k[flip] <= k:
                    flip = 1 - flip
                    count_k[flip] += count_k[1-flip] + 1
                    for j in range(i-1, i-count_k[1-flip]-1, -1):
                        lista[j] = flip
                else:
                    flip = 1 - flip
                    count_k[flip] = 1
    

    def groupWindows(self, lista):

        parts = [[lista[0], 1]]
        position = 0
        for v in lista[1:]:
            if v == parts[position][0]:
                parts[position][1] += 1
            else:
                parts.append([v, 1])
                position += 1

        return parts


    def vibrationParts(self, window = 2500, stitcher_coef = 20):
        
        index = self.getFirstSensorIndex()

        x_window = [[abs(v) for v in self.Vibrations[index][i:i+window]] for i in range(0, self.numVibrations, window)]
        X = np.array([[np.average(v)] for v in x_window])
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        labels_ = kmeans.labels_.tolist()
        self.stitcher(labels_, stitcher_coef)
        parts = self.groupWindows(labels_)
        num_parts = len(parts)

        print(kmeans.cluster_centers_[0], kmeans.cluster_centers_[1])
        if kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]:
            for i in range(num_parts):
                parts[i][0] = 1 - parts[i][0]

        for i in range(num_parts):
            parts[i][1] *= window
            print(parts[i])

        print('number of parts:', num_parts)
        self.parts = parts
        self.numParts = num_parts

        


v = VibCharts(10, 19, [2])
v.getVibrations()
v.vibrationParts()
