import matplotlib.pyplot as plt
import numpy as np
import math
import os
from sklearn.cluster import KMeans
from scipy.fft import fft, fftfreq


class VibCharts:
    def __init__(self,
            coleta = 14,
            test = 5,
            Sensores = [1],
            frequence = 100000,
            path = "/media/eduardo/HDEduardoAnacleto/DADOS/Coleta"
            ):

        self.test = test
        self.coleta = coleta
        self.Sensores = Sensores
        self.frequence = frequence
        self.path = path
        self.const_g = 0.109172
        self.unit = 'a'
        
        self.Vibrations = [[],[],[]]
        self.numVibrations = 0
        self.parts = []
        self.numParts = 0
        self.Time = []
        self.duration = 0

        self.y_rms = [0,0,0]
        self.y_std = [0,0,0]
        self.y_min_vib = [0,0,0]
        self.y_max_vib = [0,0,0]
        self.y_max_dft = [0,0,0]


    def create_directory(self, folder_name):
        isExist = os.path.exists(self.path + str(self.coleta) + "/" + folder_name)
        if not isExist:
            os.makedirs(self.path + str(self.coleta) + "/" + folder_name)

    def DFT(self, y):
        Y = np.abs(fft(y))
        N = len(Y)
        freq = fftfreq(N, 1.0 / self.frequence)
        N = N//2
        return F[:N], Y[:N]

    def DFT_normalized(self, y):
        Y = np.abs(fft(y))
        N = len(Y)
        F = fftfreq(N, 1.0 / self.frequence)
        F = F/1000.0
        N = N//2
        return N, F[:N], Y[:N]/N

    def get_dft_max(self, y):
        N, F, yf = self.DFT_normalized(y)
        max_dft = np.max(yf)
        return max_dft

    def convert_to_g(self, y):
        y_g = [v * self.const_g for v in y]
        return y_g

    def convert_to_a(self, y):
        y_a = [v / self.const_g for v in y]
        return y_a

    def get_rms(self, y):
        rms = 0
        for v in y:
            rms += v**2
        rms = math.sqrt(rms / len(y))
        return rms



    def getFirstSensorIndex(self):
        return self.Sensores[0]-1

    def getVibrations(self):
        # Read data
        text_file = open(self.path + str(self.coleta) + "/teste_" + str(self.test) + ".csv", "r" )
        #skip the first two lines
        if self.coleta != 12 or self.coleta != 11:
            next( text_file )
            next( text_file )

        self.numVibrations = 0;
        for line in text_file:
            row_text = line.split(";")
            for s in self.Sensores:
                self.Vibrations[s-1].append( float(row_text[s]))

            self.numVibrations += 1
        text_file.close()

        # Duration in seconds 
        self.duration = self.numVibrations / self.frequence

        # time vector
        self.Time = np.linspace(0.0, self.duration, self.numVibrations, endpoint=False)


    def getParameters(self):

        self.create_directory("Tables")
        self.getVibrations()

        for sensor in self.Sensores:
            self.y_rms[sensor-1] = self.get_rms(self.Vibrations[sensor-1])
            self.y_std[sensor-1] = np.std(self.Vibrations[sensor-1])
            self.y_min_vib[sensor-1] = np.min(self.Vibrations[sensor-1])
            self.y_max_vib[sensor-1] = np.max(self.Vibrations[sensor-1])
            self.y_max_dft[sensor-1] = self.get_dft_max(self.Vibrations[sensor-1])


        with open(self.path + str(self.coleta) + "/Tables/info_" + str(self.unit) + "_coleta" + ".txt", "a") as f:
            for sensor in self.Sensores:
                f.write("{:2d}".format(self.coleta) +
                        " {:2d};".format(self.test) +
                        " {:1d};".format(sensor) +
                        " {:10d};".format(self.numVibrations) +
                        " {:10.8f};".format(self.duration) +
                        " {:10.8f};".format(self.y_rms[sensor-1]) +
                        " {:10.8f};".format(self.y_std[sensor-1]) +
                        " {:10.8f};".format(self.y_min_vib[sensor-1]) +
                        " {:10.8f};".format(self.y_max_vib[sensor-1]) +
                        " {:10.8f};".format(self.y_max_dft[sensor-1]) +
                        "\n")


    def plot_histogram(self):

        vib = [abs(v) for v in self.Vibrations[0]]

        n, bins, patches = plt.hist(vib, 1000, density = False, facecolor='g')
        #plt.xlabel('Smarts')
        #plt.ylabel('Probability')
        plt.title('Histogram')
        #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        #plt.xlim(40, 160)
        #plt.ylim(0, 0.03)
        plt.grid( True )
        plt.show()

    
    def plotVibration( self ):
        
        if self.numParts == 0:
            if self.unit == 'a':
                plt.plot( self.Time, self.Vibrations[0], 'b' )
                plt.ylabel( "Amplitude (m/s^2)" )

            elif self.unit == 'g':
                plt.plot( self.Time, convert_to_g(self.Vibrations[0]), 'b' )
                plt.ylabel( "Amplitude (g)" )

            plt.xlabel( "Tempo (s)" )
            plt.grid( linestyle='--', axis='y' )
            plt.show()
            plt.close()
            plt.cla()
            plt.clf()

        else:

            for i in range(self.numParts):
                if self.unit == 'a':
                    plt.plot( self.Time[self.parts[i][1]:self.parts[i][2]], 
                        self.Vibrations[0][self.parts[i][1]:self.parts[i][2]], 'b' )
                    plt.ylabel( "Amplitude (m/s^2)" )

                elif self.unit == 'g':
                    plt.plot( self.Time[self.parts[i][1]:self.parts[i][2]], convert_to_g(self.Vibrations[0][self.parts[i][1]:self.parts[i][2]]), 'b' )
                    plt.ylabel( "Amplitude (g)" )

                plt.xlabel( "Tempo (s)" )
                plt.grid( linestyle='--', axis='y' )
                plt.show()
                plt.close()
                plt.cla()
                plt.clf()






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
        self.numParts = len(parts)
        print('number of parts:', self.numParts)

        print(kmeans.cluster_centers_[0], kmeans.cluster_centers_[1])
        if kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]:
            for i in range(self.numParts):
                parts[i][0] = 1 - parts[i][0]

        parts[0].append(0)
        print()
        parts[0][1], parts[0][2] = parts[0][2], parts[0][1] * window
        print(parts[0])
        for i in range(1, self.numParts):
            parts[i].append(parts[i-1][2])
            parts[i][1] *= window
            parts[i][1], parts[i][2] = parts[i][2], parts[i][2] + parts[i][1] 
            print(parts[i])

        self.parts = parts


        


if __name__ == '__main__':

    v = VibCharts(14, 1, [1])
    v.getParameters()
    #v.getVibrations()
    #v.plotVibration()
    #v.vibrationParts()
    #v.plotVibration()
