import matplotlib.pyplot as plt
import numpy as np
import math
import os
from sklearn.cluster import KMeans
from scipy.fft import fft, fftfreq
import dwdatareader as dw


class VibCharts:
    def __init__(self,
            coleta = 14,
            test = 5,
            Sensores = [1],
            dxdPart = [0],
            frequence = 100000,
            path = "/media/eduardo/HDEduardoAnacleto/DADOS/Coleta"
            ):

        self.test = test
        self.coleta = coleta
        self.dxdPart = dxdPart
        self.Sensores = Sensores
        self.frequence = frequence
        self.path = path
        self.const_g = 0.109172
        self.unit = 'a'

        self.Colors = ['-b', '-r', '-g']
        self.Legend = [['Sensor 1'], ['Sensor 2'], ['Sensor 3']]
        self.yLabel = dict()
        self.yLabel['a'] = "Amplitude (m/s^2)"
        self.yLabel['u'] = "Amplitude (g)"
        self.my_dpi = 100
        self.altura = 5.5
        self.largura = 20

        
        self.Vibrations = [[],[],[]]
        self.numVibrations = 0
        self.Time = []
        self.duration = 0

        self.parts = [[],[],[]]
        self.numParts = [0,0,0]
        self.numPassadas = [0,0,0]
        self.durationPassadas = [0,0,0]

        self.vib_rms = [0,0,0]
        self.vib_rms_pass = [0,0,0]
        self.vib_rms_movi = [0,0,0]

        #dispersion measures - vibrations
        self.vib_min = [0,0,0]
        self.vib_lower = [0,0,0]
        self.vib_upper = [0,0,0]
        self.vib_max = [0,0,0]

        #dispersion measures - dft
        self.dft_min = [0,0,0]
        self.dft_lower = [0,0,0]
        self.dft_upper = [0,0,0]
        self.dft_max = [0,0,0]


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


    def dispersion_measures( self, lista ):
        y = lista[:]
        y_size = len(y)
        y.sort()
        jump = []
        for i in range(1, y_size):
            if y[i] - y[i-1] > 1:
                jump.append(y[i] - y[i-1])
        lower_outlier = []
        upper_outlier = []

        if jump != []:
            s_avg = np.average(jump)
            s_std = np.std(jump)
            #print(s_std)
            #print(jump)
            
            for i in range(y_size//2, 0, -1):
                if y[i] - y[i-1] > s_avg + 3 * s_std:
                    lower_outlier.append(i-1)
                    break
            for i in range( y_size//2, y_size):
                if y[i] - y[i-1] > s_avg + 3 * s_std:
                    upper_outlier.append(i)
                    break

        if lower_outlier == []:
            lower_outlier.append(0)
        if upper_outlier == []:
            upper_outlier.append( y_size-1 )
        
        #print("outliers:", lower_outlier, upper_outlier)
        return y[0], y[lower_outlier[0]], y[upper_outlier[0]], y[-1]


    def dft_dispersion_measures(self, y):
        N, F, yf = self.DFT_normalized(y)
        return self.dispersion_measures(yf)

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

    def get_rmsParts(self, sensor, y):
        rms_movi = 0
        rms_pass = 0
        movi_size = 0
        pass_size = 0
        for i in range(self.numParts[sensor-1]):
            if self.parts[sensor-1][i][0] == 0:
                movi_size += self.parts[sensor-1][i][2] - self.parts[sensor-1][i][1]
                for j in range(self.parts[sensor-1][i][1], self.parts[sensor-1][i][2]):
                    rms_movi += y[j]**2
            else:
                pass_size += self.parts[sensor-1][i][2] - self.parts[sensor-1][i][1]
                for j in range(self.parts[sensor-1][i][1], self.parts[sensor-1][i][2]):
                    rms_pass += y[j]**2

        self.durationPassadas[sensor-1] = pass_size / self.frequence
        
        if movi_size != 0:
            rms_movi = math.sqrt( rms_movi / movi_size )
        if pass_size != 0:
            rms_pass = math.sqrt( rms_pass / pass_size )
        return rms_movi, rms_pass


    def getFirstSensorIndex(self):
        return self.Sensores[0]-1


    def getVibrations(self):
        
        if self.coleta != 11 and self.coleta != 12:

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
            # time vector
            self.Time = np.linspace(0.0, self.duration, self.numVibrations, endpoint=False)
            # Duration in seconds
            self.duration = self.numVibrations / self.frequence

        else:

            for part in range(self.dxdPart[0], self.dxdPart[1]+1):
                with dw.open( self.path + str(self.coleta) + "/Dados/faceamento " + str(self.test) + "_{:04d}.dxd".format(part)) as f:
                    canais = []
                    canal = [[],[],[]]
                    for ch in f.values():
                        canais.append( ch.name )
                    
                    for sensor in self.Sensores:                        
                        canal[sensor-1] = f[canais[sensor-1]].series()
                        self.Vibrations[sensor-1] += list(canal[sensor-1].values)
                    self.Time += list(canal[0].index)

                # number of vibrations
                self.numVibrations = len(self.Vibrations[0])
                # Duration in seconds 
                self.duration = self.numVibrations / self.frequence



    def getParameters( self ):

        self.create_directory("Data")
        self.getVibrations()

        for sensor in self.Sensores:
            self.vib_rms[sensor-1] = self.get_rms(self.Vibrations[sensor-1])

            self.vib_min[sensor-1], \
            self.vib_lower[sensor-1], \
            self.vib_upper[sensor-1], \
            self.vib_max[sensor-1] = self.dispersion_measures(self.Vibrations[sensor-1])
            
            self.dft_min[sensor-1], \
            self.dft_lower[sensor-1], \
            self.dft_upper[sensor-1], \
            self.dft_max[sensor-1] = self.dft_dispersion_measures(self.Vibrations[sensor-1])

            self.vibrationParts(sensor)
            self.vib_rms_movi[sensor-1], self.vib_rms_pass[sensor-1] = self.get_rmsParts(sensor, self.Vibrations[sensor-1])


        with open(self.path + str(self.coleta) + "/Data/data_" + str(self.unit) + ".txt", "a") as f:
            for sensor in self.Sensores:
                f.write("{:2d};".format(self.coleta) +
                    " {:2d};".format(self.test) +
                    " {:1d};".format(sensor) +
                    " {:10d};".format(self.numVibrations) +
                    " {:10.5f};".format(self.duration) +
                    " {:10.5f};".format(self.vib_rms[sensor-1]) +
                    " {:4d};".format(self.numPassadas[sensor-1]) +
                    " {:10.5f};".format(self.durationPassadas[sensor-1]) +
                    " {:10.5f};".format(self.vib_rms_pass[sensor-1]) +
                    " {:4d};".format(self.numParts[sensor-1] - self.numPassadas[sensor-1]) +
                    " {:10.5f};".format(self.duration - self.durationPassadas[sensor-1]) +
                    " {:10.5f};".format(self.vib_rms_movi[sensor-1]) +
                    " {:10.5f};".format(self.vib_min[sensor-1]) +
                    " {:10.5f};".format(self.vib_lower[sensor-1]) +
                    " {:10.5f};".format(self.vib_upper[sensor-1]) +
                    " {:10.5f};".format(self.vib_max[sensor-1]) +
                    " {:10.5f};".format(self.dft_min[sensor-1]) +
                    " {:10.5f};".format(self.dft_lower[sensor-1]) +
                    " {:10.5f};".format(self.dft_upper[sensor-1]) +
                    " {:10.5f};".format(self.dft_max[sensor-1]) +
                    "\n")


    def plotHistogram(self):
        vib = [abs(v) for v in self.Vibrations[0]]

        n, bins, patches = plt.hist( vib, 1000, density = False, facecolor='g' )
        #plt.xlabel( 'Smarts' )
        #plt.ylabel( 'Probability' )
        plt.title( 'Histogram' )
        #plt.text( 60, .025, r'$\mu=100,\ \sigma=15$' )
        #plt.xlim( 40, 160 )
        #plt.ylim( 0, 0.03 )
        plt.grid( True )
        plt.show( )

    
    def plotVibrations( self, sensor ):

        self.create_directory("Charts")

        if self.numParts[sensor-1] == 0 :
            
            plt.rc('font', **{'size' : 18})
            plt.ticklabel_format(style = 'plain')
            plt.figure( figsize= (self.largura, self.altura), dpi= self.my_dpi )
            title = "Gráfico no domínio do tempo do teste " + str(self.test) + " da coleta " + str(self.coleta)

            if self.coleta != 11 and self.coleta != 12:
                plt.title( title )
            else:
                if self.dxdPart[0] == self.dxdPart[1]:
                    plt.title( title + " (arquivo " + str(self.dxdPart[0]) + ")" )
                else:
                    plt.title( title + " (arquivos " + str(self.dxdPart) + ")" )


            if self.unit == 'a':
                plt.plot( self.Time, self.Vibrations[sensor-1], self.Colors[sensor-1] )
            elif self.unit == 'g':
                plt.plot( self.Time, convert_to_g(self.Vibrations[sensor-1]), self.Colors[sensor-1] )

            plt.ylabel( self.yLabel[self.unit] )
            plt.legend( self.Legend[sensor-1] )
            plt.xlabel( "Tempo (s)" )
            plt.grid( linestyle='--', axis='y' )

            figpath = self.path + str(self.coleta) + '/Charts/VibT' + str(self.test) + 'S' + str(sensor)
            if self.coleta != 11 and self.coleta != 12:
                plt.savefig( figpath + self.unit + '.png' )
            else:
                plt.savefig( figpath + 'P' + str(self.dxdPart) + self.unit + '.png')

            plt.close()
            plt.cla()
            plt.clf()


        else:

            for i in range(self.numParts[sensor-1]):
                if self.unit == 'a':
                    plt.plot( self.Time[self.parts[sensor-1][i][1]:self.parts[sensor-1][i][2]], 
                        self.Vibrations[0][self.parts[sensor-1][i][1]:self.parts[sensor-1][i][2]], 'b')
                    plt.ylabel( "Amplitude (m/s^2)" )

                elif self.unit == 'g':
                    plt.plot( self.Time[self.parts[sensor-1][i][1]:self.parts[sensor-1][i][2]], convert_to_g(self.Vibrations[0][self.parts[sensor-1][i][1]:self.parts[sensor-1][i][2]]), 'b' )
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


    def vibrationParts(self, sensor, window = 2500, stitcher_coef = 20):
        x_window = [[abs(v) for v in self.Vibrations[sensor-1][i:i+window]] for i in range(0, self.numVibrations, window)]
        X = np.array([[np.average(v)] for v in x_window])
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        labels_ = kmeans.labels_.tolist()
        self.stitcher(labels_, stitcher_coef)
        parts = self.groupWindows(labels_)
        self.numParts[sensor-1] = len(parts)
        #print('number of parts:', self.numParts[sensor-1])

        #print(kmeans.cluster_centers_[0], kmeans.cluster_centers_[1])
        if kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]:
            for i in range(self.numParts[sensor-1]):
                parts[i][0] = 1 - parts[i][0]

        parts[0].append(0)
        #print()
        parts[0][1], parts[0][2] = parts[0][2], parts[0][1] * window
        #print(parts[0])
        for i in range(1, self.numParts[sensor-1]):
            parts[i].append(parts[i-1][2])
            parts[i][1] *= window
            parts[i][1], parts[i][2] = parts[i][2], parts[i][2] + parts[i][1] 
            #print(parts[i])

        parts[self.numParts[sensor-1]-1][2] = self.numVibrations
        #print('\n', parts[self.numParts[sensor-1]-1])
        self.parts[sensor-1] = parts

        for i in range(self.numParts[sensor-1]):
            self.numPassadas[sensor-1] += self.parts[sensor-1][i][0]




if __name__ == '__main__':

    #Data
    for coleta in [12]:
        for test in [1]:
            for arq in range(78):
                v = VibCharts(coleta, test, [1,2], [arq,arq])
                v.getParameters()
                print("Coleta", coleta, "Test", test, "Arq", arq)
    #for coleta in [14]:
    #    for test in [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]:
    #        v = VibCharts(coleta, test, [1, 2, 3])
    #        v.getParameters()
    #        print("Coleta", coleta, "Test", test)
    #for coleta in [13]:
    #    for test in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
    #        v = VibCharts(coleta, test, [1, 2, 3])
    #        v.getParameters()
    #        print("Coleta", coleta, "Test", test)

    Sensores = [1, 2]

    v = VibCharts(12, 1, Sensores, [40, 40])
    v.getVibrations()
    for sensor in Sensores:
        v.plotVibrations(sensor)
    #v.vibrationParts()
    #v.plotVibrations()
