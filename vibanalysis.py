import matplotlib.pyplot as plt
import numpy as np
import math
import os
from sklearn.cluster import KMeans
from scipy.fft import fft, fftfreq
from scipy import signal
import dwdatareader as dw
import gc
from matplotlib import cm
import matplotlib.colors as colors
import sys

import csv
import json




class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        return np.ma.masked_array(np.interp(value, x, y,
                                            left=-np.inf, right=np.inf))
    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)



class VibAnalysis:


    def __init__(self,
            coleta = 14,
            test = 5,
            Sensores = [1],
            dxdPart = None, #preencher com [v] ou [v1,v2]
            Frequence = [7],
            skip = 0,
            path = "/media/eduardo/HDEduardoAnacleto/DADOS/Coleta"
            ):
        
        self.test = test
        if type(self.test) == list:
            if self.test[0] == self.test[1]:
                self.test = self.test[0]

        self.coleta = coleta
        self.dxdPart = dxdPart
        self.Sensores = Sensores
        self.Frequence = Frequence
        self.skip = skip


        self.path = path
        self.pathColeta = self.path + str(self.coleta)
        self.pathData = self.pathColeta + "/Data"
        self.pathCharts = self.pathColeta + "/Charts"
        self.pathChartsParts = self.pathColeta + "/ChartsParts"
        self.pathParameters = "Parameters/coleta" + str(self.coleta)


        self.Colors = ['cornflowerblue', 'lightcoral', 'yellowgreen']
        self.ColorsFace = [['cornflowerblue', 'royalblue'], ['lightcoral', 'indianred'], ['yellowgreen', 'olivedrab']]

        self.Legend = [['Sensor 1'], ['Sensor 2'], ['Sensor 3']]
        self.yLabel = dict()
        self.yLabel['a'] = "Aceleração (m/s^2)"
        self.yLabel['u'] = "Aceleração (g)"
        self.my_dpi = 100
        self.altura = 5.5
        self.largura = 20
        self.epsilon = 0.00001

        self.Vibrations = [[],[],[]]
        self.numVibrations = 0        
        self.Time = []
        self.activeVibrations = False
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
        self.dft_upper = [0,0,0]
        self.dft_max = [0,0,0]

        # Parameters
        self.parameters = dict()
        self.import_parameters()

        # adjust after import parameters
        self.parameters["sample_rate"] = self.parameters["sample_rate"] // (skip+1)



    def __del__(self):
        self.Colors.clear()
        self.Legend.clear()
        self.yLabel.clear()
        self.clearVibrations()        
        self.parts.clear()
        self.numParts.clear()
        self.numPassadas.clear()
        self.durationPassadas.clear()
        self.vib_rms.clear()
        self.vib_rms_pass.clear()
        self.vib_rms_movi.clear()
        self.vib_min.clear()
        self.vib_lower.clear()
        self.vib_upper.clear()
        self.vib_max.clear()
        self.dft_upper.clear()
        self.dft_max.clear()


    def import_parameters( self ):
        
        if self.dxdPart == None: 
            with open( self.pathParameters +  "_csv.json", "r") as f:
                self.parameters = json.load(f)
        else:
            with open( self.pathParameters +  "_dxd.json", "r") as f:
                self.parameters = json.load(f)


    def clearVibrations( self ):
        for sensor in self.Sensores:
            self.Vibrations[sensor-1].clear()
        self.Time.clear()
        self.numVibrations = 0
        self.activeVibrations = False



    def create_directory(self, folder_name):
        isExist = os.path.exists(self.path + str(self.coleta) + "/" + folder_name)
        if not isExist:
            os.makedirs(self.path + str(self.coleta) + "/" + folder_name)


    def DFT(self, y):
        Y = np.abs(fft(y))
        N = len(Y)
        freq = fftfreq(N, 1.0 / self.parameters["sample_rate"])
        N //= 2
        return F[:N], Y[:N]


    def DFTNormalized(self, y):
        Y = np.abs(fft(y))
        N = len(Y)
        F = fftfreq(N, 1.0 / self.parameters["sample_rate"])
        F /= 1000.0
        N //= 2
        return N, F[:N], Y[:N]/N


    def dispersion_measures( self, lista ):
        y = lista[:]
        #y_size = len(y)
        y.sort()
        #jump = []
        #for i in range(1, y_size):
        #    if y[i] - y[i-1] > 1:
        #        jump.append(y[i] - y[i-1])

        #lower_outlier = []
        #upper_outlier = []
        ##print("jump", jump)
        ##print("tail", y)

        #if jump != []:
        #    s_avg = np.average(jump)
        #    s_std = np.std(jump)
        #    #print('avg', s_avg)
        #    #print('std', s_std)
        #    #print(jump)
        #    n_jump = len(jump)
        #    for i in range(1, y_size//2):
        #        if y[i] - y[i-1] > s_avg + 10 * (s_std / np.sqrt(n_jump) ) + self.epsilon:
        #            lower_outlier.append(i-1)
        #            break
        #    for i in range(y_size-1, y_size//2, -1):
        #        if y[i] - y[i-1] > s_avg + 10 * ( s_std / np.sqrt(n_jump) ) + self.epsilon:
        #            upper_outlier.append(i)
        #            break

        #if lower_outlier == []:
        #    lower_outlier.append(0)
        #if upper_outlier == []:
        #    upper_outlier.append( y_size-1 )
        
        #print("outliers:", y[0], y[lower_outlier[0]], y[upper_outlier[0]], y[-1])
        #return y[0], y[lower_outlier[0]], y[upper_outlier[0]], y[-1]
        return y[0], y[0], y[-1], y[-1]


    def dft_dispersion_measures(self, y):
        N, F, yf = self.DFTNormalized(y)
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

        self.durationPassadas[sensor-1] = pass_size / self.parameters["sample_rate"]
        
        if movi_size != 0:
            rms_movi = math.sqrt( rms_movi / movi_size )
        if pass_size != 0:
            rms_pass = math.sqrt( rms_pass / pass_size )
        return rms_movi, rms_pass


    def getFirstSensorIndex(self):
        return self.Sensores[0]-1


    def determineVibRange(self, sensor, test, faceamento, passada, move_vazio = False, passada_por_faceamento = 6):
        
        position = []
        start_position = 0
        end_position = 0
        with open( self.pathData + "/VibPartsT" + str(test) + "S" + str(sensor) + ".txt", "r" ) as text_file:
            if passada == 0:
                first_passada = (faceamento-1) * passada_por_faceamento + 1
                last_passada = faceamento * passada_por_faceamento
                
                line = next(text_file)
                count_passada = 0
                for line in text_file:
                    row_text = line.split(';')
                    if move_vazio == False:
                        if int(row_text[0]) == 1:
                            count_passada += 1
                            if count_passada >= first_passada and count_passada <= last_passada:
                                position.append([int(row_text[1]), int(row_text[2])])
                    else:
                        if int(row_text[0]) == 1:
                            count_passada += 1
                            if count_passada == first_passada:
                                start_position = int(row_text[1])
                            if count_passada == last_passada:
                                end_position = int(row_text[2])

                if move_vazio == True:
                    position.append((start_position, end_position))

            else:

                p_passada = (faceamento - 1) * passada_por_faceamento + passada
                
                line = next( text_file )
                count_passada = 0
                for line in text_file:
                    row_text = line.split(';')
                    if int(row_text[0]) == 1:
                        count_passada += 1
                        if count_passada == p_passada:
                            start_position = int(row_text[1])
                            end_position = int(row_text[2])
                            break
                position.append((start_position, end_position))

        return position

    



    
    def extValues_LWFence(self, array):
        y = array[:]
        y.sort()
        N = len(y)

        Q1 = y[(N*1)//4]
        Q3 = y[(N*3)//4]
        IQR = Q3 - Q1
        min_before_lower_fence = Q1
        max_before_lower_fence = Q3

        for i in range(N):
            if y[i] > Q1 - 1.5 * IQR:
                min_before_lower_fence = y[i]
                break
        for i in range(N-1, -1, -1):
            if y[i] < Q3 + 1.5 * IQR:
                min_before_lower_fence = y[i]
                break
                
        return min_before_lower_fence, max_before_lower_fence


    def extValues_normalityBand(self, array):
        y = array[:]
        y.sort()
        #print(y)
        N = len(y)
        avg = np.average(array)
        std = np.std(array)

        lower = y[0]
        upper = y[-1]

        for i in range(N):
            if y[i] > avg - 10 * (std / np.sqrt(N)):
                lower = y[i+1]
                break
        for i in range(N-1, 0, -1):
            if y[i] < avg + 10 * (std / np.sqrt(N)):
                upper = y[i-1]
                break
                        
        return lower, upper


    #def getParameters( self, getLU = True, getParts = False ):
    #    if getLU == True:
    #        self.importLUData()
    #    if getParts == True:
    #        for sensor in self.Sensores:
    #            self.importVibParts(sensor)


    #Get Vibrations
    def getVibrations(self, test):

        if type(test) == list:
            if test[0] == test[1]:
                test = test[0]

        if self.activeVibrations == True:
            self.clearVibrations()
        
        if self.parameters["file_type"] == "csv":
            #Importação de um teste isolado ou sequência de testes
            if type(test) != list:
                # teste isolado
                # read data
                file_name = self.pathColeta + "/teste_" + str(test) + ".csv"
                if os.path.isfile(file_name) == True:
                    text_file = open(self.pathColeta + "/teste_" + str(test) + ".csv", "r" )

                    #skip the first two lines
                    next( text_file )
                    next( text_file )
                    self.numVibrations = 0;
                    for line in text_file:
                        row_text = line.split(';')
                        for sensor in self.Sensores:
                            self.Vibrations[sensor-1].append( float(row_text[sensor]) )
                        self.numVibrations += 1
                    text_file.close()

                else:
                    return False

            else:
                #Importa de parte de um teste ou uma sequência de partes
                print('Get Vibrations C', self.coleta)
                self.duration = 0
                for part in range(test[0], test[1]+1):

                    file_name = self.pathColeta + "/Dados/teste_" + str(part) + ".csv"
                    if os.path.isfile(file_name) == False:
                        print('building part', part)
                        for sensor in self.Sensores:
                            self.Vibrations[sensor-1] += [0] * (26000000// (self.skip + 1))
                            # number of vibration records
                            self.numVibrations += 26000000 // (self.skip + 1)
                            # Duration in seconds
                            self.duration += (26000000/(self.skip+1)) / self.parameters["sample_rate"]
                            # time vector
                    else:
                        print('importing part', part)
                        # Read data                    
                        text_file = open( file_name, "r" )
                        # skip the first two lines
                        next( text_file )
                        next( text_file )
                        i = 0;
                        for line in text_file:
                            row_text = line.split(';')
                            i += 1
                            for sensor in self.Sensores:
                                if i % (self.skip+1) == 0:
                                    self.Vibrations[sensor-1].append( float(row_text[sensor]) )
                                    self.Time.append( int(row_text[0]) - self.parameters['first_timestamp'] )

                        text_file.close()

        
        elif self.parameters["file_type"] == "dxd":
            
            if type(test) != list:
            
                for part in range(self.dxdPart[0], self.dxdPart[1]+1):
                    file_name = self.path + str(self.coleta) + "/Dados/Test" + str(test) + "_{:04d}.dxd".format(part)

                    print( file_name )
                    with dw.open( file_name ) as f:
                        canais = []
                        canal = [[],[],[]]
                        for ch in f.values():
                            canais.append( ch.name )
                        
                        for sensor in self.Sensores:
                            canal[sensor-1] = f[canais[sensor-1]].series()
                            
                            vibrations = list(canal[sensor-1].values)
                            for i in range(len(vibrations)):
                                if i % (self.skip+1) == 0:
                                    self.Vibrations[sensor-1].append( vibrations[i] )

            else:
                
                print()
                for t in range(test[0], test[1]+1):
                    for part in range(self.dxdPart[0], self.dxdPart[1]+1):
                        file_name = self.path + str(self.coleta) + "/Dados/Test" + str(t) + "_{:04d}.dxd".format(part)

                        if os.path.exists(file_name) == False:
                            continue

                        print('C', file_name )
                        with dw.open( file_name ) as f:
                            canais = []
                            canal = [[],[],[]]
                            for ch in f.values():
                                canais.append( ch.name )
                            
                            for sensor in self.Sensores:
                                canal[sensor-1] = f[canais[sensor-1]].series()
                                vibrations = list(canal[sensor-1].values)
                                #Time = list(canal[sensor-1].index)
                                for i in range(len(vibrations)):
                                    if i % (self.skip+1) == 0:
                                        self.Vibrations[sensor-1].append( vibrations[i] )
                                        #self.Time.append( Time[i] )
                    

        # number of vibrations
        self.numVibrations = len(self.Vibrations[self.Sensores[0]-1])

        # remoção do valor médio das vibrações
        for sensor in self.Sensores:
            avg = np.average( self.Vibrations[sensor-1] )
            self.Vibrations[sensor-1] = [v - avg for v in self.Vibrations[sensor-1]]

        # Convert g to m/s^2
        if self.parameters['unit'] == 'g':
            for sensor in self.Sensores:
                self.Vibrations[sensor-1] = [ v * self.parameters['const_g'] for v in self.Vibrations[sensor-1] ]
           
        # Duration in seconds
        self.duration = self.numVibrations / self.parameters["sample_rate"]
        # time vector
        #self.Time = np.linspace(0.0, self.duration, self.numVibrations, endpoint=False).tolist()

        self.activeVibrations = True
        return self.activeVibrations




    def plotVibration( self, sensor, details = False ):

        self.create_directory( self.parameters["path"]["charts"])

        plt.rc('font', **{'size' : 18})
        plt.ticklabel_format(style = 'plain')
        plt.figure( 
            figsize = (
                self.parameters['figure']['vibration']['lenght'], 
                self.parameters['figure']['vibration']['hight']
                ), 
            dpi= self.parameters['figure']['vibration']['dpi'] 
            )

        title = ''
        if type(self.test) != list:
            title = self.parameters['figure']['vibration']['title']['teste'].format(self.test, self.coleta)
        else:
            title = self.parameters['figure']['vibration']['title']['testes'].format(self.test[0], self.test[1], self.coleta)

        #if self.dxdPart == None:
        plt.title( title )
        #else:
        #    if self.dxdPart[0] == self.dxdPart[1]:
        #        plt.title( title + " (arquivo " + str(self.dxdPart[0]) + ")" )
        #    else:
        #        plt.title( title + " (arquivos de " + str(self.dxdPart[0]) + " até " + str(self.dxdPart[1]) + ")" )
        
        plt.plot( self.Time, self.Vibrations[sensor-1], self.parameters['sensor']['colors'][str(sensor)] )

        if self.parameters['figure']['vibration']['limits']['activated'] == str(True):
            plt.ylim( 
                self.parameters['figure']['vibration']['limits']['lower'],
                self.parameters['figure']['vibration']['limits']['upper']
                )

        plt.ylabel( self.parameters['figure']['vibration']['yLabel'] )
        plt.xlabel( self.parameters['figure']['vibration']['xLabel'] )
        plt.legend( [self.parameters['sensor']['legend'][str(sensor)]], loc='upper left' )
        plt.grid( linestyle='--', axis='y' )

        figpath = self.parameters["path"]["charts"] + '/VibT' + str(self.test) + 'S' + str(sensor)
        if self.dxdPart == None:
            plt.savefig( figpath + '.png' )
        else:
            plt.savefig( figpath + 'P' + str(self.dxdPart) + '.png')
        plt.close()
        plt.cla()
        plt.clf()


    def plotEspectro2d( self, sensor ):
        self.create_directory(self.parameters["path"]["charts"])
        f, t, Sxx = signal.spectrogram( np.array(self.Vibrations[sensor-1]), self.parameters["sample_rate"], nperseg=int(self.parameters["sample_rate"]), mode='magnitude')

        for freq in self.Frequence:
            myfilter = (f<=freq*1000)
            f_i = f[myfilter]/1000
            Sxx_i = Sxx[myfilter, ...]

            #plt.rc('font', **{'size' : 18})
            plt.ticklabel_format(style = 'plain')
            plt.figure( 
                figsize = (
                    self.parameters['figure']['Spectro2D']['lenght'], 
                    self.parameters['figure']['Spectro2D']['hight']
                    ), 
                dpi= self.parameters['figure']['Spectro2D']['dpi'] 
                )

            title = ''
            if type(self.test) != list:
                title = self.parameters['figure']['Spectro2D']['title']['teste'].format(self.test, self.coleta, self.parameters['sensor']['legend'][str(sensor)])
            else:
                title = self.parameters['figure']['Spectro2D']['title']['testes'].format(self.test[0], self.test[1], self.coleta, self.parameters['sensor']['legend'][str(sensor)] )

            #if self.dxdPart == None:
            plt.title( title )
            #else:
            #    if self.dxdPart[0] == self.dxdPart[1]:
            #        plt.title( title + " (arquivo " + str(self.dxdPart[0]) + ")" )
            #    else:
            #        plt.title( title + " (arquivos de " + str(self.dxdPart[0]) + " até " + str(self.dxdPart[1]) + ")" )
     

            #https://matplotlib.org/stable/tutorials/colors/colormapnorms.html
            #(Terrain color map)
            colors_undersea = plt.cm.terrain( np.linspace(0, 0.17, 256) )
            colors_land = plt.cm.terrain( np.linspace(0.25, 1, 256) )
            all_colors = np.vstack((colors_undersea, colors_land))
            terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

            #divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=0.025, vmax=0.5)
            divnorm = colors.TwoSlopeNorm(
                    vmin = self.parameters['figure']['Spectro2D']['limits']['lower'], 
                    vcenter = self.parameters['figure']['Spectro2D']['limits']['center'], 
                    vmax = self.parameters['figure']['Spectro2D']['limits']['upper']
                )
            plt.pcolormesh(t[None, :], f_i[:, None], Sxx_i, norm=divnorm, cmap=terrain_map, shading='auto')

            cbar = plt.colorbar()
            cbar.set_label(self.parameters['figure']['Spectro2D']['zLabel'])
            plt.xlabel( self.parameters['figure']['Spectro2D']['xLabel'] )
            plt.ylabel( self.parameters['figure']['Spectro2D']['yLabel'] )

            figpath = self.pathCharts + '/Esp2DT' + str(self.test) + 'S' + str(sensor)
            if self.dxdPart == None:
                plt.savefig( figpath + '.png' )
            else:
                plt.savefig( figpath + 'P' + str(self.dxdPart) + '.png')
            plt.close()
            plt.cla()
            plt.clf()



    # Refatorar as demais funções



    def plotVibrationParts( self, sensor, details = False ):
        self.create_directory( self.parameters["path"]["charts_parts"])

        # Constructing
        for i in range(self.numParts[sensor-1]):

            plt.rc( 'font', **{'size' : 18} )
            plt.ticklabel_format( style = 'plain' )
            plt.figure( figsize= (self.largura, self.altura), dpi= self.my_dpi )
            title = "Gráfico no domínio do tempo do teste " + str(self.test) + " da coleta " + str(self.coleta) + " no intervalo de tempo " + str(self.parts[sensor-1][i])


            start = self.parts[sensor-1][i][1]
            end = self.parts[sensor-1][i][2]


            if self.coleta != 11 and self.coleta != 12:
                plt.title( title )
            else:
                if self.dxdPart[0] == self.dxdPart[1]:
                    plt.title( title + " (arquivo " + str(self.dxdPart[0]) + ")" )
                else:
                    plt.title( title + " (arquivos " + str(self.dxdPart) + ")" )
            

                plt.plot( self.Time[start:end], self.Vibrations[sensor-1][start:end], self.Colors[sensor-1] )

            Dets = ''
            if details == True:
                plt.ylim( self.vib_lower[sensor-1], self.vib_upper[sensor-1] )
                #plt.axhline(y=self.vib_rms[sensor-1], color='m', linestyle='--')
                Dets = 'D'

            plt.ylabel( self.yLabel['a'] )
            
            #if details == True:
                #plt.legend( self.Legend[sensor-1] + ['RMS'] )
            #else:
            plt.legend( self.Legend[sensor-1] )

            plt.xlabel( "Tempo (s)" )
            plt.grid( linestyle='--', axis='y' )

            figpath = self.pathChartsParts + '/VibT' + str(self.test) + 'S' + str(sensor) + "[" + str(start) + "," + str(end) + "]" + Dets

            if self.coleta != 11 and self.coleta != 12:
                plt.savefig( figpath + '.png' )
            else:
                plt.savefig( figpath + 'P' + str(self.dxdPart) + '.png')

            plt.close()
            plt.cla()
            plt.clf()
 


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

    
    def plotBars( self ):
        rms = [[],[],[]]
        label = []

        self.create_directory("Charts")

        # Read data
        text_file = open(self.pathData + "/data_" + ".txt", "r")
        for line in text_file:
            row_text = line.split(";")
            label.append(row_text[1])
            sensor = int(row_text[2])
            rms[sensor-1].append(float(row_text[5]))

        N = len( rms[self.Sensores[0]-1] )
        label = list(set(label))
        label.sort()

        plt.figure( figsize = (self.largura, self.altura), dpi= self.my_dpi )
        title = "RMS na coleta " + str(self.coleta)
        plt.title( title, fontsize = 15 )

        x = np.arange(N)

        #label = x+1

        legend = []
        for sensor in self.Sensores:
            legend.append("Sensor " + str(sensor))
            if sensor == 1:
                if len(Sensores) != 1:
                    plt.bar(x - 0.2, rms[sensor-1], 0.2, color=self.Colors[sensor-1])
                else:
                    plt.bar(x, rms[sensor-1], 0.2, color=self.Colors[sensor-1])
            if sensor == 2:
                plt.bar(x, rms[sensor-1], 0.2, color=self.Colors[sensor-1])
            if sensor == 3:
                plt.bar(x + 0.2, rms[sensor-1], 0.2, color=self.Colors[sensor-1])
        
        plt.xticks( x, label )
        plt.xlabel("testes", fontsize=15, labelpad=10 )
        plt.ylabel( self.yLabel['a'], fontsize=15, labelpad=10 )
        plt.legend( legend )
        plt.grid( linestyle='--', axis='y' )
        figpath = self.pathCharts + '/BarC' + str(self.coleta) 
        plt.savefig( figpath + '.png' )
        plt.close()
        plt.cla()
        plt.clf()


    def plotBarsFaceamentosRMS( self, nPassadas = 6, join = False):
        rms_passadas = [
            [[],[],[],[],[],[],[]],
            [[],[],[],[],[],[],[]],
            [[],[],[],[],[],[],[]]
            ]
        records_passadas = [
            [[],[],[],[],[],[],[]],
            [[],[],[],[],[],[],[]],
            [[],[],[],[],[],[],[]]
            ]
        
        rms_max = 0
        label = [[],[],[]]
        block_change = []
        bc = 1
        count = 0
        with open(self.pathData + "/trainingDataSet_a.txt", "r") as text_file:
            line = next(text_file)
            for line in text_file:
                row_text = line.split(";")
                sensor = int(row_text[4])
                passada = int(row_text[6])
                records = int(row_text[7])
                rms = float(row_text[9])
                block = int(row_text[2])
                if rms > rms_max:
                    rms_max = rms
                rms_passadas[sensor-1][passada-1].append(rms)
                records_passadas[sensor-1][passada-1].append(records)
                
                if passada == 1:
                    label[sensor-1].append(str(int(row_text[3])) + "-" + str(int(row_text[5])))

                    if sensor == self.Sensores[0]:
                        if block != bc:
                            bc = block
                            block_change.append(count)
                        count += 1

        N = len(rms_passadas[self.Sensores[0]-1][0])
        x = np.arange(N)


        # construct rms faceamentos
        rms_faceamentos = [[],[],[]]
        for sensor in self.Sensores:
            for r in range(N):
                rms = 0
                rec = 0
                for p in range(nPassadas):
                    rms += ( rms_passadas[sensor-1][p][r]**2 ) * records_passadas[sensor-1][p][r]
                    rec += records_passadas[sensor-1][p][r]
                if rec == 0:
                    rms = 0
                else:
                    rms = np.sqrt( rms/rec )
                rms_faceamentos[sensor-1].append(rms)

                
        if join == False:
            for sensor in self.Sensores:
                fig, ax1 = plt.subplots(figsize= (self.largura, self.altura), dpi= self.my_dpi)
                ax2 = ax1.twinx()
                fig.subplots_adjust(bottom=0.2)

                title = "RMS por faceamento e desgaste na coleta " + str(self.coleta)
                plt.title( title, fontsize = 15 )

                colors = [self.Colors[sensor-1] if i not in block_change else 'gray' for i in range(N)]
                
                ax1.bar( x, rms_faceamentos[sensor-1], 0.6, color=colors )

                #Tendency line
                V = [ v for v in rms_faceamentos[sensor-1] if v != 0 ]
                U = [ u for u in x if rms_faceamentos[sensor-1][u] != 0 ]
                z = np.polyfit( U, V, 1 )
                p = np.poly1d( z )
                ax1.plot( x, p(x), "k--" )

                
                # Ajustar importação do desgaste
                #ax2.plot([-1, 49, 63], [0, 0.26, 0.56], marker = 'o', markersize=10, color='black', linestyle="None")
                ax2.plot([-1, 24, 25, 26, 27, 28, 29], [0, 0.16, 0.18, 0.16, 0.16, 0.44, 0.44], marker = 'o', markersize=10, color='black', linestyle="None")
                ax2.plot( [29,   30,   31,   32,   33,   34,   35,   36,   37,   38,   39,   40,   41,   42], [ 0, 0.13, 0.15, 0.15, 0.15, 0.18, 0.20, 0.20, 0.20, 0.20, 0.27, 0.25, 0.24, 0.31], marker = 'o', markersize=10, color='cyan', linestyle="None")

                plt.setp(ax1, xticks=x)
                plt.setp(ax2, xticks=x)
                ax1.set_xticklabels(label[sensor-1], rotation=90, fontsize=12)
                ax1.set_xlabel("teste - faceamento", fontsize=15, labelpad=10 )
                ax1.set_ylabel( self.yLabel['a'], fontsize=15, labelpad=10 )
                ax2.set_ylabel( 'Desgaste (mm)', fontsize=15, labelpad=10 )
                ax1.set_ylim( 0, rms_max*1.1)
                ax2.set_ylim(-0.02, 0.7)
                ax1.legend( ['Linha de tendência ' + str(p), 'Sensor ' + str(sensor)], loc='upper left', fontsize = 15 )
                ax2.legend( ['Desgaste aresta 1', 'Desgaste aresta 2'], loc='upper right', fontsize = 15 )
                plt.grid( linestyle='--', axis='y' )
                figpath = self.pathCharts + '/BarFaceamentosC' + str(self.coleta) + 'S' + str(sensor)  
                plt.savefig( figpath + '.png' )
                plt.close()
                plt.cla()
                plt.clf()

        else:

            fig, ax1 = plt.subplots(figsize= (self.largura, self.altura), dpi= self.my_dpi)
            ax2 = ax1.twinx()
            fig.subplots_adjust(bottom=0.2)

            title = "RMS por faceamento e desgaste na coleta " + str(self.coleta)
            plt.title( title, fontsize = 15 )
            colors = [[self.Colors[sensor] if i not in block_change else 'gray' for i in range(N)] for sensor in range(3)]
            
            position = [[], [0,0,0], [0,-0.1, 0.1], [-0.2, 0, +0.2]]
            num_pos = len(self.Sensores)
            for s in self.Sensores:
                ax1.bar(x + position[num_pos][s-1], rms_faceamentos[s-1], 0.2, color=colors[s-1])

            # Ajustar importação do desgaste
           # ax2.plot([-1, 49, 63], [0, 0.26, 0.56], marker = 'o', markersize=10, color='black', linestyle="None")
            ax2.plot([-1, 24, 25, 26, 27, 28, 29], [0, 0.16, 0.18, 0.16, 0.16, 0.44, 0.44], marker = 'o', markersize=10, color='black', linestyle="None")
            ax2.plot( [29,   30,   31,   32,   33,   34,   35,   36,   37,   38,   39,   40,   41,   42], [ 0, 0.13, 0.15, 0.15, 0.15, 0.18, 0.20, 0.20, 0.20, 0.20, 0.27, 0.25, 0.24, 0.31], marker = 'o', markersize=10, color='cyan', linestyle="None")

            plt.setp( ax1, xticks=x )
            plt.setp( ax2, xticks=x )
            ax1.set_xticklabels( label[self.Sensores[0]-1], rotation=90, fontsize=12 )
            ax1.set_xlabel("teste - faceamento", fontsize=15, labelpad=10 )
            ax1.set_ylabel( self.yLabel['a'], fontsize=15, labelpad=10 )
            ax2.set_ylabel( 'Desgaste (mm)', fontsize=15, labelpad=10 )
            ax1.set_ylim( 0, rms_max*1.1)
            ax2.set_ylim(-0.02, 0.7)
            
            ax1.legend( ['Sensor ' + str(sensor) for sensor in self.Sensores], loc='upper left', fontsize = 15 )
            ax2.legend( ['Desgaste aresta 1', 'Desgaste aresta 2'], loc='upper right', fontsize = 15 )
            plt.grid( linestyle='--', axis='y' )
            figpath = self.pathCharts + '/BarFaceamentosC' + str(self.coleta) + 'S123' 
            plt.savefig( figpath + '.png' )
            plt.close()
            plt.cla()
            plt.clf()




    def plotBarsPassadasRMS( self, nPassadas = 6 ):

        rms_passadas = [
            [[],[],[],[],[],[],[]],
            [[],[],[],[],[],[],[]],
            [[],[],[],[],[],[],[]]
            ]
        
        rms_max = 0
        label = [[],[],[]]
        block_change = []
        bc = 1
        count = 0
        with open( self.pathData + "/trainingDataSet_a.txt", "r") as text_file:
            line = next(text_file)
            for line in text_file:
                row_text = line.split(";")
                sensor = int(row_text[4])
                passada = int(row_text[6])
                rms = float(row_text[9])
                block = int(row_text[2])
                if rms > rms_max:
                    rms_max = rms
                rms_passadas[sensor-1][passada-1].append(rms)
                
                if passada == 1:
                    label[sensor-1].append(str(int(row_text[3])) + "-" + str(int(row_text[5])))

                    if sensor == self.Sensores[0]:
                        if block != bc:
                            bc = block
                            block_change.append(count)
                        count += 1
            
        N = len(rms_passadas[self.Sensores[0]-1][0])
        x = np.arange(N)



        for sensor in self.Sensores:
            fig, ax1 = plt.subplots(figsize= (self.largura, self.altura), dpi= self.my_dpi)
            ax2 = ax1.twinx()
            fig.subplots_adjust( bottom=0.2 )

            title = "RMS por passada, faceamento e desgaste na Coleta " + str(self.coleta)
            plt.title( title, fontsize = 15 )

            colors = [self.Colors[sensor-1] if i not in block_change else 'gray' for i in range(N)]

            ax1.bar( x-0.25, rms_passadas[sensor-1][0], 0.1, color=colors )
            ax1.bar( x-0.15, rms_passadas[sensor-1][1], 0.1, color=colors )
            ax1.bar( x-0.05, rms_passadas[sensor-1][2], 0.1, color=colors )
            ax1.bar( x+0.05, rms_passadas[sensor-1][3], 0.1, color=colors )
            ax1.bar( x+0.15, rms_passadas[sensor-1][4], 0.1, color=colors )
            ax1.bar( x+0.25, rms_passadas[sensor-1][5], 0.1, color=colors )


            V = []
            for i in range(nPassadas):
                for v in rms_passadas[sensor-1][i]:
                    if v != 0:
                        V.append(v)

            U = [ u for u in x if rms_passadas[sensor-1][0][u] != 0 ] * nPassadas
            z = np.polyfit(U, V, 1)
            p = np.poly1d(z)
            ax1.plot(x, p(x), "k--")


            # Ajustar importação do desgaste
            #ax2.plot([-1, 49, 63], [0, 0.26, 0.56], marker = 'o', markersize=10, color='black', linestyle="None")
            ax2.plot([-1, 24, 25, 26, 27, 28, 29], [0, 0.16, 0.18, 0.16, 0.16, 0.44, 0.44], marker = 'o', markersize=10, color='black', linestyle="None")
            ax2.plot( [29,   30,   31,   32,   33,   34,   35,   36,   37,   38,   39,   40,   41,   42], [ 0, 0.13, 0.15, 0.15, 0.15, 0.18, 0.20, 0.20, 0.20, 0.20, 0.27, 0.25, 0.24, 0.31], marker = 'o', markersize=10, color='cyan', linestyle="None")


            plt.setp( ax1, xticks=x )
            plt.setp( ax2, xticks=x )
            ax1.set_xticklabels( label[sensor-1], rotation=90, fontsize=12 )
            ax1.set_xlabel( "teste - faceamento", fontsize=15, labelpad=10 )
            ax1.set_ylabel( self.yLabel['a'], fontsize=15, labelpad=10 )
            ax2.set_ylabel( 'Desgaste (mm)', fontsize=15, labelpad=10 )
            ax1.set_ylim( 0, rms_max*1.1 )
            ax2.set_ylim( -0.02, 0.7 )
            ax1.legend( ['Linha de tendência ' + str(p),'Sensor ' + str(sensor)], loc='upper left', fontsize = 15 )
            ax2.legend( ['Desgaste aresta 1', 'Desgaste aresta 2'], loc='upper right', fontsize = 15 )
            plt.grid( linestyle='--', axis='y' )
            
            figpath = self.pathCharts + '/BarPassadasC' + str(self.coleta) + 'S' + str(sensor) 
            plt.savefig( figpath + '.png' )
            plt.close()
            plt.cla()
            plt.clf()

      
    def plotBarsPassadasPower( self, nPassadas = 6 ):
        power_passadas = [
            [[],[],[],[],[],[],[]],
            [[],[],[],[],[],[],[]],
            [[],[],[],[],[],[],[]]
            ]
        
        power_max = 0
        label = [[],[],[]]
        block_change = []
        bc = 1
        count = 0
        with open( self.pathData + "/trainingDataSet_a.txt", "r") as text_file:
            line = next(text_file)
            for line in text_file:
                row_text = line.split(";")
                sensor = int(row_text[4])
                passada = int(row_text[6])
                power = float(row_text[13])
                block = int(row_text[2])
                if power > power_max:
                    power_max = power
                power_passadas[sensor-1][passada-1].append(power)
                
                if passada == 1:
                    label[sensor-1].append(str(int(row_text[3])) + "-" + str(int(row_text[5])))

                    if sensor == self.Sensores[0]:
                        if block != bc:
                            bc = block
                            block_change.append(count)
                        count += 1
 
        N = len(power_passadas[0][0])
        x = np.arange(N)
            
        sensor = self.getFirstSensorIndex()
        colors = ['olive' if i not in block_change else 'gray' for i in range(N)]

        fig, ax1 = plt.subplots(figsize= (self.largura, self.altura), dpi= self.my_dpi)
        ax2 = ax1.twinx()
        fig.subplots_adjust(bottom=0.2)

        title = "Potência por passada, faceamento e desgaste na coleta " + str(self.coleta)
        plt.title( title, fontsize = 15 )

        ax1.bar( x-0.25, power_passadas[sensor-1][0], 0.1, color=colors)
        ax1.bar( x-0.15, power_passadas[sensor-1][1], 0.1, color=colors)
        ax1.bar( x-0.05, power_passadas[sensor-1][2], 0.1, color=colors)
        ax1.bar( x+0.05, power_passadas[sensor-1][3], 0.1, color=colors)
        ax1.bar( x+0.15, power_passadas[sensor-1][4], 0.1, color=colors)
        ax1.bar( x+0.25, power_passadas[sensor-1][5], 0.1, color=colors)

        V = []
        for i in range(nPassadas):
            for v in power_passadas[sensor-1][i]:
                if v != 0:
                    V.append(v)

        U = [u for u in x if power_passadas[sensor-1][0][u] != 0] * nPassadas
        z = np.polyfit(U, V, 1)
        p = np.poly1d(z)
        ax1.plot(x, p(x), "k--")


        ax2.plot([-1, 49, 63], [0, 0.26, 0.56], marker = 'o', markersize=10, color='black', linestyle="None")

        plt.setp(ax1, xticks=x)
        plt.setp(ax2, xticks=x)
        ax1.set_xticklabels(label[sensor-1], rotation=90, fontsize=12)
        ax1.set_xlabel("teste - faceamento", fontsize=15, labelpad=10 )
        ax1.set_ylabel( 'Potência', fontsize=15, labelpad=10 )
        ax2.set_ylabel( 'Desgaste (mm)', fontsize=15, labelpad=10 )
        ax1.set_ylim( 0, power_max*1.2)
        ax2.set_ylim(-0.02, 0.7)
        ax1.legend( ['Linha de tendência ' + str(p)], loc='upper left', fontsize = 15 )
        ax2.legend( ['Desgaste'], loc='upper right', fontsize = 15 )
        plt.grid( linestyle='--', axis='y' )
        figpath = self.pathCharts + '/BarPassadasPotenciaC' + str(self.coleta)
        plt.savefig( figpath + '.png' )
        plt.close()
        plt.cla()
        plt.clf()


                
                
        



    def buildTable(self):

        num_test = [[],[],[]]
        num_lines = [[],[],[]]
        duration = [[],[],[]]
        sensores = [[],[],[]]
        rms = [[],[],[]]
        vib_min = [[],[],[]]
        vib_max = [[],[],[]]
        dft_max = [[],[],[]]

        self.create_directory("Data")

        # Read data
        text_file = open(self.pathData + "/data_" + ".txt", "r")
        for line in text_file:
            row_text = line.split(";")
            sensor = int(row_text[2])

            num_test[sensor-1].append(int(row_text[1]))
            num_lines[sensor-1].append(int(row_text[3]))
            duration[sensor-1].append(float(row_text[4]))
            sensores[sensor-1].append(int(row_text[2]))
            rms[sensor-1].append(float(row_text[5]))
            vib_min[sensor-1].append(float(row_text[12]))
            vib_max[sensor-1].append(float(row_text[15]))
            dft_max[sensor-1].append(float(row_text[17]))
        text_file.close()

        N = len(rms[0])

        with open(self.pathData + "/table_" + ".tex", "w") as f:
            for i in range(N):
                f.write("\\midrule\n")
                for sensor in self.Sensores[:1]:
                    f.write("{:2d}".format(num_test[sensor-1][i]) + 
                            " & {:10,d}".format(num_lines[sensor-1][i]).replace(',','.') + 
                            " & {:10.2f}".format(duration[sensor-1][i]).replace('.',',') + 
                            " & {:2d}".format(sensores[sensor-1][i]) + 
                            " & {:7.2f}".format(rms[sensor-1][i]).replace('.',',') + 
                            " & {:7.2f}".format(vib_min[sensor-1][i]).replace('.',',') + 
                            " & {:7.2f}".format(vib_max[sensor-1][i]).replace('.',',') + 
                            " & {:7.2f}".format(dft_max[sensor-1][i]).replace('.',',') + 
                            " \\\\ \n")
                for sensor in self.Sensores[1:]:
                    f.write("  " + 
                            " &           " +
                            " &           " +
                            " & {:2d}".format(sensores[sensor-1][i]) + 
                            " & {:7.2f}".format(rms[sensor-1][i]).replace('.',',') + 
                            " & {:7.2f}".format(vib_min[sensor-1][i]).replace('.',',') + 
                            " & {:7.2f}".format(vib_max[sensor-1][i]).replace('.',',') + 
                            " & {:7.2f}".format(dft_max[sensor-1][i]).replace('.',',') + 
                            " \\\\ \n")

            f.write("\\midrule\n")
            f.write("\\midrule\n")
            f.write("\\multicolumn{3}{c}{\\multirow{3}{*}{Geral de 1 até " + str(num_test[0][-1]) + "}}\n")
            for sensor in self.Sensores:
                f.write("  " + 
                        "             " +
                        "             " +
                        " & {:2d}".format(sensores[sensor-1][i]) + 
                        " & {:7.2f}".format(np.max(rms[sensor-1])).replace('.',',') + 
                        " & {:7.2f}".format(np.min(vib_min[sensor-1])).replace('.',',') + 
                        " & {:7.2f}".format(np.max(vib_max[sensor-1])).replace('.',',') + 
                        " & {:7.2f}".format(np.max(dft_max[sensor-1])).replace('.',',') + 
                        " \\\\ \n")
                   
    
   
    def plotVibFaces(self, sensor, time, vib1, test1, face1, color1=0, details = False):

        plt.rc('font', **{'size' : 18})
        plt.ticklabel_format( style = 'plain' )
        plt.figure( figsize= (self.largura, self.altura), dpi= self.my_dpi )
        title = "Gráfico no domínio do tempo do(s) " + " dos testes " + str(test1) + "-" + str(face1) + " da coleta " + str(self.coleta)

        plt.plot( time, vib1, self.Colors[color1] )
        plt.legend( ['teste ' + str(test1) + ' - faceamento ' + str(face1)] )

        Dets = ''
        if details == True:
            plt.ylim( self.vib_lower[sensor-1], self.vib_upper[sensor-1] )
            Dets = 'D'
        plt.ylabel( self.yLabel['a'] )
        
        plt.xlabel( "Tempo (s)" )
        plt.grid( linestyle = '--', axis = 'y' )

        figpath = ''
        figpath = self.pathCharts + '/VibT' + str(test1) + 'F' + str(face1) + 'S' + str(sensor) + Dets

        if self.coleta != 11 and self.coleta != 12:
            plt.savefig( figpath + '.png' )
        else:
            plt.savefig( figpath + 'P' + str(self.dxdPart) + '.png' )

        plt.close()
        plt.cla()
        plt.clf()


    def plotVibFacesOver(self, sensor, time, vib1, vib2, test1, face1, test2, face2, color1 = 0, color2 = 1, invert = False, details = False):

        plt.rc('font', **{'size' : 18})
        plt.ticklabel_format( style = 'plain' )
        plt.figure( figsize= (self.largura, self.altura), dpi= self.my_dpi )
        title = "Gráfico no domínio do tempo do(s) " + " dos testes " + str(test1) + "-" + str(face1) + " e " + str(test2) + "-" + str(face2) + " da coleta " + str(self.coleta)

        if self.coleta != 11 and self.coleta != 12:
            plt.title( title )
        else:
            if self.dxdPart[0] == self.dxdPart[1]:
                plt.title( title + " (arquivo " + str(self.dxdPart[0]) + ")" )
            else:
                plt.title( title + " (arquivos " + str(self.dxdPart) + ")" )


        if invert == False:
            plt.plot( time, vib1, self.ColorsFace[sensor-1][color1] )
            plt.plot( time, vib2, self.ColorsFace[sensor-1][color2] )
            plt.legend( ['teste ' + str(test1) + ' - faceamento ' + str(face1) + ' - sensor ' + str(sensor), 'teste ' + str(test2) + ' - faceamento ' + str(face2) + ' - sensor ' + str(sensor)] )
        else:
            plt.plot( time, vib2, self.ColorsFace[sensor-1][color2] )
            plt.plot( time, vib1, self.ColorsFace[sensor-1][color1] )
            plt.legend( ['teste ' + str(test2) + ' - faceamento ' + str(face2) + ' - sensor ' + str(sensor), 'teste ' + str(test1) + ' - faceamento ' + str(face1) + ' - sensor ' + str(sensor)] )

        Dets = ''
        if details == True:
            plt.ylim( self.vib_lower[sensor-1], self.vib_upper[sensor-1] )
            Dets = 'D'
        plt.ylabel( self.yLabel['a'] )
        
        plt.xlabel( "Tempo (s)" )
        plt.grid( linestyle = '--', axis = 'y' )

        figpath = ''
        if invert == False:
            figpath = self.pathCharts + '/VibT' + str(test1) + 'F' + str(face1) + 'T' + str(test2) + 'F' + str(face2) + 'S' + str(sensor) + Dets
        else:
            figpath = self.pathCharts + '/VibT' + str(test2) + 'F' + str(face2) + 'T' + str(test1) + 'F' + str(face1) + 'S' + str(sensor) + Dets

        if self.coleta != 11 and self.coleta != 12:
            plt.savefig( figpath + '.png' )
        else:
            plt.savefig( figpath + 'P' + str(self.dxdPart) + '.png' )

        plt.close()
        plt.cla()
        plt.clf()
   

    def plotDFTFacesOver(self, sensor, vib1, vib2, test1, face1, test2, face2, color1 = 0, color2 = 1, invert = False, details = False):

        N, xf1, yf1 = self.DFTNormalized(vib1)
        N, xf2, yf2 = self.DFTNormalized(vib2)

        for freq in self.Frequence:
            plt.rc('font', **{'size' : 18})
            plt.ticklabel_format(style = 'plain')
            plt.figure( figsize= (self.largura, self.altura), dpi= self.my_dpi )

            title = "Gráfico no domínio da frequência dos testes " + str(test1) + "-" + str(face1) + " e " + str(test2) + "-" + str(face2) + " da coleta " + str(self.coleta)

            if self.coleta != 11 and self.coleta != 12:
                plt.title( title )
            else:
                if self.dxdPart[0] == self.dxdPart[1]:
                    plt.title( title + " (arquivo " + str(self.dxdPart[0]) + ")" )
                else:
                    plt.title( title + " (arquivos " + str(self.dxdPart) + ")" )

            if invert == False:
                plt.plot( xf1[:int(N * ((2000*freq) / self.parameters["sample_rate"]))], yf1[:int(N * (2000*freq/self.parameters["sample_rate"]))], self.ColorsFace[sensor-1][color1] )
                plt.plot( xf2[:int(N * ((2000*freq) / self.parameters["sample_rate"]))], yf2[:int(N * (2000*freq/self.parameters["sample_rate"]))], self.ColorsFace[sensor-1][color2] )
                plt.legend( ['teste ' + str(test1) + ' - faceamento ' + str(face1) + ' - sensor ' + str(sensor), 'teste ' + str(test2) + ' - faceamento ' + str(face2) + ' - sensor ' + str(sensor)] )
            else:
                plt.plot( xf2[:int(N * ((2000*freq) / self.parameters["sample_rate"]))], yf2[:int(N * (2000*freq/self.parameters["sample_rate"]))], self.ColorsFace[sensor-1][color2] )
                plt.plot( xf1[:int(N * ((2000*freq) / self.parameters["sample_rate"]))], yf1[:int(N * (2000*freq/self.parameters["sample_rate"]))], self.ColorsFace[sensor-1][color1] )
                plt.legend( ['teste ' + str(test2) + ' - faceamento ' + str(face2) + ' - sensor ' + str(sensor), 'teste ' + str(test1) + ' - faceamento ' + str(face1) + ' - sensor ' + str(sensor)] )

            
            Dets = ''
            if details == True:
                plt.ylim( 0, self.dft_upper[sensor-1] )
                Dets = 'D'

            plt.xlabel("Frequência (kHz)")
            plt.grid(linestyle='--', axis='y')
            plt.ylabel( self.yLabel['a'] )

            figpath = ''
            if invert == False:
                figpath = self.pathCharts + '/DFT' + str(freq) + 'kT' + str(test1) + 'F' + str(face1) + 'T' + str(test2) + 'F' + str(face2) + 'S' + str(sensor) + Dets
            else:
                figpath = self.pathCharts + '/DFT' + str(freq) + 'kT' + str(test2) + 'F' + str(face2) + 'T' + str(test1) + 'F' + str(face1) + 'S' + str(sensor) + Dets

            if self.coleta != 11 and self.coleta != 12:
                plt.savefig( figpath + '.png' )
            else:
                plt.savefig( figpath + 'P' + str(self.dxdPart) + '.png')
            plt.close()
            plt.cla()
            plt.clf()

        del xf1
        del yf1
        del xf2
        del yf2



    def plotFaceamentos( self, test1, face1, test2, face2, details = False):
        self.create_directory("Charts")

        for sensor in self.Sensores:            
            [[start1, end1]] = self.determineVibRange(sensor, test1, face1, 0, True)
            [[start2, end2]] = self.determineVibRange(sensor, test2, face2, 0, True)

            duration = end1 - start1
            if end2 -start2 < duration:
                duration = end2 - start2

            time = np.linspace( 0.0, duration / self.parameters["sample_rate"], duration, endpoint = False )

            self.getVibrations(test1)
            vib1 = np.array(self.Vibrations[sensor-1][start1:(start1 + duration)], dtype='float32')

            self.getVibrations(test2)
            vib2 = np.array(self.Vibrations[sensor-1][start2:(start2 + duration)], dtype='float32')


            self.plotVibFacesOver(sensor, time, vib1, vib2, test1, face1, test2, face2, 0, 1, False, details)
            self.plotVibFacesOver(sensor, time, vib1, vib2, test1, face1, test2, face2, 0, 1, True, details)
            self.plotDFTFacesOver(sensor, vib1, vib2, test1, face1, test2, face2, 0, 1, False, details)
            self.plotDFTFacesOver(sensor, vib1, vib2, test1, face1, test2, face2, 0, 1, True, details)

            #self.plotVibFaces(sensor, time, vib1, test1, face1, 0, details)
            #self.plotVibFaces(sensor, time, vib2, test2, face2, 1, details)

            del vib1
            del vib2
            del time
            gc.collect()

        self.getVibrations(self.test)


    




    def plotDFTParts(self, sensor, details = False):

        self.create_directory("ChartsParts")

        # Constructing
        for i in range(self.numParts[sensor-1]):

            start = self.parts[sensor-1][i][1]
            end = self.parts[sensor-1][i][2]

            y_ = self.Vibrations[sensor-1][start:end]
            N, xf, yf = self.DFTNormalized(y_)

            for freq in self.Frequence:
                plt.rc('font', **{'size' : 18})
                plt.ticklabel_format(style = 'plain')
                plt.figure( figsize= (self.largura, self.altura), dpi= self.my_dpi )
                title = "Gráfico no domínio da frequência do teste " + str(self.test) + " da coleta " + str(self.coleta) + " no intervalo de tempo " + str(self.parts[sensor-1][i])

                if self.coleta != 11 and self.coleta != 12:
                    plt.title( title )
                else:
                    if self.dxdPart[0] == self.dxdPart[1]:
                        plt.title( title + " (arquivo " + str(self.dxdPart[0]) + ")" )
                    else:
                        plt.title( title + " (arquivos " + str(self.dxdPart) + ")" )
                
                plt.plot( xf[:int(N * ((2000*freq) / self.parameters["sample_rate"]))], yf[:int(N * (2000*freq/self.parameters["sample_rate"]))], self.Colors[sensor-1] )
                
                
                Dets = ''
                if details == True:
                    plt.ylim( 0, self.dft_upper[sensor-1] )
                    Dets = 'D'

                plt.xlabel("Frequência (kHz)")
                plt.grid(linestyle='--', axis='y')

                plt.ylabel( self.yLabel['a'] )
                plt.legend( self.Legend[sensor-1] )

                figpath = self.pathChartsParts + '/DFT' + str(freq) + 'kT' + str(self.test) + 'S' + str(sensor) + "[" + str(start) + "," + str(end) + "]" + Dets
                if self.coleta != 11 and self.coleta != 12:
                    plt.savefig( figpath + '.png' )
                else:
                    plt.savefig( figpath + 'P' + str(self.dxdPart) + '.png')
                plt.close()
                plt.cla()
                plt.clf()

   
    def plotDFT(self, sensor, details = False):

        self.create_directory("Charts")

        y_ = self.Vibrations[sensor-1]
        N, xf, yf = self.DFTNormalized(y_)

        for freq in self.Frequence:
            plt.rc( 'font', **{'size' : 18} )
            plt.ticklabel_format( style = 'plain' )
            plt.figure( figsize= (self.largura, self.altura), dpi= self.my_dpi )
            title = "Gráfico no domínio da frequência do teste " + str(self.test) + " da coleta " + str(self.coleta)

            if self.coleta != 11 and self.coleta != 12:
                plt.title( title )
            else:
                if self.dxdPart[0] == self.dxdPart[1]:
                    plt.title( title + " (arquivo " + str(self.dxdPart[0]) + ")" )
                else:
                    plt.title( title + " (arquivos " + str(self.dxdPart) + ")" )
                        
            plt.plot( xf[:int(N * ((2000*freq) / self.parameters["sample_rate"]))], yf[:int(N * (2000*freq/self.parameters["sample_rate"]))], self.Colors[sensor-1] )
            
            
            Dets = ''
            if details == True:
                plt.ylim( 0, self.dft_upper[sensor-1] )
                Dets = 'D'

            plt.xlabel("Frequência (kHz)")
            plt.grid(linestyle='--', axis='y')

            plt.ylabel( self.yLabel['a'] )
            plt.legend( self.Legend[sensor-1] )

            figpath = self.pathCharts + '/DFT' + str(freq) + 'kT' + str(self.test) + 'S' + str(sensor) + Dets
            if self.coleta != 11 and self.coleta != 12:
                plt.savefig( figpath + '.png' )
            else:
                plt.savefig( figpath + 'P' + str(self.dxdPart) + '.png')
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


    def vibrationParts(self, sensor, window = 2500, stitcher_coef = 50):

        if self.numParts[sensor-1] == 0:
            vibPartsFileExists = os.path.exists(self.pathData + "/VibPartsT" + str(self.test) + "S" + str(sensor) + ".txt")

            if vibPartsFileExists == True:

                self.importVibParts(sensor)

            else:
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

                #Export parts to Data folder
                self.exportVibParts(sensor)

    def exportVibParts(self, sensor):
        self.create_directory("Data")
        if self.numParts[sensor-1] == 0:
            return False
        
        with open(self.pathData + "/VibPartsT" + str(self.test) + "S" + str(sensor) + ".txt", "w") as f:
            f.write(str(self.numParts[sensor-1]) + ";\n")
            for i in range(self.numParts[sensor-1]):
                f.write(str(self.parts[sensor-1][i][0]) + "; " + str(self.parts[sensor-1][i][1]) + "; " + str(self.parts[sensor-1][i][2]) + ";\n")
        return True


    def importVibParts(self, sensor):

        if self.numParts[sensor-1] == 0:
            text_file = open(self.pathData + "/VibPartsT" + str(self.test) + "S" + str(sensor) + ".txt", "r")
            line = next (text_file)
            row_text = line.split(";")
            self.numParts[sensor-1] = int(row_text[0])
            self.numPassadas[sensor-1] = 0
            for line in text_file:
                row_text = line.split(";")
                self.parts[sensor-1].append([int(row_text[0]), int(row_text[1]), int(row_text[2])])
                self.numPassadas[sensor-1] += int(row_text[0])


    def importLUData(self):
        lower_vib = [[],[],[]]
        upper_vib = [[],[],[]]
        upper_dft = [[],[],[]]
        text_file = open(self.pathData + "/data.txt", "r")
        iteration = 0
        for line in text_file:
            row_text = line.split(";")
            sensor = int(row_text[2])
            
            #if self.test == int(row_text[1]) and iteration == self.dxdPart[0]:
            if self.test == int(row_text[1]):
                self.vib_rms[sensor-1] = float(row_text[5])
            lower_vib[sensor-1].append(float(row_text[13]))
            upper_vib[sensor-1].append(float(row_text[14]))
            upper_dft[sensor-1].append(float(row_text[16]))
        text_file.close()
        for sensor in self.Sensores:
            self.vib_lower[sensor-1], blank = self.extValues_normalityBand(lower_vib[sensor-1])
            blank, self.vib_upper[sensor-1] = self.extValues_normalityBand(upper_vib[sensor-1])
            blank, self.dft_upper[sensor-1] = self.extValues_normalityBand(upper_dft[sensor-1])


    def importPassadaData(self, teste, sensor, faceamento, passada, passada_por_faceamento): 
        id_passada = passada + (faceamento-1)*passada_por_faceamento

        numVibrations = 0
        duration = 0
        rms = 0
        vib_min = 0
        vib_max = 0
        dft_max = 0
        with open(self.pathData + "/detaileddata_" + ".txt", "r") as text_file:
            num_passada = 0
            for line in text_file:
                row_text = line.split(';')
                
                if teste != int(row_text[1]):
                    continue
                if sensor != int(row_text[2]):
                    continue
                if int(row_text[3]) != 1:
                    continue

                num_passada += 1
                if id_passada != num_passada:
                    continue

                numVibrations = int(row_text[4])
                duration = float(row_text[5])
                rms = float(row_text[6])
                vib_min = float(row_text[7])
                vib_max = float(row_text[10])
                dft_max = float(row_text[12])
                break

        return numVibrations, duration, rms, vib_min, vib_max, dft_max


    def exportTrainingDataSet(self):
        # This functions may be used to merge the files 'detaliedData' and 'PotenciaPorPassada'
        
        arq1_title = ''
        arq1_blocoOrigem = []
        arq1_bloco = []
        arq1_teste = []
        arq1_faceamento = []
        arq1_passada = []
        arq1_potenciaMediaObservada = []
        arq1_desgasteAntes = []
        arq1_desgasteDepois = []
        with open(self.pathData + "/PotenciaDesgastePorPassada.txt", "r") as text_file:
            line = next(text_file)
            arq1_title = line.split(";")
            for line in text_file:
                row_text = line.split(';')
                arq1_blocoOrigem.append(str(row_text[0]))
                arq1_bloco.append(int(row_text[1]))
                arq1_teste.append(int(row_text[2]))
                arq1_faceamento.append(int(row_text[3]))
                arq1_passada.append(int(row_text[4]))
                arq1_potenciaMediaObservada.append(int(row_text[5]))
                arq1_desgasteAntes.append(row_text[6].strip())
                arq1_desgasteDepois.append(row_text[7].strip())

        N1 = len(arq1_teste)
        
        
        arq2_teste = []
        arq2_sensor = []
        arq2_tipo_movimento = []
        arq2_numVibrations = []
        arq2_duration = []
        arq2_rms = []
        arq2_vib_min = []
        arq2_vib_max = []
        with open(self.pathData + "/detaileddata_" + ".txt", "r") as text_file:
            for line in text_file:
                row_text = line.split(';')
                arq2_teste.append(int(row_text[1]))
                arq2_sensor.append(int(row_text[2]))
                arq2_tipo_movimento.append(int(row_text[3]))
                arq2_numVibrations.append(int(row_text[4]))
                arq2_duration.append(float(row_text[5]))
                arq2_rms.append(float(row_text[6]))
                arq2_vib_min.append(float(row_text[7]))
                arq2_vib_max.append(float(row_text[10]))
        N2 = len(arq2_teste)
        

        with open(self.pathData + "/trainingDataSet_" + ".txt", "w") as f:
            
            f.write("coleta; "+
                    "blocoOrigem; " +
                    "bloco; " +
                    "teste; " +
                    "sensor; " +
                    "faceamento; " +
                    "passada; " +
                    "numVibrations; " +
                    "duration; " +
                    "RMS; " +
                    "vibMax; " +
                    "vibMin; " +
                    "dftMax; " +
                    "potencia; " +
                    "desgasteAntes; " +
                    "desgasteDepois; " +
                    "\n")

            for i in range(N1):                
                for sensor in self.Sensores:
                    numVibrations, \
                    duration, \
                    rms, \
                    vib_min, \
                    vib_max, \
                    dft_max = self.importPassadaData(arq1_teste[i], sensor, arq1_faceamento[i], arq1_passada[i], 6)

                    f.write("{:2d};".format(self.coleta) +
                        " {:2s};".format(arq1_blocoOrigem[i]) +
                        " {:2d};".format(arq1_bloco[i]) + 
                        " {:2d};".format(arq1_teste[i]) + 
                        " {:2d};".format(sensor) + 
                        " {:2d};".format(arq1_faceamento[i]) + 
                        " {:2d};".format(arq1_passada[i]) + 
                        " {:10d};".format(numVibrations) + 
                        " {:10.5f};".format(duration) + 
                        " {:10.5f};".format(rms) + 
                        " {:10.5f};".format(vib_min) + 
                        " {:10.5f};".format(vib_max) + 
                        " {:10.5f};".format(dft_max) + 
                        " {:3d};".format(arq1_potenciaMediaObservada[i]) + 
                        " {:4s};".format(arq1_desgasteAntes[i]) + 
                        " {:4s};".format(arq1_desgasteDepois[i]) +
                        "\n")
                    


    def exportData( self, test, setLU = True, setParts = False ):

        self.create_directory("Data")
        info = self.getVibrations(test)
        
        if info == True:
            for sensor in self.Sensores:

                if setLU == True:
                    self.vib_rms[sensor-1] = self.get_rms(self.Vibrations[sensor-1])

                    self.vib_min[sensor-1], \
                    self.vib_lower[sensor-1], \
                    self.vib_upper[sensor-1], \
                    self.vib_max[sensor-1] = self.dispersion_measures(self.Vibrations[sensor-1])
                    
                    self.dft_min[sensor-1], \
                    self.dft_lower[sensor-1], \
                    self.dft_upper[sensor-1], \
                    self.dft_max[sensor-1] = self.dft_dispersion_measures(self.Vibrations[sensor-1])

                if setParts == True:
                    self.vibrationParts(sensor)
                    self.vib_rms_movi[sensor-1], self.vib_rms_pass[sensor-1] = self.get_rmsParts(sensor, self.Vibrations[sensor-1])

                with open(self.pathData + "/data_" + ".txt", "a") as f:
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
                        " {:10.5f};".format(self.dft_upper[sensor-1]) +
                        " {:10.5f};".format(self.dft_max[sensor-1]) +
                        "\n")

    def exportDetailedData( self, test ):

        self.create_directory("Data")
        info = self.getVibrations(test)

        if info == True:
            for sensor in self.Sensores:
                self.vibrationParts(sensor)
                for i in range(self.numParts[sensor-1]):
                    type_moviment = self.parts[sensor-1][i][0]
                    start = self.parts[sensor-1][i][1]
                    end = self.parts[sensor-1][i][2]

                    numVibrations = end - start
                    duration = numVibrations / self.parameters["sample_rate"]

                    vib_rms = self.get_rms(self.Vibrations[sensor-1][start:end])

                    vib_min, \
                    vib_lower, \
                    vib_upper, \
                    vib_max = self.dispersion_measures(self.Vibrations[sensor-1][start:end])
                    
                    dft_min, \
                    dft_lower, \
                    dft_upper, \
                    dft_max = self.dft_dispersion_measures(self.Vibrations[sensor-1][start:end])

                    with open(self.pathData + "/detaileddata.txt", "a") as f:
                        f.write("{:2d};".format(self.coleta) +
                            " {:2d};".format(self.test) +
                            " {:1d};".format(sensor) +
                            " {:1d};".format(type_moviment) +
                            " {:10d};".format(numVibrations) +
                            " {:10.5f};".format(duration) +
                            " {:10.5f};".format(vib_rms) +
                            " {:10.5f};".format(vib_min) +
                            " {:10.5f};".format(vib_lower) +
                            " {:10.5f};".format(vib_upper) +
                            " {:10.5f};".format(vib_max) +
                            " {:10.5f};".format(dft_upper) +
                            " {:10.5f};".format(dft_max) +
                            "\n")

    def convert_dxd_to_csv(self, test):
        
        for t in range(test[0], test[1]+1):
            for part in range(self.dxdPart[0], self.dxdPart[1]+1):
                file_name = self.path + str(self.coleta) + "/Dados/Test" + str(t) + "_{:04d}.dxd".format(part)

                if os.path.exists(file_name) == False:
                    continue

                

                print('C', file_name )
                with dw.open( file_name ) as f:

                    canais = []
                    canal = [[],[],[]]

                    print('\nInformações')
                    print(f.info)
                    print('\nEventos')
                    print(f.events())
                    print('\nData frame')
                    print(f.dataframe())
                    for ch in f.values():
                        canais.append( ch.name )
                    exit()
                    

                    
                    tmin = [[],[],[]]
                    tmax = [[],[],[]]
                    with open(self.path + str(self.coleta) + "/Dados/dxd_to_csv/Test" + str(t) + "_{:04d}.csv".format(part), 'w', newline='') as csvfile:
                        fieldnames = ['timestamp_sensor1', 'vibracao_sensor1','timestamp_sensor2', 'vibracao_sensor2','timestamp_sensor3', 'vibracao_sensor3']
                        write = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                        write.writeheader()
                    
                        vibrations = [[],[],[]]
                        times = [[],[],[]]
                        for sensor in self.Sensores:
                            canal[sensor-1] = f[canais[sensor-1]].series()
                            vibrations[sensor-1] = list(canal[sensor-1].values)
                            times[sensor-1] = list(canal[sensor-1].index)
                            tmin[sensor-1] = times[sensor-1][0]
                            tmax[sensor-1] = times[sensor-1][-1]

                        for i in range(len(times[self.Sensores[0]-1])):
                            row_value = dict()
                            for sensor in self.Sensores:
                                row_value[f'timestamp_sensor{sensor}'] = times[sensor-1][i]
                                row_value[f'vibracao_sensor{sensor}'] = vibrations[sensor-1][i]
                            write.writerow(row_value)

                    with open(self.path + str(self.coleta) + "/Dados/dxd_to_csv/times.csv", 'a', newline='') as csvfile:
                        fieldnames = ['file', 'sensor', 'min_timestamp','max_timestamp']
                        write = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                        for sensor in self.Sensores:
                            write.writerow({'file': "Test" + str(t) + "_{:04d}.csv".format(part),
                                            'sensor': sensor,
                                            'min_timestamp': tmin[sensor-1],
                                            'max_timestamp': tmax[sensor-1]})
                            
                            


if __name__ == '__main__':

    Coleta = int(sys.argv[1])
    Sensor = int(sys.argv[2])
    testFrom = int(sys.argv[3])
    testTo = int(sys.argv[4])
    condition = int(sys.argv[5])
    
    dxd = False
    dxdFrom = -1
    dxdTo = -1
    if len(sys.argv) > 6:
        dxd = True
        dxdFrom = int(sys.argv[6])
        dxdTo = int(sys.argv[7])

    TestSensores = [[],[],[],
            [1,2],
            [1,2,3],
            [1,2,3],
            [1,2,3],
            [1,2,3],
            [1,2,3],
            [2,3],
            [1],
            [1],
            [1,2,3],
            [1,2,3],
            [1,2,3],
            [1,2,3]
            ]

    

    #Plot Vibrations
    if condition == 0:
        if dxd == False:
            v = VibAnalysis( coleta=Coleta, test=[testFrom,testTo], Sensores=[Sensor], skip=15 )
            v.getVibrations(test=[testFrom,testTo])
            v.plotVibration(Sensor)
        else:
            v = VibAnalysis( coleta=Coleta, test=[testFrom,testTo], Sensores=[Sensor], dxdPart=[dxdFrom,dxdTo], skip=1 )
            v.getVibrations(test=[testFrom,testTo])
            v.plotVibration(Sensor)

        
    #Plot Spectrogram
    elif condition == 1:
        if dxd == False:
            v = VibAnalysis( coleta=Coleta, test=[testFrom,testTo], Sensores=[Sensor], skip=20, Frequence=[3] )
            v.getVibrations([testFrom,testTo])
            v.plotEspectro2d(Sensor)
        else:
            v = VibAnalysis( coleta=Coleta, test=[testFrom,testTo], Sensores=[Sensor], dxdPart=[dxdFrom,dxdTo], skip=1, Frequence=[3] )
            v.getVibrations([testFrom,testTo])
            v.plotEspectro2d(Sensor)

    #ExportData
    elif condition == 2:
        Sensores = TestSensores[Coleta-1]

        for test in range(testFrom, testTo+1):
            v = VibAnalysis(Coleta, test, Sensores)
            v.exportData(test, True, True)
            v.exportDetailedData(test)
            print("Export Data - Coleta", Coleta, "Test", test)

    elif condition == 3:
        pass
            #for test in range(testFrom, testTo+1):
            #    v = VibAnalysis(coleta, test, Sensores)
            #    v.getVibrations(test)
            #    v.getParameters(True, True)
            #    for sensor in Sensores:
            #        print("Plot - Coleta", coleta, "Test", test, "Sensor", sensor)
            #        #v.plotDFTParts(sensor, True)
            #        v.plotVibrationParts(sensor, True)

    elif condition == 4:
        v = VibAnalysis( coleta=Coleta, Sensores=TestSensores[Coleta-1])
        
        #print("Training Dataset")
        #v.exportTrainingDataSet()

        v.plotBars()
        v.plotBarsPassadasRMS(nPassadas = 7)
        v.plotBarsFaceamentosRMS( nPassadas = 7)
        v.plotBarsFaceamentosRMS( nPassadas = 7, join = True)
        
        #v.getParameters(True, True)
        #v.plotFaceamentos(1, testFrom, testTo, 1, True)
        #v.plotFaceamentos(1, testFrom, testTo, 2, True)
        
        #v.plotBarsPassadasPower(nPassadas = 7)
        #v.determineVibRange(1, 2, 2, 2, False)

    elif condition == 5:
        #Convert dxd file to csv
        v = VibAnalysis( coleta=Coleta, dxdPart=[dxdFrom,dxdTo], Sensores=[1,2,3])
        v.convert_dxd_to_csv(test=[testFrom,testTo])
    
        pass

