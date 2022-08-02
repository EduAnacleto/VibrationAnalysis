# Get Vibrations
# Quatro possibilidades estão sendo consideradas
#
# 1) Importação de um teste isolado
# 2) Importação de uma sequência de testes
# 3) Importação da parte de um teste
# 4) Importação de uma sequência de partes de um teste

def get_vibrations(self, test):

    if type(test) == list:
        if len(test) == 1:
            test = test[0]

    if self.activeVibrations == True:
        self.clearVibrations()
    
    if self.dxdPart == None:
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

                # Duration in seconds
                self.duration = self.numVibrations / self.sampleRate
                # time vector
                self.Time = np.linspace(0.0, self.duration, self.numVibrations, endpoint=False).tolist()
            else:
                return False

        else:
            #Importa de parte de um teste ou uma sequência de partes
            
            print('Get Vibrations C', self.coleta)
            self.duration = 0
            for part in range(test[0], test[1]+1):

                file_name = self.pathColeta + "/teste_" + str(part) + ".csv"
                if os.path.isfile(file_name) == False:
                    print('building part', part)
                    for sensor in self.Sensores:
                        self.Vibrations[sensor-1] += [0] * (26000000// (self.skip + 1))
                        # number of vibration records
                        self.numVibrations += 26000000 // (self.skip + 1)
                        # Duration in seconds
                        self.duration += (26000000/(self.skip+1)) / self.sampleRate
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

                    text_file.close()
                    # number of vibration records
                    self.numVibrations += i // (self.skip + 1)
                    # Duration in seconds
                    self.duration += (i/(self.skip+1)) / self.sampleRate
                    # time vector

            self.Time = np.linspace(0.0, self.duration, self.numVibrations, endpoint=False).tolist()


    else:

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
                    times = list(canal[0].index)
                    for i in range(len(vibrations)):
                        if i % (self.skip+1) == 0:
                            self.Vibrations[sensor-1].append( vibrations[i] )
                            self.Time.append( times[i] )

        # number of vibrations
        self.numVibrations = len(self.Vibrations[0])
        # Duration in seconds 
        self.duration = self.numVibrations / self.sampleRate
        #print(self.Vibrations[0][:100])
        
    self.activeVibrations = True
    return True


