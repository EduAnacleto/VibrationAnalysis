def export_vib_parts(self, sensor):
    self.create_directory("Data")
    if self.numParts[sensor-1] == 0:
        return False
    
    with open(self.pathData + "/VibPartsT" + str(self.test) + "S" + str(sensor) + ".txt", "w") as f:
        f.write(str(self.numParts[sensor-1]) + ";\n")
        for i in range(self.numParts[sensor-1]):
            f.write(str(self.parts[sensor-1][i][0]) + "; " + str(self.parts[sensor-1][i][1]) + "; " + str(self.parts[sensor-1][i][2]) + ";\n")
    return True


def import_vib_parts(self, sensor):

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


def import_lu_data(self):
    lower_vib = [[],[],[]]
    upper_vib = [[],[],[]]
    upper_dft = [[],[],[]]
    text_file = open(self.pathData + "/data_" + self.unit + ".txt", "r")
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


def import_passada_data(self, teste, sensor, faceamento, passada, passada_por_faceamento): 
    id_passada = passada + (faceamento-1)*passada_por_faceamento

    numVibrations = 0
    duration = 0
    rms = 0
    vib_min = 0
    vib_max = 0
    dft_max = 0
    with open(self.pathData + "/detaileddata_" + self.unit + ".txt", "r") as text_file:
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


def export_training_dataset(self):
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
    with open(self.pathData + "/detaileddata_" + self.unit + ".txt", "r") as text_file:
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
    

    with open(self.pathData + "/trainingDataSet_" + self.unit + ".txt", "w") as f:
        
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
                


def export_data( self, test, setLU = True, setParts = False ):

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

            with open(self.pathData + "/data_" + str(self.unit) + ".txt", "a") as f:
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

def export_detailed_data( self, test ):

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
                duration = numVibrations / self.sampleRate

                vib_rms = self.get_rms(self.Vibrations[sensor-1][start:end])

                vib_min, \
                vib_lower, \
                vib_upper, \
                vib_max = self.dispersion_measures(self.Vibrations[sensor-1][start:end])
                
                dft_min, \
                dft_lower, \
                dft_upper, \
                dft_max = self.dft_dispersion_measures(self.Vibrations[sensor-1][start:end])

                with open(self.pathData + "/detaileddata_" + str(self.unit) + ".txt", "a") as f:
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


