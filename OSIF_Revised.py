# python 3 tkinter import section
from tkinter import *
import tkinter as Tkinter
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox

# end python 3 tkinter import section

# python 2 Tkinter import section
'''
from Tkinter import *
import Tkinter
import tkFileDialog
import tkMessageBox
'''
# end python 2 Tkinter import section

import openpyxl
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # Allows for custom positioning of sub plots in matplotlib.
import scipy.optimize
import os
import re
import webbrowser
import sys


# Main program class which is called later to run the program.
class OSIF:
    def __init__(self, master):
        master.title("Open Source Impedance Fitter (OSIF) v1.25")
        master.grid()
        buttonFrame = Frame(master, pady=10)
        InputFrame = Frame(master, padx=10)
        OutputFrame = Frame(master, padx=10)
        self.plotFrame = Frame(master, bg='blue')
        self.plotFrameToolBar = Frame(master, bg='red')

        # Configure grid weighting
        Grid.grid_columnconfigure(buttonFrame, 0, weight=1)
        Grid.grid_rowconfigure(buttonFrame, 0, weight=1)
        Grid.grid_columnconfigure(self.plotFrame, 0, weight=1)
        Grid.grid_rowconfigure(self.plotFrame, 0, weight=1)
        Grid.grid_columnconfigure(self.plotFrameToolBar, 0, weight=1)
        Grid.grid_rowconfigure(self.plotFrameToolBar, 0, weight=1)

        buttonFrame.grid(row=0, columnspan=2)
        InputFrame.grid(row=1, column=0, sticky=N, pady=3)
        OutputFrame.grid(row=1, column=1, sticky=N, pady=3)
        self.plotFrame.grid(row=2, pady=1, padx=8, columnspan=5, sticky=N + S + E + W)
        self.plotFrameToolBar.grid(row=3, pady=1, padx=8, columnspan=5, sticky=S + W)

        # Add an auto-frequency window button to the buttonFrame.
        self.autoFreqButton = Button(buttonFrame, text="Auto-Freq Window", command=self.auto_freq_window_grid_search)
        self.autoFreqButton.grid(row=0, column=5, sticky=W)

        # Parameters – we now remove cell area and catalyst loading.
        self.Rmem = Param()
        self.Rcl = Param()
        self.Qdl = Param()
        self.Phi = Param()
        # Even though the original had Lwire and Theta, we now fix these at zero
        self.Lwire = Param()  # Will be fixed to 0 (not optimized)
        self.Theta = Param()  # Fixed to 0
        self.frequencyRange = Param()
        self.currentDataDir = Param()
        self.currentFileName = Tkinter.StringVar(master)
        self.model_selection = Tkinter.StringVar(master)
        self.currentFile = None
        self.avgResPer = Param()
        self.activeData = Data()

        entryFont = ("Calibri", 12)
        labelFont = ("Calibri", 12)

        # Column indices for display
        sdPerColumn = 5
        sdColumn = 4
        fitValueColumn = 3
        unitColumn = 2
        initValueColumn = 1
        varNameColumn = 0

        Label(InputFrame, text="Initial Values", font=labelFont).grid(row=1, column=initValueColumn, sticky=W)
        Label(OutputFrame, text="Fit Values", font=labelFont).grid(row=1, column=fitValueColumn, sticky=W)
        Label(OutputFrame, text="Estimated SE", font=labelFont).grid(row=1, column=sdColumn, sticky=W)
        Label(OutputFrame, text="SE % of fit value", font=labelFont).grid(row=1, column=sdPerColumn, sticky=W)

        ################################################
        ############ INPUT INITIAL VALUES ##############
        ################################################

        Label(InputFrame, text="Rmem:", font=labelFont).grid(row=2, column=varNameColumn, sticky=E)
        Label(InputFrame, text="[ohm·cm²]", font=labelFont).grid(row=2, column=unitColumn, sticky=W)
        self.Rmem.IE = Entry(InputFrame, width=10, font=entryFont)
        self.Rmem.IE.grid(row=2, column=initValueColumn)
        self.Rmem.IE.insert(0, "0.2")

        Label(InputFrame, text="Rcl:", font=labelFont).grid(row=3, column=varNameColumn, sticky=E)
        Label(InputFrame, text="[ohm·cm²]", font=labelFont).grid(row=3, column=unitColumn, sticky=W)
        self.Rcl.IE = Entry(InputFrame, width=10, font=entryFont)
        self.Rcl.IE.grid(row=3, column=initValueColumn)
        self.Rcl.IE.insert(0, "0.2")

        Label(InputFrame, text="Qdl:", font=labelFont).grid(row=4, column=varNameColumn, sticky=E)
        Label(InputFrame, text="[F/(cm²·sec^phi)]", font=labelFont).grid(row=4, column=unitColumn, sticky=W)
        self.Qdl.IE = Entry(InputFrame, width=10, font=entryFont)
        self.Qdl.IE.grid(row=4, column=initValueColumn)
        self.Qdl.IE.insert(0, "0.1")

        Label(InputFrame, text="Phi:", font=labelFont).grid(row=5, column=varNameColumn, sticky=E)
        Label(InputFrame, text="[ - ]", font=labelFont).grid(row=5, column=unitColumn, sticky=W)
        self.Phi.IE = Entry(InputFrame, width=10, font=entryFont)
        self.Phi.IE.grid(row=5, column=initValueColumn)
        self.Phi.IE.insert(0, "0.9")

        # Remove Cell Area and Catalyst Loading entirely; they are no longer needed.

        Label(InputFrame, text="Upper Frequency bound:", font=labelFont).grid(row=8, column=varNameColumn, sticky=E)
        Label(InputFrame, text="[Hz]", font=labelFont).grid(row=8, column=unitColumn, sticky=W)
        self.frequencyRange.IE = Entry(InputFrame, width=10, font=entryFont)
        self.frequencyRange.IE.grid(row=8, column=initValueColumn)
        self.frequencyRange.IE.insert(0, "10000")

        Label(InputFrame, text="Lower Frequency bound:", font=labelFont).grid(row=9, column=varNameColumn, sticky=E)
        Label(InputFrame, text="[Hz]", font=labelFont).grid(row=9, column=unitColumn, sticky=W)
        self.frequencyRange.OE = Entry(InputFrame, width=10, font=entryFont)
        self.frequencyRange.OE.grid(row=9, column=initValueColumn)
        self.frequencyRange.OE.insert(0, "10")

        ################################################
        ########### OUTPUT VALUES FROM FIT #############
        ################################################
        ioBoxWidth = 10
        self.Rmem.OE = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Rmem.OE.grid(row=2, column=fitValueColumn, sticky=W)
        self.Rmem.OE.insert(0, "---")
        self.Rmem.OE.config(state='readonly')

        self.Rcl.OE = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Rcl.OE.grid(row=3, column=fitValueColumn, sticky=W)
        self.Rcl.OE.insert(0, "---")
        self.Rcl.OE.config(state='readonly')

        self.Qdl.OE = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Qdl.OE.grid(row=4, column=fitValueColumn, sticky=W)
        self.Qdl.OE.insert(0, "---")
        self.Qdl.OE.config(state='readonly')

        self.Phi.OE = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Phi.OE.grid(row=5, column=fitValueColumn, sticky=W)
        self.Phi.OE.insert(0, "---")
        self.Phi.OE.config(state='readonly')

        # Lwire and Theta are fixed to 0 so no fit values are needed.
        self.Lwire.OE = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Lwire.OE.grid(row=6, column=fitValueColumn, sticky=W)
        self.Lwire.OE.insert(0, "0")
        self.Lwire.OE.config(state='readonly')

        self.Theta.OE = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Theta.OE.grid(row=7, column=fitValueColumn, sticky=W)
        self.Theta.OE.insert(0, "0")
        self.Theta.OE.config(state='readonly')

        ################################################
        ########### OUTPUT VALUE SD values #############
        ################################################

        self.Rmem.OESD = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Rmem.OESD.grid(row=2, column=sdColumn, sticky=W)
        self.Rmem.OESD.insert(0, "---")
        self.Rmem.OESD.config(state='readonly')

        self.Rcl.OESD = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Rcl.OESD.grid(row=3, column=sdColumn, sticky=W)
        self.Rcl.OESD.insert(0, "---")
        self.Rcl.OESD.config(state='readonly')

        self.Qdl.OESD = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Qdl.OESD.grid(row=4, column=sdColumn, sticky=W)
        self.Qdl.OESD.insert(0, "---")
        self.Qdl.OESD.config(state='readonly')

        self.Phi.OESD = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Phi.OESD.grid(row=5, column=sdColumn, sticky=W)
        self.Phi.OESD.insert(0, "---")
        self.Phi.OESD.config(state='readonly')

        self.Lwire.OESD = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Lwire.OESD.grid(row=6, column=sdColumn, sticky=W)
        self.Lwire.OESD.insert(0, "---")
        self.Lwire.OESD.config(state='readonly')

        self.Theta.OESD = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Theta.OESD.grid(row=7, column=sdColumn, sticky=W)
        self.Theta.OESD.insert(0, "---")
        self.Theta.OESD.config(state='readonly')

        ################################################
        ########## OUTPUT SD % OF VALUES #############
        ################################################

        self.Rmem.OESDP = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Rmem.OESDP.grid(row=2, column=sdPerColumn, sticky=W)
        self.Rmem.OESDP.insert(0, "---")
        self.Rmem.OESDP.config(state='readonly')

        self.Rcl.OESDP = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Rcl.OESDP.grid(row=3, column=sdPerColumn, sticky=W)
        self.Rcl.OESDP.insert(0, "---")
        self.Rcl.OESDP.config(state='readonly')

        self.Qdl.OESDP = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Qdl.OESDP.grid(row=4, column=sdPerColumn, sticky=W)
        self.Qdl.OESDP.insert(0, "---")
        self.Qdl.OESDP.config(state='readonly')

        self.Phi.OESDP = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Phi.OESDP.grid(row=5, column=sdPerColumn, sticky=W)
        self.Phi.OESDP.insert(0, "---")
        self.Phi.OESDP.config(state='readonly')

        self.Lwire.OESDP = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Lwire.OESDP.grid(row=6, column=sdPerColumn, sticky=W)
        self.Lwire.OESDP.insert(0, "---")
        self.Lwire.OESDP.config(state='readonly')

        self.Theta.OESDP = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.Theta.OESDP.grid(row=7, column=sdPerColumn, sticky=W)
        self.Theta.OESDP.insert(0, "---")
        self.Theta.OESDP.config(state='readonly')

        ################################################
        ########## OUTPUT AVG RES |Z| VALUE ############
        ################################################

        Label(OutputFrame, text="Avg. |Z| residual % of data |Z|:", font=labelFont).grid(row=9,
                                                                                         column=(sdPerColumn - 2),
                                                                                         columnspan=2, pady=20,
                                                                                         sticky=E)
        self.avgResPer.AVGRESPER = Entry(OutputFrame, width=ioBoxWidth, font=entryFont)
        self.avgResPer.AVGRESPER.grid(row=9, column=sdPerColumn, sticky=W)
        self.avgResPer.AVGRESPER.insert(0, "---")
        self.avgResPer.AVGRESPER.config(state='readonly')

        ################################################
        #################### BUTTONS ###################
        ################################################

        self.simB = Button(buttonFrame, text="Select Data Directory", command=lambda: self.SelectDataDir())
        self.simB.grid(row=0, column=0, sticky=E)

        self.simB = Button(buttonFrame, text="Model Info.", command=self.openModelInfo)
        self.simB.grid(row=0, column=1, sticky=E)

        self.simB = Button(buttonFrame, text="Citation Info.", command=self.openCitationInfo)
        self.simB.grid(row=0, column=2, sticky=W)

        self.fitB = Button(buttonFrame, text="Fit", command=self.PerformFit)
        self.fitB.grid(row=0, column=3, sticky=E)

        self.simB = Button(buttonFrame, text="Simulate", command=self.PerformSim)
        self.simB.grid(row=0, column=4, sticky=W)

        self.simB = Button(OutputFrame, text="Save Model Data", command=self.SaveData)
        self.simB.grid(row=9, column=sdPerColumn - 2, columnspan=3, sticky=N)

        Label(buttonFrame, text="Current Directory:", font=labelFont).grid(row=1, column=0, sticky=E)
        self.currentDataDir.IE = Entry(buttonFrame, width=60, font=entryFont)
        self.currentDataDir.IE.grid(row=1, column=1, sticky=W, columnspan=2)
        self.currentDataDir.IE.insert(0, "---")
        self.currentDataDir.IE.config(state='readonly')

        Label(buttonFrame, text="Current Data File:", font=labelFont).grid(row=2, column=0, sticky=E)
        self.choices = ['Data Directory not selected']
        self.fileSelectComboBox = OptionMenu(buttonFrame, self.currentFileName, *self.choices)
        self.fileSelectComboBox.grid(row=2, column=1, sticky=EW, columnspan=2)
        self.fileSelectComboBox.config(font=entryFont)

        Label(buttonFrame, text="Select Circuit Model:", font=labelFont).grid(row=3, column=0, sticky=E)
        self.eis_model = ["Transmission Line", "1-D Linear Diffusion", "1-D Spherical Diffusion"]
        self.model_selection.set(self.eis_model[0])
        self.fileSelectModelBox = OptionMenu(buttonFrame, self.model_selection, *self.eis_model)
        self.fileSelectModelBox.grid(row=3, column=1, sticky=EW, columnspan=2)
        self.fileSelectModelBox.config(font=entryFont)

    def openModelInfo(self):
        if self.model_selection.get() == self.eis_model[0]:
            print("Using Transmission Line model (with Lwire and Theta fixed to 0).")
            webbrowser.open("https://iopscience.iop.org/article/10.1149/2.0361506jes")
        elif self.model_selection.get() == self.eis_model[1]:
            print("1-D linear diffusion model selected.")
            webbrowser.open(
                "https://www.researchgate.net/publication/342833389_Handbook_of_Electrochemical_Impedance_Spectroscopy_DIFFUSION_IMPEDANCES")
        elif self.model_selection.get() == self.eis_model[2]:
            print("1-D spherical diffusion model selected.")
            webbrowser.open(
                "https://www.researchgate.net/publication/342833389_Handbook_of_Electrochemical_Impedance_Spectroscopy_DIFFUSION_IMPEDANCES")

    def openCitationInfo(self):
        print(
            "Citation Info:\nPlease cite matplotlib, scipy, and numpy appropriately.\nFor this code, refer to the OSIF GitHub repository.")
        webbrowser.open("https://www.scipy.org/citing.html")
        webbrowser.open("https://github.com/NREL/OSIF")

    def SelectDataDir(self):
        newDir = tkFileDialog.askdirectory(title="Select EIS data directory") + '/'
        self.currentDataDir.IE.config(state='normal')
        self.currentDataDir.IE.delete(0, END)
        self.currentDataDir.IE.insert(0, newDir)
        self.currentDataDir.IE.config(state='readonly')

        dirList = os.listdir(newDir)
        dirList = [dataFile for dataFile in dirList if
                   ('.txt' in dataFile) | ('.xls' in dataFile) | ('.xlsx' in dataFile)]
        self.fileSelectComboBox.configure(state='normal')
        menu = self.fileSelectComboBox.children['menu']
        menu.delete(0, 'end')
        for file in dirList:
            menu.add_command(label=file, command=lambda v=self.currentFileName, l=file: v.set(l))
        print('Selected Data Directory: ' + self.currentDataDir.IE.get())

    def LoadSElectedFile(self):
        if (len(self.currentFileName.get()) == 0) or (self.currentFileName.get() == '---'):
            print('Attempt to load on null selection')
            return

        print("\nLoading file: " + self.currentFileName.get())
        self.activeData.rawFrequency = []
        self.activeData.rawzPrime = []
        self.activeData.rawZdoublePrime = []
        self.activeData.rawzMod = []
        self.activeData.rawZExperimentalComplex = []
        self.activeData.rawmodZExperimentalComplex = []
        self.activeData.rawPhase = []

        self.activeData.dataName = self.currentFileName.get()

        if ".xlsx" in self.activeData.dataName:
            self.activeData.dataNameNoExt = self.activeData.dataName[:-5]
        else:
            self.activeData.dataNameNoExt = self.activeData.dataName[:-4]

        if self.currentFileName.get().endswith('.txt'):
            self.activeData.dataNameNoExt = self.activeData.dataName[:-4]
            self.currentFile = open(self.currentDataDir.IE.get() + self.currentFileName.get())
            dataLineString = self.currentFile.readline()
            freqCol = 0;
            zPrimeCol = 1;
            zDoublePrimeCol = 2;
            zModCol = 3;
            negateImZ = 1
            if 'Frequency' in dataLineString:
                freqCol = 1;
                zPrimeCol = 2;
                zDoublePrimeCol = 3;
                zModCol = 4
                if ("-Z''" in dataLineString):
                    print("Data has -Z''; negating zDoublePrime column.")
                    negateImZ = -1
            while dataLineString:
                if (len(dataLineString) > 2) and (dataLineString[0] != '#') and (
                        re.match(r'^\d*.\d*\t\d*.\d*\t\d*.\d*', dataLineString) is not None):
                    lineList = dataLineString.split("\t")
                    lineList[-1] = lineList[-1].strip()
                    self.activeData.rawFrequency.append(float(lineList[freqCol]))
                    self.activeData.rawzPrime.append(float(lineList[zPrimeCol]))
                    self.activeData.rawZdoublePrime.append(negateImZ * float(lineList[zDoublePrimeCol]))
                    self.activeData.rawzMod.append(float(lineList[zModCol]))
                dataLineString = self.currentFile.readline()
            self.currentFile.close()
            for i, real in enumerate(self.activeData.rawzPrime):
                z_complex = real + 1j * self.activeData.rawZdoublePrime[i]
                self.activeData.rawZExperimentalComplex.append(z_complex)
                self.activeData.rawmodZExperimentalComplex.append(abs(z_complex))
            self.activeData.rawFrequency = np.array(self.activeData.rawFrequency)
        elif self.currentFileName.get().endswith('.xlsx') or self.currentFileName.get().endswith('.xls'):
            xlsx = openpyxl.load_workbook(self.currentDataDir.IE.get() + self.currentFileName.get(), data_only=True)
            sheet1 = xlsx.active
            data = []
            for row in sheet1.iter_rows(values_only=True):
                data.append(list(row))
            del xlsx
            self.activeData.rawFrequency = [row[1] for row in data[1:]]
            self.activeData.rawzPrime = [row[2] for row in data[1:]]
            self.activeData.rawZdoublePrime = [row[3] for row in data[1:]]
            self.activeData.rawzMod = [row[4] for row in data[1:]]
            for i, real in enumerate(self.activeData.rawzPrime):
                z_complex = real + 1j * self.activeData.rawZdoublePrime[i]
                self.activeData.rawZExperimentalComplex.append(z_complex)
                self.activeData.rawmodZExperimentalComplex.append(abs(z_complex))
            self.activeData.rawFrequency = np.array(self.activeData.rawFrequency)

        for i in range(len(self.activeData.rawzPrime)):
            self.activeData.rawPhase.append(
                (180 / np.pi) * np.arctan(self.activeData.rawZdoublePrime[i] / self.activeData.rawzPrime[i]))

        print("Done loading file.")

    def ChopFreq(self):
        tempFreq = []
        self.activeData.frequency = np.array([])
        low_bound = float(self.frequencyRange.OE.get())
        up_bound = float(self.frequencyRange.IE.get())
        for freq in self.activeData.rawFrequency:
            if (freq > low_bound) and (freq < up_bound):
                tempFreq.append(freq)
        self.activeData.frequency = np.array(tempFreq)
        minIndex = self.activeData.rawFrequency.tolist().index(self.activeData.frequency[0])
        maxIndex = self.activeData.rawFrequency.tolist().index(self.activeData.frequency[-1])
        self.activeData.zPrime = self.activeData.rawzPrime[minIndex:maxIndex + 1]
        self.activeData.ZdoublePrime = self.activeData.rawZdoublePrime[minIndex:maxIndex + 1]
        self.activeData.zMod = self.activeData.rawzMod[minIndex:maxIndex + 1]
        self.activeData.modZExperimentalComplex = self.activeData.rawmodZExperimentalComplex[minIndex:maxIndex + 1]
        self.activeData.phase = self.activeData.rawPhase[minIndex:maxIndex + 1]

    def PerformSim(self):
        self.LoadSElectedFile()
        if len(self.activeData.rawzPrime) == 0:
            tkMessageBox.showinfo("Error!", "No data file loaded or data is in incorrect format")
            return
        else:
            self.ChopFreq()
            # We now fix Lwire and Theta to 0.
            params = [0,
                      float(self.Rmem.IE.get()),
                      float(self.Rcl.IE.get()),
                      float(self.Qdl.IE.get()),
                      float(self.Phi.IE.get()),
                      0]
            self.CreateFigures(params, 'sim')
            print("Model Selection Made:", self.model_selection.get())
            simResiduals = self.funcCost(params)
            self.resPercentData = np.sum(simResiduals / self.activeData.zMod * 100) / len(simResiduals)
            self.avgResPer.AVGRESPER.config(state='normal')
            self.avgResPer.AVGRESPER.delete(0, END)
            self.avgResPer.AVGRESPER.insert(0, '%5.4f' % self.resPercentData)
            self.avgResPer.AVGRESPER.config(state='readonly')

    def PerformFit(self):
        self.LoadSElectedFile()
        print('\nFitting file: ' + self.currentFileName.get() + '\n')
        if len(self.activeData.rawzPrime) == 0:
            tkMessageBox.showinfo("Error!", "No data file loaded or data is in incorrect format")
            return
        else:
            self.ChopFreq()
            # Form free parameter vector: [Rmem, Rcl, Qdl, Phi]
            params = [float(self.Rmem.IE.get()),
                      float(self.Rcl.IE.get()),
                      float(self.Qdl.IE.get()),
                      float(self.Phi.IE.get())]
            # Use bounds: Rmem, Rcl, Qdl >=0, and Phi between 0 and 1.
            finalOutput = scipy.optimize.least_squares(self.funcCost, params,
                                                       bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, 1]),
                                                       max_nfev=50000, method='trf', xtol=1e-11,
                                                       ftol=1e-11, gtol=1e-11, verbose=1)
            # Reconstruct full parameter vector with Lwire=0, Theta=0.
            self.finalParams = [0, finalOutput.x[0], finalOutput.x[1], finalOutput.x[2], finalOutput.x[3], 0]

            # Estimate standard errors using Gauss-Newton approximation
            Jacob = finalOutput.jac
            try:
                hessian_approx = Jacob.T @ Jacob
                cov_matrix = np.linalg.inv(hessian_approx)
                print("Jacobian is full-rank. Using regular inverse.")
            except np.linalg.LinAlgError:
                print("Warning: Hessian matrix is singular — using pseudo-inverse instead.")
                tkMessageBox.showwarning("Warning",
                                         "Jacobian matrix singular. Some parameter errors may be unreliable.")
                cov_matrix = np.linalg.pinv(hessian_approx)
            dof = finalOutput.fun.shape[0] - len(finalOutput.x)
            if dof <= 0:
                sigma_squared = 0
                print("Warning: Degrees of freedom <= 0.")
                tkMessageBox.showwarning("Warning", "Degrees of freedom <= 0. Check data points.")
            else:
                sigma_squared = (finalOutput.fun.T @ finalOutput.fun) / dof
            estVars = np.diag(cov_matrix * sigma_squared)
            self.standardDeviation = np.sqrt(estVars)
            self.L2NormOfRes = np.sqrt(np.sum(finalOutput.fun ** 2))
            self.resPercentData = np.sum(finalOutput.fun / self.activeData.zMod * 100) / finalOutput.fun.shape[0]
            print("Fit to: " + self.activeData.dataNameNoExt)
            self.percentSigma = self.standardDeviation / finalOutput.x * 100

            # Output fit results (only free parameters; Lwire and Theta remain 0)
            self.Rmem.OE.config(state='normal')
            self.Rmem.OE.delete(0, END)
            self.Rmem.OE.insert(0, '%5.8f' % (finalOutput.x[0]))
            self.Rmem.OE.config(state='readonly')

            self.Rcl.OE.config(state='normal')
            self.Rcl.OE.delete(0, END)
            self.Rcl.OE.insert(0, '%5.8f' % (finalOutput.x[1]))
            self.Rcl.OE.config(state='readonly')

            self.Qdl.OE.config(state='normal')
            self.Qdl.OE.delete(0, END)
            self.Qdl.OE.insert(0, '%5.8f' % (finalOutput.x[2]))
            self.Qdl.OE.config(state='readonly')

            self.Phi.OE.config(state='normal')
            self.Phi.OE.delete(0, END)
            self.Phi.OE.insert(0, '%5.8f' % (finalOutput.x[3]))
            self.Phi.OE.config(state='readonly')

            self.Lwire.OE.config(state='normal')
            self.Lwire.OE.delete(0, END)
            self.Lwire.OE.insert(0, '0')
            self.Lwire.OE.config(state='readonly')

            self.Theta.OE.config(state='normal')
            self.Theta.OE.delete(0, END)
            self.Theta.OE.insert(0, '0')
            self.Theta.OE.config(state='readonly')

            self.Rmem.OESD.config(state='normal')
            self.Rmem.OESD.delete(0, END)
            self.Rmem.OESD.insert(0, '%5.8f' % (self.standardDeviation[0]))
            self.Rmem.OESD.config(state='readonly')

            self.Rcl.OESD.config(state='normal')
            self.Rcl.OESD.delete(0, END)
            self.Rcl.OESD.insert(0, '%5.8f' % (self.standardDeviation[1]))
            self.Rcl.OESD.config(state='readonly')

            self.Qdl.OESD.config(state='normal')
            self.Qdl.OESD.delete(0, END)
            self.Qdl.OESD.insert(0, '%5.8f' % (self.standardDeviation[2]))
            self.Qdl.OESD.config(state='readonly')

            self.Phi.OESD.config(state='normal')
            self.Phi.OESD.delete(0, END)
            self.Phi.OESD.insert(0, '%5.8f' % (self.standardDeviation[3]))
            self.Phi.OESD.config(state='readonly')

            self.Lwire.OESD.config(state='normal')
            self.Lwire.OESD.delete(0, END)
            self.Lwire.OESD.insert(0, '0')
            self.Lwire.OESD.config(state='readonly')

            self.Theta.OESD.config(state='normal')
            self.Theta.OESD.delete(0, END)
            self.Theta.OESD.insert(0, '0')
            self.Theta.OESD.config(state='readonly')

            self.Rmem.OESDP.config(state='normal')
            self.Rmem.OESDP.delete(0, END)
            self.Rmem.OESDP.insert(0, '%5.4f' % (self.percentSigma[0]))
            self.Rmem.OESDP.config(state='readonly')

            self.Rcl.OESDP.config(state='normal')
            self.Rcl.OESDP.delete(0, END)
            self.Rcl.OESDP.insert(0, '%5.4f' % (self.percentSigma[1]))
            self.Rcl.OESDP.config(state='readonly')

            self.Qdl.OESDP.config(state='normal')
            self.Qdl.OESDP.delete(0, END)
            self.Qdl.OESDP.insert(0, '%5.4f' % (self.percentSigma[2]))
            self.Qdl.OESDP.config(state='readonly')

            self.Phi.OESDP.config(state='normal')
            self.Phi.OESDP.delete(0, END)
            self.Phi.OESDP.insert(0, '%5.4f' % (self.percentSigma[3]))
            self.Phi.OESDP.config(state='readonly')

            self.avgResPer.AVGRESPER.config(state='normal')
            self.avgResPer.AVGRESPER.delete(0, END)
            self.avgResPer.AVGRESPER.insert(0, '%5.4f' % self.resPercentData)
            self.avgResPer.AVGRESPER.config(state='readonly')

            self.CreateFigures(self.finalParams, 'fit')

    def CreateFigures(self, params, fitOrSim):
        if fitOrSim == 'fit':
            graphLabel = 'Full complex fit: '
        elif fitOrSim == 'sim':
            graphLabel = 'Simulated using: '
        else:
            graphLabel = ''
        plt.close('all')
        gs0 = gridspec.GridSpec(1, 2)
        gs00 = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs0[0])
        gs01 = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs0[1])
        f = plt.figure(1, figsize=[8, 3.5], tight_layout='true')

        # Nyquist plot
        nyGraph = plt.Subplot(f, gs01[:, :])
        f.add_subplot(nyGraph)
        nyGraph.plot(self.activeData.zPrime, self.activeData.ZdoublePrime, 'bo', ls='--', markersize=2, linewidth=1,
                     label='data: ' + self.activeData.dataNameNoExt)
        if self.model_selection.get() == self.eis_model[0]:
            nyGraph.plot(self.funcreal(params), self.funcImg(params), 'ro', markersize=2,
                         label='\n%s\nRmem=%5.8f\nRcl=%5.8f\nQdl=%5.5f\nphi=%5.5f' % (
                             graphLabel, params[1], params[2], params[3], params[4] if len(params) > 4 else params[3]))
        elif self.model_selection.get() == self.eis_model[1]:
            nyGraph.plot(self.funcreal_l(params), self.funcImg_l(params), 'ro', markersize=2,
                         label=graphLabel)
        elif self.model_selection.get() == self.eis_model[2]:
            nyGraph.plot(self.funcreal_s(params), self.funcImg_s(params), 'ro', markersize=2,
                         label=graphLabel)
        plt.gca().invert_yaxis()
        plt.xticks(rotation=20)
        plt.xlabel("Re(Z)")
        plt.ylabel("Im(Z)")
        plt.legend(loc=2, fontsize=6)

        # Phase vs frequency
        phaseGraph = plt.Subplot(f, gs00[-4, :3])
        f.add_subplot(phaseGraph)
        phaseGraph.plot(self.activeData.frequency, self.activeData.phase, 'bo', ls='--', markersize=2, linewidth=1)
        phaseGraph.plot(self.activeData.frequency, self.funcPhase(params), 'ro', markersize=2)
        plt.ylabel("Phase")
        plt.gca().set_xscale('log')

        # |Z| vs frequency
        modZgraph = plt.Subplot(f, gs00[-3, :3])
        f.add_subplot(modZgraph)
        modZgraph.plot(self.activeData.frequency, self.activeData.modZExperimentalComplex, 'bo', ls='--', markersize=2,
                       linewidth=1)
        modZgraph.plot(self.activeData.frequency, self.funcAbs(params), 'ro', markersize=2)
        plt.ylabel("|Z|")
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')

        # Im(Z) vs frequency
        imZgraph = plt.Subplot(f, gs00[-2, :3])
        f.add_subplot(imZgraph)
        imZgraph.plot(self.activeData.frequency, self.activeData.ZdoublePrime, 'bo', ls='--', markersize=2, linewidth=1)
        if self.model_selection.get() == self.eis_model[0]:
            imZgraph.plot(self.activeData.frequency, self.funcImg(params), 'ro', markersize=2)
        elif self.model_selection.get() == self.eis_model[1]:
            imZgraph.plot(self.activeData.frequency, self.funcImg_l(params), 'ro', markersize=2)
        elif self.model_selection.get() == self.eis_model[2]:
            imZgraph.plot(self.activeData.frequency, self.funcImg_s(params), 'ro', markersize=2)
        plt.ylabel("Im(Z)")
        plt.gca().set_xscale('log')
        plt.gca().set_xticks([])

        # Re(Z) vs frequency
        reZgraph = plt.Subplot(f, gs00[-1, :3])
        f.add_subplot(reZgraph)
        reZgraph.plot(self.activeData.frequency, self.activeData.zPrime, 'bo', ls='--', markersize=2, linewidth=1)
        if self.model_selection.get() == self.eis_model[0]:
            reZgraph.plot(self.activeData.frequency, self.funcreal(params), 'ro', markersize=2)
        elif self.model_selection.get() == self.eis_model[1]:
            reZgraph.plot(self.activeData.frequency, self.funcreal_l(params), 'ro', markersize=2)
        elif self.model_selection.get() == self.eis_model[2]:
            reZgraph.plot(self.activeData.frequency, self.funcreal_s(params), 'ro', markersize=2)
        plt.xlabel("Frequency")
        plt.ylabel("Re(Z)")
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')

        # Draw the figure in the Tkinter frame
        for widget in self.plotFrame.winfo_children():
            widget.destroy()
        for widget in self.plotFrameToolBar.winfo_children():
            widget.destroy()

        dataPlot = FigureCanvasTkAgg(f, master=self.plotFrame)
        dataPlot.draw()
        dataPlot.get_tk_widget().grid(row=0, sticky=N + S + E + W)
        toolbar = NavigationToolbar2Tk(dataPlot, self.plotFrameToolBar)
        toolbar.update()
        dataPlot.get_tk_widget().grid(row=0, sticky=W + S)

        print("Plotting complete.")

    def KILLALL(self):
        for widget in self.plotFrame.winfo_children():
            widget.destroy()
        print("All plots cleared.")

    def SaveData(self):
        if (len(self.currentFileName.get()) == 0) or (self.currentFileName.get() == '---'):
            print("No data loaded")
            tkMessageBox.showinfo("Error!", "No data file loaded")
            return
        dataOutFile = open(self.currentDataDir.IE.get() + self.activeData.dataNameNoExt + '_fit.txt', "w+")
        i = 0
        dataOutFile.write('#Fitted model at fitting frequencies:\n#Frequency\tRe(Z)\tIm(Z)\t|Z|\n')
        for real in self.realFinalModel:
            dataOutFile.write(str(self.activeData.frequency[i]) + '\t' + str(self.realFinalModel[i]) + '\t' +
                              str(self.imagFinalModel[i]) + '\t' + str(self.zModFinalModel[i]) + '\n')
            i += 1
        dataOutFile.write("\n# Fit Results:\n")
        dataOutFile.write("# Rmem = %5.8f\n# Rcl = %5.8f\n# Qdl = %5.8f\n# phi = %5.8f\n" %
                          (self.finalParams[1], self.finalParams[2], self.finalParams[3], self.finalParams[4]))
        dataOutFile.write("# Lwire = 0\n# Theta = 0\n")
        dataOutFile.write("# Avg. |Z| residual %% = " + str(self.resPercentData))
        dataOutFile.close()
        print("Saved fitted data in: " + self.currentDataDir.IE.get() + self.activeData.dataNameNoExt + '_fit.txt')

    # JPcoth function deals with coth for complex numbers.
    def JPcoth(self, x):
        return (np.exp(x) + np.exp(-x)) / (np.exp(x) - np.exp(-x))

    # The cost function returns the combined error (real and imaginary parts)
    def funcCost(self, params):
        # Reconstruct the full parameter vector:
        # Full params = [0, Rmem, Rcl, Qdl, Phi, 0]
        full_params = [0, params[0], params[1], params[2], params[3], 0]
        if self.model_selection.get() == self.eis_model[0]:
            diff = (self.funcreal(full_params) - np.array(self.activeData.zPrime)) ** 2 + \
                   (self.funcImg(full_params) - np.array(self.activeData.ZdoublePrime)) ** 2
            return np.sqrt(diff)
        elif self.model_selection.get() == self.eis_model[1]:
            diff = (self.funcreal_l(full_params) - np.array(self.activeData.zPrime)) ** 2 + \
                   (self.funcImg_l(full_params) - np.array(self.activeData.ZdoublePrime)) ** 2
            return np.sqrt(diff)
        elif self.model_selection.get() == self.eis_model[2]:
            diff = (self.funcreal_s(full_params) - np.array(self.activeData.zPrime)) ** 2 + \
                   (self.funcImg_s(full_params) - np.array(self.activeData.ZdoublePrime)) ** 2
            return np.sqrt(diff)
        else:
            print("Error in Model Selection")

    # Model functions for the Transmission Line (with inductive term removed)
    def funcreal(self, param):
        Rmem = param[1]
        Rcl = param[2]
        Qdl = param[3]
        Phi = param[4]
        omega = 1j * 2 * np.pi * self.activeData.frequency
        Z = Rmem + np.sqrt(Rcl / (Qdl * (omega ** Phi))) * self.JPcoth(np.sqrt(Rcl * Qdl * (omega ** Phi)))
        return np.real(Z)

    def funcImg(self, param):
        Rmem = param[1]
        Rcl = param[2]
        Qdl = param[3]
        Phi = param[4]
        omega = 1j * 2 * np.pi * self.activeData.frequency
        Z = Rmem + np.sqrt(Rcl / (Qdl * (omega ** Phi))) * self.JPcoth(np.sqrt(Rcl * Qdl * (omega ** Phi)))
        return np.imag(Z)

    # 1-D Linear Diffusion model functions
    def funcreal_l(self, param):
        Rmem = param[1]
        Rcl = param[2]
        Qdl = param[3]
        Phi = param[4]
        omega = 1j * 2 * np.pi * self.activeData.frequency
        Z = Rmem + Rcl * (Rcl * (Qdl * (omega ** Phi))) ** (-0.5) * self.JPcoth(np.sqrt(Rcl * Qdl * (omega ** Phi)))
        return np.real(Z)

    def funcImg_l(self, param):
        Rmem = param[1]
        Rcl = param[2]
        Qdl = param[3]
        Phi = param[4]
        omega = 1j * 2 * np.pi * self.activeData.frequency
        Z = Rmem + Rcl * (Rcl * (Qdl * (omega ** Phi))) ** (-0.5) * self.JPcoth(np.sqrt(Rcl * Qdl * (omega ** Phi)))
        return np.imag(Z)

    # 1-D Spherical Diffusion model functions
    def funcreal_s(self, param):
        Rmem = param[1]
        Rcl = param[2]
        Qdl = param[3]
        Phi = param[4]
        omega = 1j * 2 * np.pi * self.activeData.frequency
        Z = Rmem + Rcl / (np.sqrt(Rcl * Qdl * (omega ** Phi)) * self.JPcoth(np.sqrt(Rcl * Qdl * (omega ** Phi))) - 1)
        return np.real(Z)

    def funcImg_s(self, param):
        Rmem = param[1]
        Rcl = param[2]
        Qdl = param[3]
        Phi = param[4]
        omega = 1j * 2 * np.pi * self.activeData.frequency
        Z = Rmem + Rcl / (np.sqrt(Rcl * Qdl * (omega ** Phi)) * self.JPcoth(np.sqrt(Rcl * Qdl * (omega ** Phi))) - 1)
        return np.imag(Z)

    def funcAbs(self, param):
        if self.model_selection.get() == self.eis_model[0]:
            return abs(self.funcreal(param) + 1j * self.funcImg(param))
        elif self.model_selection.get() == self.eis_model[1]:
            return abs(self.funcreal_l(param) + 1j * self.funcImg_l(param))
        elif self.model_selection.get() == self.eis_model[2]:
            return abs(self.funcreal_s(param) + 1j * self.funcImg_s(param))

    def funcPhase(self, param):
        if self.model_selection.get() == self.eis_model[0]:
            return 180 / np.pi * np.arctan(self.funcImg(param) / self.funcreal(param))
        elif self.model_selection.get() == self.eis_model[1]:
            return 180 / np.pi * np.arctan(self.funcImg_l(param) / self.funcreal_l(param))
        elif self.model_selection.get() == self.eis_model[2]:
            return 180 / np.pi * np.arctan(self.funcImg_s(param) / self.funcreal_s(param))

    def ChopFreq(self):
        """
        Revised ChopFreq: use a boolean mask so that frequency and all corresponding arrays
        (zPrime, ZdoublePrime, zMod, modZExperimentalComplex, phase) have the same length.
        """
        try:
            low_bound = float(self.frequencyRange.OE.get())
            up_bound = float(self.frequencyRange.IE.get())
        except ValueError:
            print("Frequency bounds must be numeric.")
            return
        raw_freq = np.array(self.activeData.rawFrequency)
        mask = (raw_freq > low_bound) & (raw_freq < up_bound)
        self.activeData.frequency = raw_freq[mask]
        self.activeData.zPrime = np.array(self.activeData.rawzPrime)[mask]
        self.activeData.ZdoublePrime = np.array(self.activeData.rawZdoublePrime)[mask]
        self.activeData.zMod = np.array(self.activeData.rawzMod)[mask]
        self.activeData.modZExperimentalComplex = np.array(self.activeData.rawmodZExperimentalComplex)[mask]
        self.activeData.phase = np.array(self.activeData.rawPhase)[mask]

    def funcCostFree(self, free_params):
        """
        Wrapper that uses only the free parameters:
          free_params = [Rmem, Rcl, Qdl, Phi]
        Reconstruct the full parameter vector as [0, Rmem, Rcl, Qdl, Phi, 0]
        (with Lwire and Theta fixed to 0) and return the cost computed by funcCost.
        """
        full_params = [0] + list(free_params) + [0]
        return self.funcCost(full_params)

    def fit_with_free_params(self):
        """
        Performs a least-squares fit on the current chopped data (assumed set in self.activeData.frequency
        and corresponding arrays) using only free parameters. Returns the cost computed as the average percent
        standard error (SE) for Rmem and Rcl.
        """
        free_params = [
            float(self.Rmem.IE.get()),
            float(self.Rcl.IE.get()),
            float(self.Qdl.IE.get()),
            float(self.Phi.IE.get())
        ]
        result = scipy.optimize.least_squares(
            self.funcCostFree, free_params,
            bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, 1]),
            max_nfev=20000, method='trf',
            xtol=1e-11, ftol=1e-11, gtol=1e-11, verbose=0
        )
        J = result.jac
        try:
            cov = np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(J.T @ J)
        s2 = (result.fun.T @ result.fun) / (len(result.fun) - len(free_params))
        estVars = np.diag(cov * s2)
        std = np.sqrt(estVars)
        percentSE = std / result.x * 100  # percent SE for free parameters
        # Cost is the average percent SE for Rmem (index 0) and Rcl (index 1)
        cost = (percentSE[0] + percentSE[1]) / 2.0
        return cost

    def fit_chopped_window(self, upper_idx, lower_idx, chopped_freq, N_initial):
        """
        Given a candidate sub-window defined by candidate indices [upper_idx, lower_idx]
        (in the chopped frequency array), extract the corresponding candidate subarrays for
        frequency and the measured data. Temporarily set self.activeData to these candidate arrays,
        perform the free parameter fit, and compute the candidate cost.
        A penalty for reduced data size is applied.
        """
        temp_freq = chopped_freq[upper_idx: lower_idx + 1]
        temp_zPrime = np.array(self.activeData.zPrime)[upper_idx: lower_idx + 1]
        temp_ZdoublePrime = np.array(self.activeData.ZdoublePrime)[upper_idx: lower_idx + 1]
        temp_zMod = np.array(self.activeData.zMod)[upper_idx: lower_idx + 1]
        temp_modZ = np.array(self.activeData.modZExperimentalComplex)[upper_idx: lower_idx + 1]
        temp_phase = np.array(self.activeData.phase)[upper_idx: lower_idx + 1]

        # Backup current chopped data.
        orig_freq = self.activeData.frequency.copy()
        orig_zPrime = np.array(self.activeData.zPrime).copy()
        orig_ZdoublePrime = np.array(self.activeData.ZdoublePrime).copy()
        orig_zMod = np.array(self.activeData.zMod).copy()
        orig_modZ = np.array(self.activeData.modZExperimentalComplex).copy()
        orig_phase = np.array(self.activeData.phase).copy()

        # Set candidate data.
        self.activeData.frequency = temp_freq
        self.activeData.zPrime = temp_zPrime.tolist()
        self.activeData.ZdoublePrime = temp_ZdoublePrime.tolist()
        self.activeData.zMod = temp_zMod.tolist()
        self.activeData.modZExperimentalComplex = temp_modZ.tolist()
        self.activeData.phase = temp_phase.tolist()

        cost = self.fit_with_free_params()

        # Restore original data.
        self.activeData.frequency = orig_freq
        self.activeData.zPrime = orig_zPrime.tolist()
        self.activeData.ZdoublePrime = orig_ZdoublePrime.tolist()
        self.activeData.zMod = orig_zMod.tolist()
        self.activeData.modZExperimentalComplex = orig_modZ.tolist()
        self.activeData.phase = orig_phase.tolist()

        # Candidate window size and penalty:
        N_candidate = len(temp_freq)
        # Apply a penalty factor that increases steeply if candidate window size is much smaller.
        penalty_exponent = 1.5  # Adjust exponent as needed.
        penalty = max(1, (float(N_initial) / N_candidate) ** penalty_exponent)
        adjusted_cost = cost * penalty
        return adjusted_cost

    def auto_freq_window_grid_search(self):
        """
        Single-click grid search for the optimal frequency window.
        1. First, call ChopFreq() so that self.activeData.frequency contains only data within the initial GUI bounds.
        2. Then, set up a small grid of candidate windows (by choosing candidate indices for upper and lower boundaries).
        3. For each candidate window, compute an adjusted cost (cost from the free-parameter fit times a penalty for window size reduction).
        4. Choose the candidate with the lowest adjusted cost and update the GUI.
        """
        # First, chop the data.
        self.ChopFreq()
        chopped_freq = self.activeData.frequency.copy()
        npts = len(chopped_freq)
        if npts < 10:
            print("Not enough data points in the initial window.")
            return
        N_initial = npts
        # Define candidate indices for the upper boundary and lower boundary in the chopped array.
        # For example, try a few candidate values:
        cand_upper_indices = sorted(list({0, npts // 4, npts // 2}))
        cand_lower_indices = sorted(list({npts - 1, 3 * npts // 4, npts // 2}))

        best_cost = float('inf')
        best_upper_idx = None
        best_lower_idx = None

        for ui in cand_upper_indices:
            for li in cand_lower_indices:
                if ui >= li:
                    continue
                candidate_cost = self.fit_chopped_window(ui, li, chopped_freq, N_initial)
                print(
                    f"Candidate window: {chopped_freq[ui]:.2f} Hz to {chopped_freq[li]:.2f} Hz, cost: {candidate_cost:.2f}%")
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_upper_idx = ui
                    best_lower_idx = li

        if best_upper_idx is None or best_lower_idx is None:
            # No candidate improved on the initial window; use the entire chopped window.
            best_upper_idx = 0
            best_lower_idx = npts - 1
            best_cost = self.fit_chopped_window(best_upper_idx, best_lower_idx, chopped_freq, N_initial)

        optimal_upper = chopped_freq[best_upper_idx]
        optimal_lower = chopped_freq[best_lower_idx]
        self.frequencyRange.IE.delete(0, END)
        self.frequencyRange.IE.insert(0, f"{optimal_upper:.2f}")
        self.frequencyRange.OE.delete(0, END)
        self.frequencyRange.OE.insert(0, f"{optimal_lower:.2f}")
        print(
            f"Optimal window found: {optimal_upper:.2f} Hz to {optimal_lower:.2f} Hz with adjusted cost {best_cost:.2f}%")


class Param():
    def __init__(self):
        self.IE = Entry()
        self.OE = Entry()
        self.OESD = Entry()
        self.OESDP = Entry()
        self.AVGRESPER = Entry()


class Data():
    def __init__(self):
        self.dataName = ''
        self.dataNameNoExt = ''
        self.zPrime = []
        self.ZdoublePrime = []
        self.zMod = []
        self.modZExperimentalComplex = []
        self.frequency = np.array([])
        self.phase = []
        self.rawzPrime = []
        self.rawZdoublePrime = []
        self.rawzMod = []
        self.rawmodZExperimentalComplex = []
        self.rawFrequency = []
        self.rawPhase = []


def on_closing():
    if tkMessageBox.askokcancel("Quit", "Do you want to quit?"):
        app.KILLALL()
        root.destroy()
        os._exit(0)


root = Tk()
app = OSIF(root)
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
