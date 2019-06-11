import win32com.client
import pythoncom
import numpy as np
import time
import re

class zen:
    def __init__(self):
        self.ConnectZEN()
        self.ProgressCoord = self.VBA.Lsm5.ExternalDsObject().ScanController().GetProgressCoordinates
        self.LastProgressCoord = self.ProgressCoord()
        #self.VBA.Lsm5.ExternalCpObject().pHardwareObjects.pHighResFoc().bSetAnalogMode(0)
        
    def reconnect(fun):
        def f(self,*args):
            try:
                fun(self,*args)
            except:
                print('reconnecting...')
                self.ZEN = None
                self.VBA = None
                self.ConnectZEN()
                fun(*args)
        return f

    @property
    def MagStr(self):
        return '{:.0f}x{:.0f}'.format(self.ObjectiveMag, 10*self.OptovarMag)

    @property
    def ObjectiveMag(self):
        try:
            return float(re.search('\d+x', self.ZEN.GUI.Acquisition.AcquisitionMode.Objective.ByName).group()[:-1])
        except:
            return 0

    @property
    def ObjectiveNA(self):
        try:
            return float(re.search('\d\.\d+', self.ZEN.GUI.Acquisition.AcquisitionMode.Objective.ByName).group())
        except:
            return 1

    @property
    def OptovarMag(self):
        try:
            return float(self.ZEN.GUI.Acquisition.LightPath.TubeLens.ByName[5:-1].replace(',', '.'))
        except:
            return 0

    @property
    def pxsize(self):
        return 1e9*self.ZEN.GUI.Document.DsRecordingDoc.VoxelSizeX

    def ConnectZEN(self):
        pythoncom.CoInitialize()
        self.ZEN = win32com.client.Dispatch('Zeiss.Micro.AIM.ApplicationInterface.ApplicationInterface')
        self.VBA = win32com.client.Dispatch('Lsm5Vba.Application')
        #self.AimImage = win32com.client.Dispatch('AimImage.Image')
        #self.AET = win32com.client.Dispatch('AimExperiment.TreeNode')
        self.ZENid = pythoncom.CoMarshalInterThreadInterfaceInStream(pythoncom.IID_IDispatch, self.ZEN)
        self.VBAid = pythoncom.CoMarshalInterThreadInterfaceInStream(pythoncom.IID_IDispatch, self.VBA)
        
    def DisconnectZEN(self):
        self.VBA.Lsm5.ExternalCpObject().pHardwareObjects.pHighResFoc().bSetAnalogMode(0)
        self.ZEN = None
        self.VBA = None
        
    def NewThreadZEN(self):
        pythoncom.CoInitialize()
        self.VBA = win32com.client.Dispatch(pythoncom.CoGetInterfaceAndReleaseStream(self.VBAid, pythoncom.IID_IDispatch))
        self.ZEN = win32com.client.Dispatch(pythoncom.CoGetInterfaceAndReleaseStream(self.ZENid, pythoncom.IID_IDispatch))    
		
    def SaveDouble(self, name, value):
        self.ZEN.SetDouble(name, value)
        #self.AET.Image(0).ApplicationTags().SetDoubleValue(name, value)
        #self.AimImage.ApplicationTags().SetDoubleValue(name, value)

    @property
    def FileName(self):
        return self.ZEN.GUI.Document.FullFileName.value

    @property
    def GetCurrentZ(self):
        return self.VBA.Lsm5.Hardware().CpFocus().Position

    @property
    def FrameSize(self):
        ScanDoc = self.VBA.Lsm5.DsRecordingActiveDocObject
        return ScanDoc.GetDimensionX(), ScanDoc.GetDimensionY()
        
    def GetFrameCenter(self, Channel=1, Size=32):
        Size = 2*int((Size+1)/2) #Ensure evenness
        ScanDoc = self.VBA.Lsm5.DsRecordingActiveDocObject
        xMax = ScanDoc.GetDimensionX()
        yMax = ScanDoc.GetDimensionY()

        x0 = (xMax - Size) / 2
        y0 = (yMax - Size) / 2

        Time = self.GetTime
        if Time > 0:
            Time -= 1
        return np.reshape(ScanDoc.GetSubregion(Channel, x0, y0, 0, Time, 1, 1, 1, 1, Size, Size, 1, 1, 2)[0],(Size,Size)), Time

    @property
    def GetTime(self):
        return self.VBA.Lsm5.ExternalDsObject().ScanController().GetProgressCoordinates()[2]
    
    def NextFrameWait(self, TimeOut=5):
        SleepTime = 0.02
        WaitTime = 0.0
        while (self.ProgressCoord() == self.LastProgressCoord) & (TimeOut > WaitTime) & self.ExperimentRunning():
            WaitTime += SleepTime
            time.sleep(SleepTime)
        self.LastProgressCoord = self.ProgressCoord()

    @property
    def ExperimentRunning(self):
        return self.ZEN.GUI.Acquisition.IsExperimentRunning.value

    @property
    def TimeInterval(self):
        #in s
        TI = self.ZEN.GUI.Acquisition.TimeSeries.Interval.value
        unit = self.ZEN.GUI.Acquisition.TimeSeries.IntervalTimeUnit.ByIndex
        if unit == 0:
            TI *= 60
        elif unit == 2:
            TI /= 1000

        if TI == 0:
            self.ZEN.GUI.Acquisition.Channels.Track.ByIndex = 0
            self.ZEN.GUI.Acquisition.Channels.Track.Channel.ByIndex = 0
            TI = self.ZEN.GUI.Acquisition.Channels.Track.Channel.Camera.ExposureTime.value / 1000
        return TI

    def DrawEllipse(self, X, Y, R, E, T, Color=65025, LineWidth=2):
        #X, Y, R in pixels, T in rad
        if E <= 0:
            E = 1
        Overlay = self.VBA.Lsm5.DsRecordingActiveDocObject.VectorOverlay()

        #Remove all previous drawings
        Overlay.RemoveAllDrawingElements()

        Overlay.LineWidth = LineWidth
        Overlay.Color = Color #Green

        Overlay.AddDrawingElement(5,3,(X,X+R*np.sqrt(E)*np.cos(T),X+R/np.sqrt(E)*np.sin(T)),(Y,Y-R*np.sqrt(E)*np.sin(T),Y+R/np.sqrt(E)*np.cos(T)))
        return

    @property
    def PiezoPos(self):
        #in um
        return self.VBA.Lsm5.Hardware().CpHrz().Position

    @PiezoPos.setter
    def PiezoPos(self, Z):
        # in um
        if Z > 250:
            Z = 250
        elif Z < -250:
            Z = -250
        #self.VBA.Lsm5.ExternalCpObject().pHardwareObjects.pHighResFoc().bSetAnalogMode(1)
        self.VBA.Lsm5.Hardware().CpHrz().Position = Z
        #self.VBA.Lsm5.ExternalCpObject().pHardwareObjects.pHighResFoc().bSetAnalogMode(0)
        return

    def MovePiezoRel(self, Z):
        #in um
        self.PiezoPos += Z
        return

    @property
    def StagePos(self):
        # in um
        return self.VBA.Lsm5.Hardware().CpStages().PositionX, self.VBA.Lsm5.Hardware().CpStages().PositionY

    @StagePos.setter
    def StagePos(self, P):
        X, Y = P
        if X is not None:
            Lim = self.VBA.Lsm5.Hardware().CpStages().UpperLimitX
            self.VBA.Lsm5.Hardware().CpStages().PositionX = np.clip(X, -Lim, Lim)
        if Y is not None:
            Lim = self.VBA.Lsm5.Hardware().CpStages().UpperLimitY
            self.VBA.Lsm5.Hardware().CpStages().PositionY = np.clip(Y, -Lim, Lim)

    def MoveStageRel(self, x=None, y=None):
        #in um
        X, Y = self.StagePos
        if X is not None:
            X += x
        if Y is not None:
            Y += y
        self.StagePos = (X, Y)
        return

    @property
    def duolinkcalpat(self):
        pass