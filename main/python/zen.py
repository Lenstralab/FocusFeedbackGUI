import win32com.client
import pythoncom
import numpy as np
from re import search
from time import sleep
import inspect

#global property to keep track of indices of drawings in ZEN across threads
zendrawinglist = []
lastpiezopos = None

def cst(str):
    NoButtonMouseMove = 202
    LButtonMouseMove = 201
    LeftButtonDown = 203
    LeftButtonUp = 204
    RButtonMouseMove = 207
    RightButtonDown = 208
    RightButtonUp = 209

    Circle = 6
    ClosedArrow = 12
    ClosedBezier = 10
    ClosedPolyLine = 8
    Ellipse = 5
    Line = 2
    #None = 0
    OpenArrow = 11
    OpenBezier = 9
    OpenPolyLine = 7
    Palette = 14
    Rectangle = 4
    ScaleBar = 3
    Select = 1
    Text = 13

    try:
        return eval('{}'.format(str))
    except:
        return 0

class zen:
    def __init__(self, EventHandler=None):
        self.EventHandler = EventHandler
        self.ConnectZEN()
        self.ProgressCoord = self.VBA.Lsm5.ExternalDsObject().ScanController().GetProgressCoordinates
        self.LastProgressCoord = self.ProgressCoord()
        #self.VBA.Lsm5.Hardware().CpFocus().ConnectHrzToFocus = False
        self.SetAnalogMode(True)
        self.SetExtendedRange(False)

        curf = inspect.currentframe()
        calf = inspect.getouterframes(curf, 2)

        f = open('D:\CyllensGUI\z.txt', 'a')
        f.write('caller name: {}\n'.format(calf[1][3]))
        f.close()

    @property
    def ready(self):
        return not self.VBA.Lsm5.DsRecordingActiveDocObject is None

    def SetAnalogMode(self, val=True):
        self.VBA.Lsm5.ExternalCpObject().pHardwareObjects.pHighResFoc().bSetAnalogMode(val)

    def SetExtendedRange(self, val=False):
        self.VBA.Lsm5.ExternalCpObject().pHardwareObjects.pHighResFoc().bSetExtendedZRange(val)

    @property
    def ZDL(self):
        global zendrawinglist
        return zendrawinglist

    @ZDL.setter
    def ZDL(self, val):
        global zendrawinglist
        zendrawinglist = val

    def reconnect(fun):
        def f(self, *args):
            try:
                return fun(self, *args)
            except:
                # print('reconnecting...')
                self.ZEN = None
                self.VBA = None
                self.ConnectZEN()
                return fun(self, *args)
        return f

    def EnableEvent(self, event):
        if self.ready:
            self.VBA.Lsm5.DsRecordingActiveDocObject.EnableImageWindowEvent(cst(event), True)

    def DisableEvent(self, event):
        if self.ready:
            self.VBA.Lsm5.DsRecordingActiveDocObject.EnableImageWindowEvent(cst(event), False)

    @property
    def IsBusy(self):
        if self.ready:
            return self.VBA.Lsm5.DsRecordingActiveDocObject.IsBusy()
        else:
            return True

    @property
    def MousePosition(self):
        if self.ready:
            X = self.VBA.Lsm5.DsRecordingActiveDocObject.GetCurrentMousePosition()
            return (X[5] + 1, X[4] + 1)
        else:
            return (0, 0)

    @property
    def MagStr(self):
        return '{:.0f}x{:.0f}'.format(self.ObjectiveMag, 10*self.OptovarMag)

    @property
    def ObjectiveMag(self):
        try:
            return float(search('\d+x', self.ZEN.GUI.Acquisition.AcquisitionMode.Objective.ByName).group()[:-1])
        except:
            return 0

    @property
    def ObjectiveNA(self):
        try:
            return float(search('\d\.\d+', self.ZEN.GUI.Acquisition.AcquisitionMode.Objective.ByName).group())
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
        if self.EventHandler is None:
            self.VBA = win32com.client.Dispatch('Lsm5Vba.Application')
        else:
            self.VBA = win32com.client.DispatchWithEvents('Lsm5Vba.Application', self.EventHandler)
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
    def Title(self):
        if self.ready:
            return self.VBA.Lsm5.DsRecordingActiveDocObject.Title()
        else:
            return ''

    @property
    def GetCurrentZ(self):
        return self.VBA.Lsm5.Hardware().CpFocus().Position

    @property
    def FrameSize(self):
        if self.ready:
            ScanDoc = self.VBA.Lsm5.DsRecordingActiveDocObject
            return ScanDoc.GetDimensionX(), ScanDoc.GetDimensionY()
        else:
            return 0, 0

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

    def GetFrame(self, Channel=1, X=None, Y=None, Sx=32, Sy=32):
        Sx = 2*int((Sx+1)/2) #Ensure evenness
        Sy = 2*int((Sy+1)/2)
        ScanDoc = self.VBA.Lsm5.DsRecordingActiveDocObject
        xMax = ScanDoc.GetDimensionX()
        yMax = ScanDoc.GetDimensionY()
        if X is None:
            X = int((xMax - Sx) / 2)
        else:
            X -= int(Sx/2)
        if Y is None:
            Y = int((yMax - Sy) / 2)
        else:
            Y -= int(Sy/2)

        Time = self.GetTime
        if Time > 0:
            Time -= 1
        return np.reshape(ScanDoc.GetSubregion(Channel, X, Y, 0, Time, 1, 1, 1, 1, Sx, Sy, 1, 1, 2)[0], (Sx, Sy)), Time

    @property
    def ChannelColorsHex(self):
        ScanDoc = self.VBA.Lsm5.DsRecordingActiveDocObject
        color = []
        for channel in range(self.nChannels):
            h = np.base_repr(ScanDoc.ChannelColor(channel), 16, 6)[-6:]
            color.append('#'+h[4:]+h[2:4]+h[:2])
        return color

    @property
    def ChannelColors(self):
        return [[int(hcolor[2*i+1:2*(i+1)+1], 16)/255 for i in range(3)] for hcolor in self.ChannelColorsHex]

    @property
    def ChannelNames(self):
        ScanDoc = self.VBA.Lsm5.DsRecordingActiveDocObject
        return [ScanDoc.ChannelName(i) for i in range(self.nChannels)]

    @staticmethod
    def CameraFromChannelName(ChannelName):
        return int(search('(?<=TV)\d', ChannelName).group(0))-1

    @property
    def nChannels(self):
        ScanDoc = self.VBA.Lsm5.DsRecordingActiveDocObject
        return ScanDoc.GetDimensionChannels()

    @property
    def GetTime(self):
        return self.VBA.Lsm5.ExternalDsObject().ScanController().GetProgressCoordinates()[2]

    def NextFrameWait(self, TimeOut=5):
        SleepTime = 0.02
        WaitTime = 0.0
        while (self.ProgressCoord() == self.LastProgressCoord) & (TimeOut > WaitTime) & self.ExperimentRunning():
            WaitTime += SleepTime
            sleep(SleepTime)
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

    def DrawEllipse(self, X, Y, R, E, T, Color=65025, LineWidth=2, index=None):
        if self.ready and np.all(np.isfinite((X, Y, R, E, T))):
            #X, Y, R in pixels, T in rad
            if E <= 0:
                E = 1
            Overlay = self.VBA.Lsm5.DsRecordingActiveDocObject.VectorOverlay()

            index = self.ManipulateDrawingList(Overlay, index)

            Overlay.LineWidth = LineWidth
            Overlay.Color = Color #Green

            Overlay.AddDrawingElement(5,3,(X,X+R*np.sqrt(E)*np.cos(T),X+R/np.sqrt(E)*np.sin(T)),(Y,Y-R*np.sqrt(E)*np.sin(T),Y+R/np.sqrt(E)*np.cos(T)))
            return index
        else:
            self.RemoveDrawing(index)
            return None

    def DrawRectangle(self, X, Y, Sx, Sy, Color=65025, LineWidth=2, index=None):
        if self.ready and np.all(np.isfinite((X, Y, Sx, Sy))):
            Overlay = self.VBA.Lsm5.DsRecordingActiveDocObject.VectorOverlay()

            index = self.ManipulateDrawingList(Overlay, index)

            Overlay.LineWidth = LineWidth
            Overlay.Color = Color #Green

            Overlay.AddDrawingElement(4,2,(float(X-Sx/2),float(X+Sx/2)),(float(Y-Sy/2),float(Y+Sy/2)))
            return index
        else:
            self.RemoveDrawing(index)
            return None

    def RemoveDrawings(self):
        if self.ready:
            self.VBA.Lsm5.DsRecordingActiveDocObject.VectorOverlay().RemoveAllDrawingElements()
        self.ZDL = []

    def RemoveDrawing(self, index):
        if self.ready:
            Overlay = self.VBA.Lsm5.DsRecordingActiveDocObject.VectorOverlay()
            lst = self.ZDL
            if Overlay.GetNumberDrawingElements() != len(lst):
                lst = list(range(Overlay.GetNumberDrawingElements()))
            if index in lst:
                Overlay.RemoveDrawingElement(lst.index(index))
                lst.remove(index)
                self.ZDL = lst
        else:
            self.ZDL = []

    def ManipulateDrawingList(self, Overlay, index):
        lst = self.ZDL
        if Overlay.GetNumberDrawingElements() != len(lst):
            lst = list(range(Overlay.GetNumberDrawingElements()))
        if index is not None:
            if index in lst:
                Overlay.RemoveDrawingElement(lst.index(index))
                lst.remove(index)
        else:
            if not lst:
                index = 0
            else:
                index = max(lst) + 1
        lst.append(index)
        self.ZDL = lst
        return index

    def GetPiezoPos(self):
        # in um
        p = [self.VBA.Lsm5.Hardware().CpHrz().Position]
        for i in range(200):
            sleep(0.01)
            p.append(self.VBA.Lsm5.Hardware().CpHrz().Position)
        global lastpiezopos
        lastpiezopos = np.mean(np.unique(p))
        #print('piezo: {} +- {}'.format(np.mean(np.unique(p)), np.std(np.unique(p))))

    @property
    def PiezoPos(self):
        #in um
        global lastpiezopos
        if lastpiezopos is None:
            self.GetPiezoPos()
        return lastpiezopos

    @PiezoPos.setter
    def PiezoPos(self, Z):
        # in um
        if Z > 250:
            Z = 250
        elif Z < -250:
            Z = -250
        self.VBA.Lsm5.Hardware().CpHrz().Position = Z
        global lastpiezopos
        lastpiezopos = Z
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
    def DLFilter(self):
        FS = self.VBA.Lsm5.Hardware().CpFilterSets()
        FS.Select('2C_FilterSlider')
        return FS.FilterSetPosition-1

    @DLFilter.setter
    def DLFilter(self, position):
        FS = self.VBA.Lsm5.Hardware().CpFilterSets()
        FS.Select('2C_FilterSlider')
        FS.FilterSetPosition = position+1