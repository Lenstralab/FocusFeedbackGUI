import win32com.client
import pythoncom
import numpy as np
from re import search
from time import sleep
from threading import get_ident
from functools import partial
from dataclasses import dataclass
from enum import Enum

if __package__ == '':
    from utilities import qthread, errwrap
else:
    from .utilities import qthread, errwrap

#global property to keep track of indices of drawings in ZEN across threads
zendrawinglist = []
lastpiezopos = None

class cst(Enum):
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
    # None = 0
    OpenArrow = 11
    OpenBezier = 9
    OpenPolyLine = 7
    Palette = 14
    Rectangle = 4
    ScaleBar = 3
    Select = 1
    Text = 13


class zen:
    def __init__(self, EventHandler=None):
        self.EventHandler = EventHandler
        self._ZEN_VBA = {}
        self._ID = {}
        self.ProgressCoord = self.VBA.Lsm5.ExternalDsObject().ScanController().GetProgressCoordinates
        self.LastProgressCoord = self.ProgressCoord()
        self.SetAnalogMode(True)
        self.SetExtendedRange(False)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kargs):
        self.close()

    def wait(self, clg, callback):
        def fun():
            while not self.ready and not clg.stop and not clg.quit:
                sleep(0.1)
        self.wait_thread = qthread(fun, callback)

    def _wait_ready(self, *args, **kwargs):
        self.wait.quit()
        self.wait.wait()
        self.wait_callback(*args, **kwargs)

    def close(self):
        id = get_ident()
        if id in self._ZEN_VBA:
            ZEN, VBA = self._ZEN_VBA.pop(id)
            del ZEN, VBA
        if id in self._ID:
            ID = self._ID.pop(id)
            del ID

        # if len(self._ID)==1:
        #     self.SetAnalogMode(False)
        #     print('closing last zen')
        # if id in self._ID:
        #     self._ID[id][0] = None
        #     self._ID[id][1] = None
        #     self._ZEN.pop(id)

    @property
    def ZEN(self):
        return self.reconnect()[0]

    @property
    def VBA(self):
        return self.reconnect()[1]

    def reconnect(self):
        id = get_ident()
        if not id in self._ZEN_VBA:
            pythoncom.CoInitialize()
            success = False
            for ZENid, VBAid in self._ID.values(): #First try comarshalling
                try:
                    ZEN = win32com.client.Dispatch(pythoncom.CoGetInterfaceAndReleaseStream(ZENid, pythoncom.IID_IDispatch))
                    VBA = win32com.client.Dispatch(pythoncom.CoGetInterfaceAndReleaseStream(VBAid, pythoncom.IID_IDispatch))
                    success = True
                    break
                except:
                    continue
            if not success:
                ZEN = win32com.client.Dispatch('Zeiss.Micro.AIM.ApplicationInterface.ApplicationInterface')
                if self.EventHandler is None:
                    VBA = win32com.client.Dispatch('Lsm5Vba.Application')
                else:
                    VBA = win32com.client.DispatchWithEvents('Lsm5Vba.Application', self.EventHandler(self))
                self._ID[id] = (pythoncom.CoMarshalInterThreadInterfaceInStream(pythoncom.IID_IDispatch, ZEN),
                                pythoncom.CoMarshalInterThreadInterfaceInStream(pythoncom.IID_IDispatch, VBA))
            self._ZEN_VBA[id] = (ZEN, VBA)
        return self._ZEN_VBA[id]

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

    @property
    def CurrentDoc(self):
        return self.VBA.Lsm5.DsRecordingActiveDocObject

    def EnableEvent(self, event, doc=None):
        doc = doc or self.CurrentDoc
        for i in range(25):
            if not doc is None:
                break
            sleep(0.2)
            doc = self.CurrentDoc
        doc.EnableImageWindowEvent(cst[event].value, True)

    def DisableEvent(self, event, doc=''):
        if doc == '':
            doc = self.CurrentDoc
        if not doc is None:
            doc.EnableImageWindowEvent(cst[event].value, False)

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

    def SaveDouble(self, name, value):
        self.ZEN.SetDouble(name, value)

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
        Sx = 2*int((Sx+1)/2)  # Ensure evenness
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
    def ChannelColorsInt(self):
        ScanDoc = self.VBA.Lsm5.DsRecordingActiveDocObject
        return [ScanDoc.ChannelColor(channel) for channel in range(self.nChannels)]

    @property
    def ChannelColorsHex(self):
        color = []
        for cci in self.ChannelColorsInt:
            h = np.base_repr(cci, 16, 6)[-6:]
            color.append('#'+h[4:]+h[2:4]+h[:2])
        return color

    @property
    def ChannelColorsRGB(self):
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
        ScanDoc = self.VBA.Lsm5.DsRecordingActiveDocObject
        return self.ZEN.GUI.Acquisition.IsExperimentRunning.value and ScanDoc.GetDimensionTime() > 1\
            and ScanDoc.GetDimensionZ() <= 1

    @property
    def isZStack(self):
        return self.VBA.Lsm5.DsRecordingActiveDocObject.GetDimensionZ() > 1

    @property
    def isTimeSeries(self):
        return self.VBA.Lsm5.DsRecordingActiveDocObject.GetDimensionTime() > 1

    @property
    def TimeInterval(self):
        # in s
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

    def DrawRectangle(self, X, Y, Sx, Sy, Color=16777215, LineWidth=2, index=None):
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
        # in um
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
        FS.FilterSetPosition = position + 1


class EventHandlerMetaClass(type):
    """
    A meta class for event handlers that don't repsond to all events.
    Without this an error would be raised by win32com when it tries
    to call an event handler method that isn't defined by the event
    handler instance.
    """
    @staticmethod
    def null_event_handler(event, *args, **kwargs):
        print(('Unhandled event {}'.format(event), args, kwargs))
        return None

    def __new__(mcs, name, bases, dict):
        # Construct the new class.
        cls = type.__new__(mcs, name, bases, dict)

        # Create dummy methods for any missing event handlers.
        cls._dispid_to_func_ = getattr(cls, "_dispid_to_func_", {})
        for dispid, name in cls._dispid_to_func_.items():
            func = getattr(cls, name, None)
            if func is None:
                setattr(cls, name, partial(EventHandlerMetaClass.null_event_handler, name))
        return cls


def EventHandler(ZEN, CLG):
    class EventHandlerCls(metaclass=EventHandlerMetaClass):
        def __init__(self):
            self.clg = CLG
            self.zen = ZEN

        @errwrap
        def OnThrowEvent(self, *args):
            if args[0] == cst.Text.value:
                return
            # print(('OnThrowEvent:', args))
            if args[0] == cst.LeftButtonDown.value and self.clg.centerbox.isChecked():
                X = self.zen.MousePosition
                FS = self.zen.FrameSize
                pxsize = self.zen.pxsize
                Pos = self.clg.conf['ROIPos']
                d = [(FS[i]/2 - X[i] + Pos[i]) * pxsize/1000 for i in range(2)]
                d[1] *= -1
                self.zen.MoveStageRel(*d)

        @errwrap
        def OnThrowPropertyEvent(self, *args):
            # print(('OnThrowProperyEvent:', args))
            if args[1] == '2C_FilterSlider' and self.clg.DLFilter != self.zen.DLFilter:
                self.clg.DLFilter = self.zen.DLFilter
                self.clg.dlf.setText(self.clg.dlfs.currentText().split(' & ')[self.zen.DLFilter])
                self.clg.chdlf.changeState(self.zen.DLFilter)
            elif args[1] == 'Stage':
                LP = self.zen.StagePos
                FS = self.zen.FrameSize
                pxsize = self.zen.pxsize
                self.clg.map.numel_data(LP[0]/1000, LP[1]/1000, FS[0]*pxsize/1e6, FS[1]*pxsize/1e6)
                self.clg.map.draw()
            elif args[1] in ('DataColorPalette', 'FramesPerStack'):
                self.clg.changeColor()
    return EventHandlerCls


@dataclass
class mem:
    z: zen
    clg: type
    prevzen: list
    curzen: list
    exprunning: bool

    def title(self):
        return '' if self.curzen[0] is None else self.curzen[0].Title()

    def checks(self):  # events that don't throw an event
        if self.title != self.z.Title:  # current document changed
            self.curzen = [self.z.CurrentDoc, False]  # ActiveDocObject, LMB enabled
            self.z.EnableEvent('LeftButtonDown')

        # draw a black rectangle in the map after an experiment started
        exprunning, self.exprunning = self.exprunning, self.z.ExperimentRunning
        if not exprunning and self.exprunning:
            LP = self.z.StagePos
            FS = self.z.FrameSize
            pxsize = self.z.pxsize
            self.clg.map.append_data_docs(LP[0] / 1000, LP[1] / 1000, FS[0] * pxsize / 1e6, FS[1] * pxsize / 1e6)
            self.clg.map.draw()

        # the LMB event is not enabled on the current doc
        if not self.curzen[1] and self.clg.centerbox.isChecked():
            if not self.prevzen[0] is None:
                self.z.DisableEvent('LeftButtonDown', self.prevzen[0])
            if not self.curzen[0] is None:
                self.z.EnableEvent('LeftButtonDown', self.curzen[0])
                self.curzen[1] = True

        if self.clg.MagStr != self.z.MagStr:  # Some things in the optical path have changed
            self.clg.MagStr = self.z.MagStr
            self.clg.confload()


def Events(clg):
    def fun():
        with zen(partial(EventHandler, CLG=clg)) as z:
            m = mem(z, clg, [None, False], [None, False], False)
            z.EnableEvent('LeftButtonDown')
            while not clg.quit:
                sleep(.01)
                pythoncom.PumpWaitingMessages()
                m.checks()
    return qthread(fun)
