import pythoncom
from functools import partial
from time import sleep
from .zen import zen, cst
from .utilities import thread, errwrap
from dataclasses import dataclass

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
            if args[0] == cst('Text'):
                return
            # print(('OnThrowEvent:', args))
            if args[0] == cst('LeftButtonDown') and self.clg.centerbox.isChecked():
                X = self.zen.MousePosition
                FS = self.zen.FrameSize
                pxsize = self.zen.pxsize
                Pos = self.clg.conf.ROIPos
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
    z: type
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
            self.clg.confopen(self.clg.conf.filename)

@thread
def events(clg):
    with zen(partial(EventHandler, CLG=clg)) as z:
        m = mem(z, clg, (None, False), (None, False), False)
        z.EnableEvent('LeftButtonDown')
        while not clg.quit:
            sleep(.01)
            pythoncom.PumpWaitingMessages()
            m.checks()



