import pythoncom
from functools import partial
from time import sleep
from threading import Thread
from zen import zen, cst

def thread(fun):
    """ decorator to run function in a separate thread to keep the gui responsive
    """
    return lambda *args: Thread(target=fun, args=args).start()

def errwrap(fun):
    def e(*args, **kwargs):
        try:
            fun(*args, **kwargs)
        except:
            pass
    return e

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

def EventHandler(CLG):
    class EventHandlerCls(metaclass=EventHandlerMetaClass):
        clg = CLG #reference to app class
        zen = clg.zen

        @thread
        @errwrap
        def OnThrowEvent(self, *args):
            if args[0] == cst('LeftButtonDown') and self.clg.centerbox.isChecked():
                #print(('OnThrowEvent:', args))
                X = self.zen.MousePosition
                FS = self.zen.FrameSize
                pxsize = self.zen.pxsize
                Pos = self.clg.conf.ROIPos
                d = [(FS[i]/2 - X[i] + Pos[i]) * pxsize/1000 for i in range(2)]
                d[1] *= -1
                self.zen.MoveStageRel(*d)

        @thread
        @errwrap
        def OnThrowPropertyEvent(self, *args):
            # print(('OnThrowProperyEvent:', args))

            #make sure lmb event is enabled on a new document
            if self.clg.curzentitle != self.zen.Title:
                self.clg.curzentitle = self.zen.Title
                self.zen.EnableEvent('LeftButtonDown')

                #draw a black rectangle in the map when a new document is saved
                if self.zen.FileName[-4:] == '.czi':
                    LP = self.zen.StagePos
                    FS = self.zen.FrameSize
                    pxsize = self.zen.pxsize
                    self.clg.map.append_data_docs(LP[0] / 1000, LP[1] / 1000, FS[0] * pxsize / 1e6, FS[1] * pxsize / 1e6)
                    self.clg.map.draw()
                registereventwithdelay('LeftButtonDown', 2)

            if args[1] == 'TransmissionSpot' and self.clg.MagStr != self.zen.MagStr:
                self.clg.MagStr = self.zen.MagStr
                self.clg.confopen(self.clg.conf.filename)
            elif args[1] == '2C_FilterSlider' and self.clg.DLFilter != self.zen.DLFilter:
                self.clg.DLFilter = self.zen.DLFilter
                self.clg.dlf.setText(self.clg.dlfs.currentText().split(' & ')[self.zen.DLFilter])
                self.clg.chdlf.changeState(self.zen.DLFilter)
            elif args[1] == 'Stage':
                LP = self.clg.zen.StagePos
                FS = self.zen.FrameSize
                pxsize = self.zen.pxsize
                self.clg.map.numel_data(LP[0]/1000, LP[1]/1000, FS[0]*pxsize/1e6, FS[1]*pxsize/1e6)
                self.clg.map.draw()
            elif args[1] in ('DataColorPalette', 'FramesPerStack'):
                self.clg.changeColor()

    return EventHandlerCls

@thread
def events(clg):
    with zen(EventHandler(clg)) as z:
        z.EnableEvent('LeftButtonDown')
        while not clg.quit:
            sleep(.01)
            pythoncom.PumpWaitingMessages()

@thread
def registereventwithdelay(event, delay):
    sleep(delay)
    with zen() as z:
        z.EnableEvent(event)