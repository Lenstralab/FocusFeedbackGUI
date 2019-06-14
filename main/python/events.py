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

        @thread
        @errwrap
        def OnThrowEvent(self, *args):
            z = zen()
            if args[0] == cst('LeftButtonDown') and self.clg.docenter:
                # print(('OnThrowEvent:', args, X))
                #self.clg.docenter = False
                #self.clg.centerbtn.setText('Center')
                X = z.MousePosition
                FS = z.FrameSize
                pxsize = z.pxsize
                d = [(FS[i]/2 - X[i]) * pxsize/1000 for i in range(2)]
                d[1] *= -1
                #print('Moving ({}; {})  um.'.format(*d))
                z.MoveStageRel(*d)
            z.DisconnectZEN()

        @thread
        @errwrap
        def OnThrowPropertyEvent(self, *args):
            z = zen()
            print(('OnThrowProperyEvent:', args))
            if args[1] == 'TransmissionSpot' and self.clg.MagStr != z.MagStr:
                self.clg.MagStr = z.MagStr
                self.clg.confopen(self.clg.conf.filename)
            elif args[1] == '2C_FilterSlider' and self.clg.DLFilter != z.DLFilter:
                self.clg.DLFilter = z.DLFilter
                self.clg.dlf.setText(self.clg.dlfs.currentText().split(' & ')[z.DLFilter])
            elif args[1] == 'Stage':
                LP = z.StagePos
                self.clg.map.numel_data(LP[0], LP[1])
            z.DisconnectZEN()

    return EventHandlerCls

@thread
def events(clg):
    z = zen(EventHandler(clg))
    z.EnableEvent('LeftButtonDown')

    while not clg.stop:
        sleep(.01)
        pythoncom.PumpWaitingMessages()