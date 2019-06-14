import pythoncom
from functools import partial
from time import sleep
from threading import Thread
from zen import zen, cst

LMBDown = 203

def thread(fun):
    """ decorator to run function in a separate thread to keep the gui responsive
    """
    return lambda *args: Thread(target=fun, args=args).start()

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

def EventHandler(CLG, ZEN):
    class EventHandlerCls(metaclass=EventHandlerMetaClass):
        clg = CLG
        zen = ZEN
        def OnThrowEvent(self, *args):
            if args[0] == cst('LeftButtonDown') and self.clg.docenter:
                # print(('OnThrowEvent:', args, X))
                self.clg.docenter = False
                self.clg.centerbtn.setText('Center')
                X = self.zen.MousePosition
                FS = self.zen.FrameSize
                pxsize = self.zen.pxsize
                d = [(FS[i]/2 - X[i]) * pxsize/1000 for i in range(2)]
                d[1] *= -1
                #print('Moving ({}; {})  um.'.format(*d))
                self.zen.MoveStageRel(*d)

        #OnThrowPropertyEvent

    return EventHandlerCls

@thread
def events(clg):
    z = zen(partial(EventHandler, CLG=clg))
    z.EnableEvent('LeftButtonDown')

    while not clg.stop:
        sleep(.05)
        pythoncom.PumpWaitingMessages()