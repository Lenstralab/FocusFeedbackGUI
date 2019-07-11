from zen import zen
from time import time, sleep
from matplotlib import pyplot
import numpy

def piezotest():
    z = zen()
    z.SetAnalogMode(False)
    t0 = time()
    p = []
    t = []

    while time()-t0 < 10:
        t.append(time()-t0)
        p.append(z.PiezoPos)
        sleep(0.01)

    pyplot.plot(t, p)
    pyplot.title('Digital mode')
    print('Digital mode\n std: {} nm\n pos: {}'.format(1000*numpy.std(p), len(numpy.unique(p))))

    z.SetAnalogMode(True)
    t0 = time()
    p = []
    t = []

    while time()-t0 < 10:
        t.append(time() - t0)
        p.append(z.PiezoPos)
        sleep(0.01)

    pyplot.figure()
    pyplot.plot(t, p)
    pyplot.title('Analog mode')
    print('Analog mode\n std: {} nm\n pos: {}'.format(1000*numpy.std(p), len(numpy.unique(p))))

def piezosteptest():
    z = zen()
    z.SetAnalogMode(False)
    t0 = time()
    p = []
    t = []

    while time()-t0 < 10:
        t.append(time()-t0)
        p.append(z.PiezoPos)
        sleep(0.01)

    pyplot.plot(t, p)
    pyplot.title('Digital mode')
    print('Digital mode\n std: {} nm\n pos: {}'.format(1000*numpy.std(p), len(numpy.unique(p))))

    z.SetAnalogMode(True)
    t0 = time()
    p = []
    t = []

    while time()-t0 < 10:
        t.append(time() - t0)
        p.append(z.PiezoPos)
        sleep(0.01)

    pyplot.figure()
    pyplot.plot(t, p)
    pyplot.title('Analog mode')
    print('Analog mode\n std: {} nm\n pos: {}'.format(1000*numpy.std(p), len(numpy.unique(p))))