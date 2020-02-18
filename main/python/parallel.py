from __future__ import print_function
from time import sleep
from multiprocessing import Queue, cpu_count, Pool, Event, queues
from tqdm.auto import tqdm


def parfor(iterable=None, args=(), kwargs={}, length=None, desc=None, bar=True, qbar=True, nP=None, serial=4):
    def decfun(fun):
        return pmap(fun, iterable, args, kwargs, length, desc, bar, qbar, nP, serial)

    return decfun

def worker(fun, Qi, Qo, V, args, kwargs):
    while not V.is_set():
        try:
            i, c = Qi.get(True, 0.02)
            Qo.put((i, fun(c, *args, **kwargs)))
        except queues.Empty:
            continue
    try:
        del i, c, args, kwargs
    except:
        pass

def pmap(fun, iterable=None, args=(), kwargs={}, length=None, desc=None, bar=True, qbar=False, nP=None, serial=4):
    """ map a function fun to each iteration in iterable
            best use: iterable is a generator and length is given to this function

        fun:    function taking arguments: image from self(frames[i]), item from coiter[i], other arguments
                    defined in args & kwargs
        iterable: iterable from which an item is given to fun as a first argument
        args:   tuple with other unnamed arguments to fun
        kwargs: dict with other named arguments to fun
        desc:   string with description of the progress bar
        bar:    bool enable bar
        pbar:   bool enable buffer indicator bar
        nP:     number of workers, default: number of cpu's/4
        serial: switch to serial if number of tasks less than serial, default: 4

        wp@tl20200204
    """

    if length is None:
        try:
            length = len(iterable)
        except:
            length = 100

    #serial = 1e6

    if length and length < serial:  # serial case
        return [fun(c, *args, **kwargs) for c in tqdm(iterable, desc=desc, total=length)]
    else:  # parallel case
        nP = nP or cpu_count()
        nP = min(nP, length)

        V = Event()
        Qi = Queue(3 * nP)
        Qo = Queue(3 * nP)

        res = []

        P = Pool(nP, initializer=worker, initargs=(fun, Qi, Qo, V, args, kwargs))
        with tqdm(total=length, desc=desc, disable=not bar) as bar:
            for i, j in enumerate(iterable):
                while Qi.full():
                    try:
                        res.append(Qo.get(True, 0.02))
                        bar.update()
                    except queues.Empty:
                        pass
                Qi.put((i, j))
                if bar.total < i + 1:
                    bar.total = i + 1
            sleep(0.1)
            Qi.close()
            while len(res) <= i:
                try:
                    res.append(Qo.get(True, 0.02))
                    bar.update()
                except queues.Empty:
                    pass
            V.set()
            Qo.close()
            Qi.join_thread()
            Qo.join_thread()
            P.close()
            P.join()

        res.sort(key=lambda t: t[0])
        return [i[1] for i in res]