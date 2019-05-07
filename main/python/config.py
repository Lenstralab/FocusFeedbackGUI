import yaml
import os

def prop(*loc):
    #Define a class property as an entry in a dictionary (classinstance.conf) recursively (dict with dicts in it)
    def dictnavget(d, loc):
        try:
            for l in loc:
                if l in d:
                    d = d[l]
                else:
                    return None
            return d
        except:
            return None

    def dictnavset(d, loc, var):
        def fun(d, loc, var):
            for l in loc[:-1]:
                if l in d:
                    d = d[l]
                else:
                    d[l] = dict()
                    d = d[l]
            d[loc[-1]] = var

        fun(d, loc, var)
        return d

    def getter(loc):
        def gett(self):
            return dictnavget(self.conf, loc)
        return gett

    def setter(loc):
        def sett(self, var):
            self.conf = dictnavset(self.conf, loc, var)
        return sett

    return property(getter(loc), setter(loc))

class conf:
    def __init__(self, filename='D:\CylLensGUI\src\main\python\conf.yml'):
        self._filename = filename

    q = prop('q')
    maxStep = prop('maxStep')
    theta = prop('theta')

    @property
    def conf(self):
        try:
            y = open(self.filename, 'r')
            c = yaml.full_load(y)
            y.close()
            if c is None:
                c = dict()
        except:
            c = dict()
        return c

    @conf.setter
    def conf(self, conf):
        y = open(self.filename, 'w')
        yaml.dump(conf, y, default_flow_style=None)
        y.close()

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        if not filename[-4:] == '.yml':
            filename += '.yml'
        if not os.path.isfile(filename):
            c = self.conf
            y = open(filename, 'w+')
            yaml.dump(c, y, default_flow_style=None)
            y.close()
        self._filename = filename