import yaml
import os

class strct:
    def __init__(self, parent, key):
        self.__dict__['parent'] = parent
        self.__dict__['key'] = key

    def __call__(self, *args, **kwargs):
        return None

    def __contains__(self, item):
        c = self.parent.getconf()
        if not self.key in c:
            return False
        c = c[self.key]
        if not item in c:
            return False
        return True

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, item, value):
        return self.__setattr__(item, value)

    def __getattr__(self, item):
        c = self.parent.getconf()
        if not self.key in c:
            return strct(self, item)
        c = c[self.key]
        if not item in c:
            return strct(self, item)
        i = c[item]
        if isinstance(i, dict):
            return strct(self, item)
        return i

    def __setattr__(self, item, value):
        c = self.parent.getconf()
        if not self.key in c:
            c[self.key] = dict()
        if not item in c[self.key]:
            c[self.key][item] = dict()
        c[self.key][item] = value
        self.parent.setconf(c)

    def getconf(self):
        c = self.parent.getconf()
        if not self.key in c:
            return dict()
        return self.parent.getconf()[self.key]

    def setconf(self, conf):
        c = self.parent.getconf()
        c[self.key] = conf
        self.parent.setconf(c)

class conf:
    def __init__(self, filename='D:\CylLensGUI\src\main\python\conf.yml'):
        self.__dict__['_filename'] = filename

    def __contains__(self, item):
        c = self.getconf()
        if not item in c:
            return False
        return True

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, key, value):
        return self.__setattr__(key, value)

    def __getattr__(self, item):
        if item == 'filename':
            return self._filename
        c = self.getconf()
        if not item in c:
            return strct(self, item)
        i = c[item]
        if isinstance(i, dict):
            return strct(self, item)
        return i

    def __setattr__(self, item, value):
        if item == 'filename':
            self.setfilename(value)
        else:
            c = self.getconf()
            c[item] = value
            self.setconf(c)

    def getconf(self):
        try:
            y = open(self._filename, 'r')
            c = yaml.full_load(y)
            y.close()
            if c is None:
                return dict()
            return c
        except:
            return dict()

    def setconf(self, conf):
        y = open(self._filename, 'w')
        yaml.dump(conf, y, default_flow_style=None)
        y.close()

    def setfilename(self, filename):
        if not filename[-4:] == '.yml':
            filename += '.yml'
        if not os.path.isfile(filename):
            c = self.conf
            y = open(filename, 'w+')
            yaml.dump(c, y, default_flow_style=None)
            y.close()
        self.__dict__['_filename'] = filename