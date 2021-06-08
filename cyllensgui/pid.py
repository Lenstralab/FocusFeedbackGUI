from time import time

class pid:
    def __init__(self, SetVal=0, CurOutVal=0, MaxStep=0.25, IntervalTime=1, Gain=5e-4):
        self.time = time()
        self.K = Gain
        self.U = IntervalTime/10
        self.S = CurOutVal*self.U/self.K/2
        self.LP = 0
        self.LPID = CurOutVal
        self.MaxStep = MaxStep
        self.SetVal = SetVal
        
    def __call__(self, CurVal):
        dt, self.time = -self.time, time()
        dt += self.time
        if dt<1e-6:
            dt = 10*self.U
        P = self.SetVal - CurVal
        self.S += P*dt
        I = 2*self.S/self.U
        D = self.U*(P-self.LP)/dt/8
        PID = self.K*(P+I+D)
        if PID>(self.LPID + self.MaxStep):
            PID = self.LPID + self.MaxStep
            self.S = PID*self.U/self.K/2
        elif PID<(self.LPID - self.MaxStep):
            PID = self.LPID - self.MaxStep
            self.S = PID*self.U/self.K/2
        self.LPID = PID
        self.LP = P
        return PID