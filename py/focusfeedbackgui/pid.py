from time import time


class Pid:
    def __init__(self, set_val=0, current_out_val=0, max_step=0.25, interval_time=1, gain=5e-4):
        self.time = time()
        self.gain_k = gain
        self.gain_u = interval_time / 10
        self.gain_s = current_out_val * self.gain_u / self.gain_k / 2
        self.last_pid_p = 0
        self.last_value = current_out_val
        self.max_step = max_step
        self.set_val = set_val

    def __call__(self, current_val):
        dt, self.time = -self.time, time()
        dt += self.time
        if dt < 1e-6:
            dt = 10 * self.gain_u
        pid_p = self.set_val - current_val
        self.gain_s += pid_p * dt
        pid_i = 2 * self.gain_s / self.gain_u
        pid_d = self.gain_u * (pid_p - self.last_pid_p) / dt / 8
        value = self.gain_k * (pid_p + pid_i + pid_d)
        if value > (self.last_value + self.max_step):
            value = self.last_value + self.max_step
            self.gain_s = value * self.gain_u / self.gain_k / 2
        elif value < (self.last_value - self.max_step):
            value = self.last_value - self.max_step
            self.gain_s = value * self.gain_u / self.gain_k / 2
        self.last_value = value
        self.last_pid_p = pid_p
        return value
