"""One Euro Filter для стабилизации ландмарок."""

import math
import time


class _LowPass:
    __slots__ = ('hatxprev', 'y', 'initialized')

    def __init__(self):
        self.hatxprev = 0.0
        self.y = 0.0
        self.initialized = False

    def filter(self, x, alpha):
        if not self.initialized:
            self.hatxprev = x
            self.y = x
            self.initialized = True
            return x
        self.y = alpha * x + (1.0 - alpha) * self.hatxprev
        self.hatxprev = self.y
        return self.y

    def hatx(self):
        return self.hatxprev


class OneEuroFilter:
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x = _LowPass()
        self.dx = _LowPass()
        self.last_time = None

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, value, timestamp=None):
        if timestamp is None:
            timestamp = time.monotonic()
        if self.last_time is not None and timestamp > self.last_time:
            self.freq = 1.0 / (timestamp - self.last_time)
        self.last_time = timestamp

        prev_x = self.x.hatx() if self.x.initialized else value
        dx = (value - prev_x) * self.freq
        edx = self.dx.filter(dx, self._alpha(self.dcutoff, self.freq))
        cutoff = self.mincutoff + self.beta * abs(edx)
        return self.x.filter(value, self._alpha(cutoff, self.freq))


class Vec2Filter:
    def __init__(self, **kwargs):
        self.fx = OneEuroFilter(**kwargs)
        self.fy = OneEuroFilter(**kwargs)

    def __call__(self, x, y, timestamp=None):
        return self.fx(x, timestamp), self.fy(y, timestamp)
