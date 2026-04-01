import cv2
import numpy as np


class BallKalman:
    def __init__(self, dt=1.0):
        self.dt = dt
        self.kf = cv2.KalmanFilter(4, 2)
        self.initialized = False
        self._setup()

    def _setup(self):
        dt = self.dt

        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.04
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 15.0

    def init(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return None
        p = self.kf.predict()
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])

    def correct(self, x, y):
        if not self.initialized:
            self.init(x, y)
            return self.get_state()

        m = np.array([[x], [y]], dtype=np.float32)
        c = self.kf.correct(m)
        return float(c[0]), float(c[1]), float(c[2]), float(c[3])

    def get_state(self):
        if not self.initialized:
            return None
        s = self.kf.statePost
        return float(s[0]), float(s[1]), float(s[2]), float(s[3])

    def reset(self):
        self.initialized = False
