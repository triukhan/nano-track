import numpy as np

# ---- Motion model ----
F = np.array([
    [1, 0, 1, 0, 0],  # cx += vx
    [0, 1, 0, 1, 0],  # cy += vy
    [0, 0, 1, 0, 0],  # vx
    [0, 0, 0, 1, 0],  # vy
    [0, 0, 0, 0, 1],  # scale
], dtype=np.float32)

# ---- Process noise ----
Q = np.diag([1, 1, 5, 5, 2]).astype(np.float32)


class Kalman:
    def __init__(self):
        self.x = np.zeros(5)
        self.P = np.eye(5) * 10

    def predict(self):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, H, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P


H_nano = np.array([
    [1, 0, 0, 0, 0],  # cx
    [0, 1, 0, 0, 0],  # cy
    [0, 0, 0, 0, 1],  # scale
], dtype=np.float32)

R_nano = np.diag([5, 5, 10]).astype(np.float32)


# 🔹 LK: velocity
H_lk = np.array([
    [0, 0, 1, 0, 0],  # vx
    [0, 0, 0, 1, 0],  # vy
], dtype=np.float32)

R_lk = np.diag([2, 2]).astype(np.float32)