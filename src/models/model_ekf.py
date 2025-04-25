"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains the implementation of an Extended Kalman Filter for filtering and predicting the trajectory of bicycles.
"""




# #############################################################################
# IMPORTS
import numpy as np
import torch
# from tqdm import tqdm
import time




# #############################################################################
# MODEL

class ModelXKalman:
    def __init__(self, prediction_length):
        self.prediction_length = prediction_length
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
    """
    def __call__(self, traj_hist):
        batch_size = traj_hist.shape[0]
        forecasts = []
        start_time = time.time()
        for record in tqdm(traj_hist, desc="Kalman predictions", unit="traj"):
            ekf_pred = do_prediction(
                record[:, 0], record[:, 1],
                prediction_length=self.prediction_length,
                device=self.device
            )
            forecasts.append(ekf_pred)
        total_time = time.time() - start_time
        print(f"Processed {batch_size} trajectories in {total_time:.2f} seconds "
              f"({total_time/batch_size:.4f} sec/trajectory)")
        trajectory = torch.from_numpy(np.stack(forecasts, axis=0)).float()
        return trajectory
    """
    def __call__(self, traj_hist):
        batch_size = traj_hist.shape[0]
        forecasts = []
        start_time = time.time()
        for record in traj_hist: 
            ekf_pred = do_prediction(
                record[:, 0], record[:, 1],
                prediction_length=self.prediction_length,
                device=self.device
            )
            forecasts.append(ekf_pred)
        total_time = time.time() - start_time
        trajectory = torch.from_numpy(np.stack(forecasts, axis=0)).float()
        return trajectory
    
class ExtendedKalmanFilterWithEstimationTorch:
    def __init__(self, L=1.75, dt=0.1, device='cpu'):
        self.L = L
        self.dt = dt
        self.device = device

        # State: [x, y, theta, v]
        self.state = torch.zeros(4, device=device)
        self.P = torch.eye(4, device=device)
        self.Q = torch.eye(4, device=device) * 0.1
        self.R = torch.eye(2, device=device) * 0.5  # Only x, y are measured
        self.H = torch.zeros((2, 4), device=device)
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y

    def predict(self, delta, v):
        x, y, theta, _ = self.state
        dx = v * torch.cos(theta) * self.dt
        dy = v * torch.sin(theta) * self.dt
        dtheta = (v / self.L) * torch.tan(delta) * self.dt

        # Update state
        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += dtheta
        self.state[3] = v  # Assume velocity is constant for this step

        # Jacobian
        F = torch.eye(4, device=self.device)
        F[0, 2] = -v * torch.sin(theta) * self.dt
        F[1, 2] = v * torch.cos(theta) * self.dt
        F[2, 3] = (torch.tan(delta) * self.dt) / self.L
        F[3, 3] = 1.0

        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        # z: [x, y]
        y = z - self.H @ self.state  # Innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ torch.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (torch.eye(4, device=self.device) - K @ self.H) @ self.P

    def estimate_velocity_and_heading(self, trajectory):
        dx = torch.diff(trajectory[:, 0])
        dy = torch.diff(trajectory[:, 1])
        velocities = torch.hypot(dx, dy) / self.dt
        headings = torch.atan2(dy, dx)
        return velocities, headings

    def filter_trajectory(self, trajectory):
        n = trajectory.shape[0]
        filtered_states = []

        velocities, headings = self.estimate_velocity_and_heading(trajectory)
        self.state[0], self.state[1] = trajectory[0]
        self.state[2] = headings[0] if len(headings) > 0 else torch.tensor(0.0, device=self.device)
        self.state[3] = velocities[0] if len(velocities) > 0 else torch.tensor(0.0, device=self.device)

        for i in range(1, n):
            v = velocities[i-1] if i-1 < len(velocities) else self.state[3]
            delta = torch.tensor(0.0, device=self.device)
            self.predict(delta, v)
            self.update(trajectory[i])
            filtered_states.append(self.state.clone())

        return torch.stack(filtered_states)

    def predict_future_trajectory(self, n_steps, delta=None, v=None):
        predicted_states = []
        state_backup = self.state.clone()
        P_backup = self.P.clone()

        v = self.state[3] if v is None else v
        delta = torch.tensor(0.0, device=self.device) if delta is None else delta

        for _ in range(n_steps):
            self.predict(delta, v)
            predicted_states.append(self.state.clone())

        self.state = state_backup
        self.P = P_backup

        return torch.stack(predicted_states)

def do_prediction(x_hist, y_hist, prediction_length, device='cpu'):
    x_hist = torch.as_tensor(x_hist, dtype=torch.float32, device=device)
    y_hist = torch.as_tensor(y_hist, dtype=torch.float32, device=device)
    lane_xy = torch.stack((x_hist, y_hist), dim=1)
    ekf = ExtendedKalmanFilterWithEstimationTorch(device=device)
    filtered_traj = ekf.filter_trajectory(lane_xy)
    ekf_pred = ekf.predict_future_trajectory(n_steps=prediction_length)
    return ekf_pred[:, :2].cpu().numpy()