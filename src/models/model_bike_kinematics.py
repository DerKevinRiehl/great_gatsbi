"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains the implementation of a bicycle kinematics model for bicycles inspired by models for cars.
Given historical trajectories, it estimates steering and heading angles to predict curvy trajectories.
"""




# #############################################################################
# IMPORTS
import numpy as np
import torch




# #############################################################################
# MODEL

class ModelBikeKinematics:
    def __init__(self, prediction_length):
        self.prediction_length = prediction_length

    def __call__(self, traj_hist):
        forecasts = [
            np.stack(predict_trajectory_kinematic(record[:, 0], record[:, 1], prediction_length=self.prediction_length), axis=1)
            for record in traj_hist
        ]
        trajectory = np.stack(forecasts, axis=0)  # (batch_size, prediction_length, 2)
        return torch.from_numpy(trajectory).float()
        
def estimate_velocity_and_steering_angle(x_hist, y_hist, dt=0.1, L=1.75):
    x_hist = np.asarray(x_hist)
    y_hist = np.asarray(y_hist)
    dx = np.diff(x_hist)
    dy = np.diff(y_hist)
    velocities = np.hypot(dx, dy) / dt

    heading = np.arctan2(dy, dx)
    heading_prev = np.roll(heading, 1)
    heading_prev[0] = heading[0]
    delta_theta = heading - heading_prev

    # Use last dx/dy for length calculation (avoid division by zero)
    dist = np.hypot(dx, dy)
    dist[dist == 0] = 1e-8
    steering_angles = np.arctan(np.tan(delta_theta) * L / dist)
    return velocities, steering_angles

def predict_trajectory_kinematic(x_hist, y_hist, prediction_length, dt=0.04, L=1.75):
    x_hist = np.asarray(x_hist)
    y_hist = np.asarray(y_hist)
    # print(type(x_hist), type(y_hist))
    velocities, steering_angles = estimate_velocity_and_steering_angle(x_hist, y_hist, dt, L)
    v = velocities[-1] if len(velocities) > 0 else 0.0
    delta = steering_angles[-1] if len(steering_angles) > 0 else 0.0

    x_pred = np.empty(prediction_length, dtype=np.float32)
    y_pred = np.empty(prediction_length, dtype=np.float32)

    # Initial state
    x = x_hist[-1]
    y = y_hist[-1]
    # Estimate initial heading from last two points
    if len(x_hist) >= 2:
        theta = np.arctan2(y_hist[-1] - y_hist[-2], x_hist[-1] - x_hist[-2])
    else:
        theta = np.pi / 2

    for i in range(prediction_length):
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = (v / L) * np.tan(delta) * dt
        x += dx
        y += dy
        theta += dtheta
        x_pred[i] = x
        y_pred[i] = y

    return x_pred, y_pred
