"""
Great GATsBi: Hybrid, Multimodal, Trajectory Forecasting for Bicycles using Anticipation Mechanism
-------------------------------------------
Authors:        ANONYMOUS
Organization:   ANONYMOUS
Development:    2025
Submitted to:   Conference on Neural Information Processing Systems (NEURIPS25)
-------------------------------------------
This file contains the implementation of common benchmark models for cars, using constant velocity and acceleration for prediction.
"""




# #############################################################################
# IMPORTS
import numpy as np
import torch




# #############################################################################
# MODEL

class ModelClassic:
    def __init__(self, model_func, prediction_length):
        self.model_func = model_func
        self.prediction_length = prediction_length

    def eval(self):
        pass
    
    def __call__(self, traj_hist):
        forecasts = [
            np.stack(self.model_func(record[:, 0], record[:, 1], prediction_length=self.prediction_length), axis=1)
            for record in traj_hist
        ]
        trajectory = np.stack(forecasts, axis=0)  # (batch_size, prediction_length, 2)
        return torch.from_numpy(trajectory).float()
        
def model_classic(model_func, prediction_lenght):
    return model_func()

def constant_velocity_predictor(x_hist, y_hist, history_dt=0.04, prediction_length=50):
    """
    Predicts future x, y positions assuming constant velocity.
    
    Parameters:
        x_hist (np.ndarray): History of x positions, shape (N,)
        y_hist (np.ndarray): History of y positions, shape (N,)
        history_dt (float): Time step between observations in seconds (default: 0.025)
        prediction_length (int): Number of future time steps to predict
        
    Returns:
        x_pred (np.ndarray): Predicted x positions, shape (prediction_length,)
        y_pred (np.ndarray): Predicted y positions, shape (prediction_length,)
    """
    # estimate velocity from last two points (or use filtered velocity if available)
    n = 1
    vx = (x_hist[-1] - x_hist[-1-n]) / (n * history_dt)
    vy = (y_hist[-1] - y_hist[-1-n]) / (n * history_dt)

    # generate time steps
    future_times = np.arange(1, prediction_length + 1) * history_dt

    # predict future positions
    x_pred = x_hist[-1] + vx * future_times
    y_pred = y_hist[-1] + vy * future_times
    
    return x_pred, y_pred

def constant_acceleration_predictor(x_hist, y_hist, history_dt=0.04, prediction_length=50):
    """
    Predicts future x, y positions assuming constant acceleration.
    
    Parameters:
        x_hist (np.ndarray): History of x positions, shape (N,)
        y_hist (np.ndarray): History of y positions, shape (N,)
        history_dt (float): Time step between observations in seconds
        prediction_length (int): Number of future time steps to predict

    Returns:
        x_pred (np.ndarray): Predicted x positions, shape (prediction_length,)
        y_pred (np.ndarray): Predicted y positions, shape (prediction_length,)
    """
    # estimate velocities
    vx_curr = (x_hist[-1] - x_hist[-2]) / history_dt
    vx_prev = (x_hist[-2] - x_hist[-3]) / history_dt
    vy_curr = (y_hist[-1] - y_hist[-2]) / history_dt
    vy_prev = (y_hist[-2] - y_hist[-3]) / history_dt

    # estimate accelerations
    ax = (vx_curr - vx_prev) / history_dt
    ay = (vy_curr - vy_prev) / history_dt

    # predict future positions using constant acceleration
    future_times = np.arange(1, prediction_length + 1) * history_dt
    x_pred = x_hist[-1] + vx_curr * future_times + 0.5 * ax * (future_times ** 2)
    y_pred = y_hist[-1] + vy_curr * future_times + 0.5 * ay * (future_times ** 2)

    return x_pred, y_pred