"""
Great GATsBi: Social-Force-Informed, Multimodal Bicycle Trajectory Prediction using GATs
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
# import torch
import math















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








# #############################################################################
# MODEL

# class ModelPhysics:
#     def __init__(self, prediction_length):
#         self.prediction_length = prediction_length

#     def __call__(self, traj_hist):
#         forecasts = [
#             np.stack(predict_trajectory_kinematic(record[:, 0], record[:, 1], prediction_length=self.prediction_length), axis=1)
#             for record in traj_hist
#         ]
#         trajectory = np.stack(forecasts, axis=0)  # (batch_size, prediction_length, 2)
#         return torch.from_numpy(trajectory).float()
        
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

"""
# from data.data_loader import load_trajectories
# from data.data_gen_social_lstm import generate_training_data_social_lstm_one_record

# n_neighbors = 5
# history_length = 100
# prediction_length = 100
# sequence = "DJI_20240906105621_0010_D.MP4-PART_3"        
# ego_vehicle_id = "BICYCLE_1"
# frame_id=2500

# trajectory_data = load_trajectories()


# neighbor_trajs, lane_xy, pred_traj = generate_training_data_social_lstm_one_record(trajectory_data, n_neighbors, history_length, 
#                                                                                     prediction_length, sequence, 
#                                                                                     ego_vehicle_id, frame_id)

# np.savetxt('myarray.txt', lane_xy)
# np.savetxt('myarray2.txt', pred_traj)


prediction_length = 25
lane_xy = np.loadtxt('myarray.txt')
true_xy = np.loadtxt('myarray2.txt')
x_hist = lane_xy[:,0]
y_hist = lane_xy[:,1]

x_pred_kin, y_pred_kin = predict_trajectory_kinematic(x_hist, y_hist, prediction_length=100, dt=0.04, L=1.75)
x_pred_cv, y_pred_cv = constant_velocity_predictor(x_hist, y_hist, history_dt=0.04, prediction_length=100)
x_pred_ca, y_pred_ca = constant_acceleration_predictor(x_hist, y_hist, history_dt=0.04, prediction_length=100)


import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.plot(x_hist, y_hist)


def drawTimeSeriesAndDots(true_xy, color, style=None):
    if style is not None:
        plt.plot(true_xy[:,0], true_xy[:,1], "--", color=color)
    else:
        plt.plot(true_xy[:,0], true_xy[:,1], color=color)
    plt.scatter([true_xy[25-1,0]], [true_xy[25-1,1]], color=color)
    plt.scatter([true_xy[50-1,0]], [true_xy[50-1,1]], color=color)
    plt.scatter([true_xy[75-1,0]], [true_xy[75-1,1]], color=color)
    plt.scatter([true_xy[100-1,0]], [true_xy[100-1,1]], color=color)

plt.scatter([0], [0])
drawTimeSeriesAndDots(true_xy, color="green", style="--")
drawTimeSeriesAndDots(np.column_stack((x_pred_kin, y_pred_kin)), color="blue", style="--")
drawTimeSeriesAndDots(np.column_stack((x_pred_cv, y_pred_cv)), color="red", style="--")
drawTimeSeriesAndDots(np.column_stack((x_pred_ca, y_pred_ca)), color="gray", style="--")

plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.xlabel("Relative X [Lane Coordinates]")
plt.ylabel("Relative Y [Lane Coordinates]")
plt.gca().set_aspect('equal', adjustable='box')
"""









# """


class ExtendedKalmanFilterWithEstimation:
    def __init__(self, L=1.75, dt=0.1):
        self.L = L  # Bicycle length
        self.dt = dt  # Time step

        # State: [x, y, theta, v]
        self.state = np.zeros(4)
        self.P = np.eye(4)
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 0.5  # Only x, y are measured
        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y

    def predict(self, delta, v):
        x, y, theta, _ = self.state
        dx = v * np.cos(theta) * self.dt
        dy = v * np.sin(theta) * self.dt
        dtheta = (v / self.L) * np.tan(delta) * self.dt

        # Update state
        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += dtheta
        self.state[3] = v  # Assume velocity is constant for this step

        # Jacobian
        F = np.eye(4)
        F[0, 2] = -v * np.sin(theta) * self.dt
        F[1, 2] = v * np.cos(theta) * self.dt
        F[2, 3] = (np.tan(delta) * self.dt) / self.L
        F[3, 3] = 1.0

        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        # z: [x, y]
        y = z - self.H @ self.state  # Innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def estimate_velocity_and_heading(self, trajectory):
        dx = np.diff(trajectory[:, 0])
        dy = np.diff(trajectory[:, 1])
        velocities = np.hypot(dx, dy) / self.dt
        headings = np.arctan2(dy, dx)
        return velocities, headings

    def filter_trajectory(self, trajectory):
        n = len(trajectory)
        filtered_states = []

        velocities, headings = self.estimate_velocity_and_heading(trajectory)
        self.state[0], self.state[1] = trajectory[0]
        self.state[2] = headings[0] if len(headings) > 0 else 0.0
        self.state[3] = velocities[0] if len(velocities) > 0 else 0.0

        for i in range(1, n):
            v = velocities[i-1] if i-1 < len(velocities) else self.state[3]
            # Optionally estimate delta from heading change, but with only x,y it's hard
            delta = 0.0
            self.predict(delta, v)
            self.update(trajectory[i])
            filtered_states.append(self.state.copy())

        return np.array(filtered_states)

    def predict_future_trajectory(self, n_steps, delta=None, v=None):
        """
        Predicts n_steps into the future using the current state.
        Optionally, you can provide delta and v, otherwise uses current state's v and delta=0.
        Returns an array of predicted [x, y, theta, v] states.
        """
        predicted_states = []
        # Save current state to restore later
        state_backup = self.state.copy()
        P_backup = self.P.copy()

        # Use current state values if not provided
        v = self.state[3] if v is None else v
        delta = 0.0 if delta is None else delta

        for _ in range(n_steps):
            self.predict(delta, v)
            predicted_states.append(self.state.copy())

        # Restore state
        self.state = state_backup
        self.P = P_backup

        return np.array(predicted_states)
    
# """



# from data.data_loader import load_trajectories
# from data.data_gen_social_lstm import generate_training_data_social_lstm_one_record

# n_neighbors = 5
# history_length = 100
# prediction_length = 100
# sequence = "DJI_20240906105621_0010_D.MP4-PART_3"        
# ego_vehicle_id = "BICYCLE_1"
# frame_id=2500

# trajectory_data = load_trajectories()


# neighbor_trajs, lane_xy, pred_traj = generate_training_data_social_lstm_one_record(trajectory_data, n_neighbors, history_length, 
#                                                                                     prediction_length, sequence, 
#                                                                                     ego_vehicle_id, frame_id)

# np.savetxt('myarray.txt', lane_xy)
# np.savetxt('myarray2.txt', pred_traj)


prediction_length = 100
lane_xy = np.loadtxt('myarray.txt')
true_xy = np.loadtxt('myarray2.txt')
x_hist = lane_xy[:,0]
y_hist = lane_xy[:,1]

x_pred_kin, y_pred_kin = predict_trajectory_kinematic(x_hist, y_hist, prediction_length=100, dt=0.04, L=1.75)
x_pred_cv, y_pred_cv = constant_velocity_predictor(x_hist, y_hist, history_dt=0.04, prediction_length=100)
x_pred_ca, y_pred_ca = constant_acceleration_predictor(x_hist, y_hist, history_dt=0.04, prediction_length=100)

ekf = ExtendedKalmanFilterWithEstimation()
prep = np.stack((x_hist, y_hist), axis=1)
filtered_traj = ekf.filter_trajectory(prep)
ekf_pred = ekf.predict_future_trajectory(n_steps=prediction_length)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.plot(x_hist, y_hist)


def drawTimeSeriesAndDots(true_xy, color, lbl, style=None):
    if style is not None:
        plt.plot(true_xy[:,0], true_xy[:,1], "--", label=lbl, color=color)
    else:
        plt.plot(true_xy[:,0], true_xy[:,1], color=color)
    plt.scatter([true_xy[25-1,0]], [true_xy[25-1,1]], color=color)
    plt.scatter([true_xy[50-1,0]], [true_xy[50-1,1]], color=color)
    plt.scatter([true_xy[75-1,0]], [true_xy[75-1,1]], color=color)
    plt.scatter([true_xy[100-1,0]], [true_xy[100-1,1]], color=color)

plt.scatter([0], [0])
drawTimeSeriesAndDots(true_xy, color="green", lbl="truth", style="--")
drawTimeSeriesAndDots(np.column_stack((x_pred_kin, y_pred_kin)), color="blue", lbl="bike kinematics", style="--")
drawTimeSeriesAndDots(np.column_stack((x_pred_cv, y_pred_cv)), color="red", lbl="const velocity", style="--")
drawTimeSeriesAndDots(np.column_stack((x_pred_ca, y_pred_ca)), color="gray", lbl="const acceleration", style="--")
drawTimeSeriesAndDots(ekf_pred, color="orange", lbl="extended kalman filter", style="--")

plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.xlabel("Relative X [Lane Coordinates]")
plt.ylabel("Relative Y [Lane Coordinates]")

plt.legend(fontsize="small", loc="lower right")
plt.gca().set_aspect('equal', adjustable='box')









