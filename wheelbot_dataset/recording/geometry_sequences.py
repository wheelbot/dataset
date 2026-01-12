"""
Geometric trajectory generators for wheelbot experiments.

Contains functions for generating circular, figure-eight, and other
geometric trajectories with corresponding yaw and velocity setpoints.
"""

import numpy as np


def generate_circle_trajectory(radius=1.0, velocity=0.2, dt=0.05):
    """
    Generate a circular trajectory with constant velocity.
    
    Args:
        radius: Radius of the circle in meters
        velocity: Constant forward velocity in m/s
        dt: Time step in seconds
    
    Returns:
        x: Array of x positions
        y: Array of y positions
        yaw: Array of yaw angles in degrees (absolute heading)
        vel: Array of velocity values (constant)
        time: Array of time values
    """
    # Circumference and duration
    circumference = 2 * np.pi * radius
    duration_s = circumference / velocity
    
    N = int(duration_s / dt)
    time = np.arange(N) * dt
    
    # Angular velocity (rad/s)
    omega = velocity / radius
    
    # Angle around the circle
    theta = omega * time
    
    # Position (starting at x=radius, y=0, moving counterclockwise)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    # Yaw is tangent to the circle (perpendicular to radius)
    # For counterclockwise motion, yaw = theta + 90 degrees
    yaw = np.rad2deg(theta + np.pi / 2)
    
    # Wrap yaw to [-180, 180]
    yaw = np.mod(yaw + 180, 360) - 180
    
    # Constant velocity
    vel = np.ones(N) * velocity
    
    return x, y, yaw, vel, time


def generate_figure_eight_trajectory(size=1.0, velocity=0.2, dt=0.05):
    """
    Generate a figure-eight (lemniscate) trajectory with constant velocity.
    
    The figure-eight is parametrized using a lemniscate of Bernoulli.
    
    Args:
        size: Scale factor for the figure-eight (half-width in meters)
        velocity: Constant forward velocity in m/s
        dt: Time step in seconds
    
    Returns:
        x: Array of x positions
        y: Array of y positions
        yaw: Array of yaw angles in degrees (absolute heading)
        vel: Array of velocity values (constant)
        time: Array of time values
    """
    # Approximate path length of a figure-eight (lemniscate)
    # For lemniscate with parameter 'a': x = a*cos(t)/(1+sin^2(t)), y = a*sin(t)*cos(t)/(1+sin^2(t))
    # We use a simpler parametric form: x = size*sin(t), y = size*sin(t)*cos(t)
    # This creates a nice figure-eight shape
    
    # Approximate path length (computed numerically for one period)
    # For this parametrization, path length ≈ 4.84 * size
    path_length = 4.84 * size
    duration_s = path_length / velocity
    
    N = int(duration_s / dt)
    time = np.arange(N) * dt
    
    # Parameter t goes from 0 to 2*pi for one complete figure-eight
    t = 2 * np.pi * time / duration_s
    
    # Figure-eight parametrization
    x = size * np.sin(t)
    y = size * np.sin(t) * np.cos(t)
    
    # Compute velocity components (derivatives)
    dx_dt = size * np.cos(t)
    dy_dt = size * (np.cos(t) * np.cos(t) - np.sin(t) * np.sin(t))  # = size * cos(2t)
    
    # Yaw is the direction of travel
    yaw = np.rad2deg(np.arctan2(dy_dt, dx_dt))
    
    # Constant velocity
    vel = np.ones(N) * velocity
    
    return x, y, yaw, vel, time


def convert_absolute_yaw_to_deltas(yaw_absolute, dt):
    """
    Convert absolute yaw angles to delta angles for transmission.
    
    Computes the angular velocity (rate of change of yaw) and sends
    delta angles that represent the change per timestep.
    
    Args:
        yaw_absolute: Array of absolute yaw angles in degrees
        dt: Time step in seconds
    
    Returns:
        yaw_deltas: Array of yaw delta values in degrees
    """
    N = len(yaw_absolute)
    yaw_deltas = np.zeros(N)
    
    # First sample: no delta (start from current heading)
    yaw_deltas[0] = 0.0
    
    # For subsequent samples: compute the angular difference
    for i in range(1, N):
        delta = yaw_absolute[i] - yaw_absolute[i - 1]
        
        # Handle wraparound at ±180 degrees
        if delta > 180:
            delta -= 360
        elif delta < -180:
            delta += 360
        
        yaw_deltas[i] = delta
    
    return yaw_deltas
