import unreal as ue
from unreal import Vector, Rotator
from unreal import AutomationLibrary as AL
from unreal import SystemLibrary as SL
from unreal import GameplayStatics as GS

import matplotlib.pyplot as plt
import numpy as np
import math

METERS = 100




def apply_project_cam_matrix(vec, fov, IMG_OFFSET):
    val = np.ones(4)
    val[:3] = vec
    f = IMG_OFFSET[0] / np.tan(np.deg2rad(fov / 2))
    fx = f / vec[2]
    fy = f / vec[2]
    cam = np.array([[fx, 0, 0, IMG_OFFSET[0]],
                    [0, fy, 0, IMG_OFFSET[1]],
                    [0, 0, 1, 0]])
    proj = cam @ val
    return proj


def line_trace(world, anchor, cam_loc):
    loc = anchor.get_world_location()
    objT = ue.ObjectTypeQuery
    col = ue.LinearColor.RED
    arr = ue.Array(objT)
    arr.append(ue.ObjectTypeQuery.OBJECT_TYPE_QUERY1)
    arr.append(ue.ObjectTypeQuery.OBJECT_TYPE_QUERY2)
    arr.append(ue.ObjectTypeQuery.OBJECT_TYPE_QUERY3)
    arr.append(ue.ObjectTypeQuery.OBJECT_TYPE_QUERY4)
    arr.append(ue.ObjectTypeQuery.OBJECT_TYPE_QUERY5)
    arr.append(ue.ObjectTypeQuery.OBJECT_TYPE_QUERY6)
    ddt = ue.DrawDebugTrace.NONE
    arr2 = ue.Array(ue.Actor)
    line = SL.line_trace_multi_for_objects(world, cam_loc, loc,
                                           arr, True, arr2, ddt, False, col, col)
    return line


def relative_location(point, cam_trans):

    loc = cam_trans.inverse_transform_location(point)
    # coordinates in image plane as camera axis are (Y,Z,X) -> to normal (X,Y,Z)
    relative_point = (np.array([loc.y, -loc.z, loc.x]) / METERS)
    return relative_point


def proj_relative_location(point, cam_trans, fov, IMAGE_OFFSET):
    loc = cam_trans.inverse_transform_location(point)
    # coordinates in image plane as camera axis are (Y,Z,X) -> to normal (X,Y,Z)
    relative_point = (np.array([loc.y, -loc.z, loc.x]) / METERS)
    relative_point = apply_project_cam_matrix(relative_point, fov, IMAGE_OFFSET).round(1)
    return relative_point


def jitter_locator(dist):
    x_val, y_val, z_val = 400, 700, 200
    if dist == 30 * METERS:
        x_val, y_val, z_val = 400, 700, 200
    if dist == 50 * METERS:
        x_val, y_val, z_val = 1100, 1600, 200
    if dist == 70 * METERS:
        x_val, y_val, z_val = 1700, 2400, 500

    jitter = ue.Vector(np.random.uniform(-x_val, x_val),
                    np.random.uniform(-y_val, y_val),
                    np.random.uniform(-z_val, z_val), )
    return jitter


def jitter_earth():
    jitter = ue.Vector(np.random.uniform(-3500000*METERS, 3500000*METERS),
                    np.random.uniform(-3500000*METERS, 3500000*METERS),
                    np.random.uniform(0, 0), )
    return jitter


# Track target
def track_target(ego, target, jitter=False, jitter_val=5):
    ego_loc = ego.get_actor_location()
    tar_loc = target.get_actor_location()
    cam_rot = (tar_loc - ego_loc).rotator()
    if jitter:
        jit = Rotator(np.random.uniform(-jitter_val, jitter_val),
                      np.random.uniform(-jitter_val, jitter_val),
                      np.random.uniform(-jitter_val, jitter_val))
        return cam_rot.combine(jit)
    return cam_rot


def fibonacci_sphere(num_points, plot=False):
    ga = (3 - np.sqrt(5)) * np.pi  # golden angle

    # Create a list of golden angle increments along tha range of number of points
    theta = ga * np.arange(num_points)

    # Z is a split into a range of -1 to 1 in order to create a unit circle
    z = np.linspace(1 / num_points - 1, 1 - 1 / num_points, num_points)

    # a list of the radii at each height step of the unit circle
    radius = np.sqrt(1 - z * z)

    # Determine where xy fall on the sphere, given the azimuthal and polar angles
    y = radius * np.sin(theta)
    x = radius * np.cos(theta)

    # Display points in a scatter plot
    print(plot)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        plt.show()

    return np.array((x, y, z)).T


def euler_to_quaternion(r):
    yaw, pitch, roll = r[0], r[1], r[2]
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]


def quaternion_to_euler(q):
    (x, y, z, w) = (q[0], q[1], q[2], q[3])
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [roll, pitch, yaw]
