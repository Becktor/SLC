import unreal as ue
from unreal import Vector, Rotator
from unreal import AutomationLibrary as AL
import pandas as pd
import itertools as it
import os
import sys
import numpy as np
from datetime import datetime

print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
from utils import *

METERS = 100
IMG_DIM = np.array([1280, 960])
avg_exec_time = []

np.random.seed(0)


def jitter_rotator(val=5):
    jitter_val = val
    jitter = Rotator(np.random.uniform(-jitter_val, jitter_val),
                     np.random.uniform(-jitter_val, jitter_val),
                     np.random.uniform(-jitter_val, jitter_val))
    return jitter


def jitter_locator(dist, fov):
    y_val = dist / np.tan(np.deg2rad(fov / 2))
    x_val = y_val * 0.75
    z_val = 500
    if dist == 10 * METERS:
        z_val = 200

    jitter = Vector(np.random.uniform(-x_val, x_val),
                    np.random.uniform(-y_val, y_val),
                    np.random.uniform(-z_val, z_val), )
    return jitter


class OnTick(object):

    def _setup_actors(self):
        self.actors = (actor for actor in ue.EditorLevelLibrary.get_selected_level_actors())
        lst_actors = ue.EditorLevelLibrary.get_all_level_actors()
        self.sat = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Soyuz")[0]
        self.cam = ue.EditorFilterLibrary.by_actor_label(lst_actors, "CameraActor")[0]
        self.lbl_cam = ue.EditorFilterLibrary.by_actor_label(lst_actors, "LabelCamera")[0]
        self.stars = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Space_SkyBox")[0]
        self.scene_cap = ue.EditorFilterLibrary.by_actor_label(lst_actors, "SceneCapture")[0]
        self.scene_cap_lbl = ue.EditorFilterLibrary.by_actor_label(lst_actors, "SceneCapture_label")[0]
        self.sun = ue.EditorFilterLibrary.by_actor_label(lst_actors, "DirectionalLight")[0]
        self.earth = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Earth")[0]
        self.moon = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Moon")[0]

    def _set_actor_ini_pos(self):
        self.sat.set_actor_location(Vector(0, 0, 1000 * METERS), True, True)
        self.cam.set_actor_location(Vector(0, 0, 990 * METERS), True, True)

        self.scene_cap.set_actor_location(Vector(0, 0, 990 * METERS), True, True)
        # self.scene_cap_lbl.set_actor_location(Vector(0, 0, 900 * METERS), True, True)

        self.cam.set_actor_rotation(Rotator(0, 90, 0), True)
        self.scene_cap.set_actor_rotation(Rotator(0, 90, 0), True)

        # set lbl cam
        pos = self.cam.get_actor_location()
        self.lbl_cam.set_actor_location(pos, True, True)

        rot = self.cam.get_actor_rotation()
        self.lbl_cam.set_actor_rotation(rot, True)

        self.earth_start_location = Vector(0, 0, 6872000 * METERS)
        self.earth.set_actor_location(self.earth_start_location, True, True)
        self.moon_start_location = Vector(0, 0, 384400000 * METERS)
        self.moon.set_actor_location(self.moon_start_location, True, True)

    def __init__(self):

        self._setup_actors()
        self._set_actor_ini_pos()

        self.earth_base_pos = [(0, 0), (5000000, 0),
                               (5000000, 5000000), (0, 5000000),
                               (-5000000, 5000000), (-5000000, 0),
                               (-5000000, -5000000), (0, -5000000)]

        # normal sampling
        r = np.arange(0, 360., 120)
        # test sampling
        # r = np.arange(-54, 306., 90)
        sat_sampling = list(it.product(r, r, r))

        # sun sampling
        # train sampling
        arr = np.arange(10, 171., 80)  # hard data
        # arr = np.arange(0, 180., 65)

        sun_sampling = list(it.product([0], arr, arr))
        # # Get sampling angles
        sun_angs = [Rotator(elem[0], elem[1], elem[2]) for elem in sun_sampling]
        sat_angs = [Rotator(elem[0], elem[1], elem[2]) for elem in sat_sampling]
        self.distances = [10 * METERS, 20 * METERS, 30 * METERS, 40 * METERS, 50 * METERS]

        # # Get All possible combinations
        self.sun_sat_angs_and_dist = list(it.product(self.distances, sun_angs, sat_angs, self.earth_base_pos))
        print(len(self.sun_sat_angs_and_dist))
        self.n_samplings = len(self.sun_sat_angs_and_dist)
        sample_rate = self.n_samplings // 5 if self.n_samplings > 10 else 10

        star_sampling = fibonacci_sphere(sample_rate, False)

        star_sampling = np.resize(star_sampling, [self.n_samplings, 3])

        self.star_sampling = [Vector(elem[0], elem[1], elem[2]).rotator() for elem in star_sampling]

        star_sampling = fibonacci_sphere(sample_rate, False)

        star_sampling = np.resize(star_sampling, [self.n_samplings, 3])

        self.star_sampling = [Vector(elem[0], elem[1], elem[2]).rotator() for elem in star_sampling]

        self.on_tick = ue.register_slate_post_tick_callback(self.__tick__)
        self.on_post_tick = ue.register_slate_post_tick_callback(self.__posttick__)

        self.time, self.i, self.theta, self.ticks, self.start = 0, 0, 0, 0, 0

        self.base_so_location = self.sat.get_actor_location()
        self.data = np.zeros((self.n_samplings, 7))
        self.name = []
        self.path = ue.Paths.screen_shot_dir()

        self.cam.camera_component.set_field_of_view(90)
        self.cam_fov = self.cam.camera_component.field_of_view
        self.lbl_cam.camera_component.set_field_of_view(90)
        self.lbl_cam_fov = self.cam.camera_component.field_of_view

    def __posttick__(self, deltatime):
        self.start = datetime.now()
        try:
            if self.n_samplings > self.i:
                print("n: {} of {}".format(self.time, self.n_samplings))
                if self.ticks % 2 == 0:
                    self.sun.set_actor_hidden_in_game(False)
                    distance, sun_angle, sat_angle, earth_pos = self.sun_sat_angs_and_dist[self.i]
                    jitter = jitter_rotator(45)
                    sat_angle = sat_angle.combine(jitter)
                    jloc = jitter_locator(distance, self.cam_fov)
                    new_sat_pos = Vector(0, 0, 1000 * METERS) + jloc
                    self.sat.set_actor_location(new_sat_pos, True, True)
                    self.sat.set_actor_rotation(sat_angle, True)

                    jitter = jitter_rotator(45)
                    sun_angle = sun_angle.combine(jitter)
                    self.sun.set_actor_rotation(sun_angle, True)

                    self.stars.set_actor_rotation(self.star_sampling[self.time], True)

                    # Moon Pos/Rot
                    moon_sampler = np.random.uniform(low=-50000000 * METERS, high=50000000 * METERS, size=2)
                    new_moon_pos = self.moon_start_location + Vector(moon_sampler[0], moon_sampler[1], 0)
                    self.moon.set_actor_location(new_moon_pos, True, True)
                    self.moon.set_actor_rotation(self.star_sampling[self.time], True)

                    # Earth pos/rot
                    earth_jitter = jitter_earth()
                    new_earth_pos = self.earth_start_location + \
                                    Vector(earth_pos[0] * METERS, earth_pos[1] * METERS, ) + earth_jitter
                    self.earth.set_actor_location(new_earth_pos, True, True)
                    self.earth.set_actor_rotation(self.star_sampling[self.time], True)

                    # Ego Pos
                    updated_pos = Vector(0, 0, 1000 * METERS - distance)
                    self.cam.set_actor_location(updated_pos, True, True)
                    self.lbl_cam.set_actor_location(updated_pos, True, True)
                    self.scene_cap.set_actor_location(updated_pos, True, True)

                    self.time += 1
                    self.i += 1
                    self.ticks += 1
                else:
                    self.ticks += 1
            else:
                ue.unregister_slate_post_tick_callback(self.on_post_tick)
        except Exception as error:
            print(error)
            ue.unregister_slate_post_tick_callback(self.on_post_tick)

    def __tick__(self, deltatime):
        try:
            if self.n_samplings > self.i:
                path = os.path.join(self.path, 'seg')
                if self.ticks % 2 == 0:
                    if self.time % 5 == 0:
                        self.earth.set_actor_hidden_in_game(True)
                    else:
                        self.earth.set_actor_hidden_in_game(False)

                    self.moon.set_actor_hidden_in_game(False)
                    path = os.path.join(path, "img")
                    name = "p_{}_{}.png".format(str(self.i), str(self.cam_fov))
                    # path2 = os.path.join(path, "2"+name)

                    if not os.path.exists(path):
                        os.makedirs(path)

                    val = IMG_DIM[0] / IMG_DIM[1]
                    self.cam.camera_component.set_constraint_aspect_ratio(True)
                    self.cam.camera_component.set_aspect_ratio(val)
                    # AL.take_high_res_screenshot(1280, 960, path2, self.cam)

                    self.scene_cap.capture_component2d.capture_scene()
                    ue.RenderingLibrary.export_render_target(self.scene_cap,
                                                             self.scene_cap.capture_component2d.texture_target,
                                                             path, name)
                    avg_exec_time.append(datetime.now() - self.start)
                    avg_exec = np.mean(avg_exec_time)
                    est_remaining_time = avg_exec * (self.n_samplings - self.time)
                    print("avg exec time: {}, est time remaining: {}".format(avg_exec, est_remaining_time / 60))

                else:
                    self.earth.set_actor_hidden_in_game(True)
                    self.moon.set_actor_hidden_in_game(True)
                    path = os.path.join(path, "mask")
                    if not os.path.exists(path):
                        os.makedirs(path)
                    name = "p_{}_{}.png".format(str(self.i), str(self.cam_fov))
                    # path2 = os.path.join(path, "2" + name)
                    val = IMG_DIM[0] / IMG_DIM[1]
                    self.lbl_cam.camera_component.set_constraint_aspect_ratio(True)
                    self.lbl_cam.camera_component.set_aspect_ratio(val)
                    # AL.take_high_res_screenshot(1280, 960, path2, self.lbl_cam)
                    self.scene_cap_lbl.capture_component2d.capture_scene()
                    ue.RenderingLibrary.export_render_target(self.scene_cap_lbl,
                                                             self.scene_cap_lbl.capture_component2d.texture_target,
                                                             path, name)
                    avg_exec_time.append(datetime.now() - self.start)
                    avg_exec = np.mean(avg_exec_time)
                    est_remaining_time = avg_exec * (self.n_samplings - self.time)
                    print("avg exec time: {}, est time remaining: {}".format(avg_exec, est_remaining_time / 60))
        except Exception as error:
            print(error)
            ue.unregister_slate_post_tick_callback(self.on_tick)


if __name__ == '__main__':
    instance = OnTick()
