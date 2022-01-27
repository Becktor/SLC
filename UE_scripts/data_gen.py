import random

import unreal as ue
from unreal import Vector, Rotator
from unreal import AutomationLibrary as AL
from unreal import EditorLevelLibrary as ELL
import itertools as it
import os
import sys
import numpy as np
from datetime import datetime
import keyboard

print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
from utils import *

METERS = 100
IMG_DIM = np.array([1280, 960])
avg_exec_time = []


# np.random.seed(0)


def jitter_rotator_z(val=5):
    jitter_val = val
    jitter = Rotator(0,
                     0,
                     np.random.uniform(-jitter_val, jitter_val))
    return jitter


def jitter_rotator_all(val=5):
    jitter_val = val
    jitter = Rotator(np.random.uniform(-jitter_val, jitter_val),
                     np.random.uniform(-jitter_val, jitter_val),
                     np.random.uniform(-jitter_val, jitter_val))
    return jitter


def jitter_locator(val=1000):
    jitter = Vector(np.random.uniform(-val, val),
                    np.random.uniform(-val, val),
                    0)
    return jitter


class OnTick(object):

    def _setup_actors(self):
        self.actors = (actor for actor in ue.EditorLevelLibrary.get_selected_level_actors())
        lst_actors = ue.EditorLevelLibrary.get_all_level_actors()
        self.cam = ue.EditorFilterLibrary.by_actor_label(lst_actors, "CameraActor")[0]
        self.lbl_cam = ue.EditorFilterLibrary.by_actor_label(lst_actors, "LabelCamera")[0]
        self.sky = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Ultra_Dynamic_Sky")[0]
        self.weather = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Ultra_Dynamic_Weather")[0]
        self.scene_cap = ue.EditorFilterLibrary.by_actor_label(lst_actors, "SceneCapture")[0]
        self.scene_cap_lbl = ue.EditorFilterLibrary.by_actor_label(lst_actors, "LabelCapture")[0]
        self.bg = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Background_resource")[0]
        self.buoy_g = ue.EditorFilterLibrary.by_actor_label(lst_actors, "BuoyGreen")[0]
        self.buoy_r = ue.EditorFilterLibrary.by_actor_label(lst_actors, "BuoyRed")[0]
        self.sail_u = ue.EditorFilterLibrary.by_actor_label(lst_actors, "SailU")[0]
        self.sail_d = ue.EditorFilterLibrary.by_actor_label(lst_actors, "SailD")[0]
        self.motor = ue.EditorFilterLibrary.by_actor_label(lst_actors, "MotorBoat")[0]
        self.fishing = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Fishing")[0]
        self.ocean = ue.EditorFilterLibrary.by_actor_label(lst_actors, "OceanBp")[0]

        self.movable_actors = [(self.buoy_g, 4), (self.buoy_r, 4), (self.motor, 2), (self.sail_d, 3), (self.sail_u, 4),
                               (self.fishing, 2)]
        self.orig_actor_scales = []
        for obj,_ in self.movable_actors:
            self.orig_actor_scales.append(obj.get_actor_scale3d())
        self.tmp = []

    def _set_actor_ini_pos(self):
        self.cam.set_actor_location(Vector(0, 800, 960), True, True)
        self.scene_cap.set_actor_location(Vector(0, 800, 960), True, True)
        self.cam.set_actor_rotation(Rotator(0, 0, 90), True)
        self.scene_cap.set_actor_rotation(Rotator(0, 0, 90), True)

        # set lbl cam
        pos = self.cam.get_actor_location()
        self.lbl_cam.set_actor_location(pos, True, True)

        rot = self.cam.get_actor_rotation()
        self.lbl_cam.set_actor_rotation(rot, True)

        self.bg.set_actor_rotation(Rotator(0, 0, 0), True)

    def __init__(self):
        self._setup_actors()
        self._set_actor_ini_pos()

        n = 1000000
        xy_min = [-8000, 2000]
        xy_max = [8000, 30000]
        self.sample_points = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))

        self.n_samplings = 100
        # # Get All possible combinations

        self.on_tick = ue.register_slate_post_tick_callback(self.__tick__)
        # self.on_post_tick = ue.register_slate_post_tick_callback(self.__posttick__)

        self.i, self.theta, self.ticks, self.start = 0, 0, 0, 0
        self.name = []
        self.path = ue.Paths.screen_shot_dir()

        self.cam.camera_component.set_field_of_view(110)
        self.cam_fov = self.cam.camera_component.field_of_view
        self.lbl_cam.camera_component.set_field_of_view(110)
        self.lbl_cam_fov = self.cam.camera_component.field_of_view

    def take_step(self, deltatime):
        self.start = datetime.now()

        if self.n_samplings >= self.i:
            if keyboard.is_pressed('q'):
                for t in self.tmp:
                    ELL.destroy_actor(t)
                self.i = 800
                assert False
            self.sky.set_editor_property("Cloud Coverage", random.uniform(0, 2))
            self.sky.set_editor_property("Time Of Day", random.uniform(600, 1750))
            self.sky.set_editor_property("Sun Angle", random.uniform(0, 360))
            # print(dir(self.ocean.water_waves))
            # self.ocean.water_waves.gerstner_wave_generator.min_amplitude = random.uniform(0.01, 8)
            # self.ocean.water_waves.gerstner_wave_generator.max_amplitude = random.uniform(10, 50)
            print("n: {} of {}".format(self.i, self.n_samplings))

            num = random.randint(1, 4)
            objects_in_frame = random.sample(self.movable_actors, num)
            for obj, cnt in objects_in_frame:
                num = random.randint(1, cnt)
                samples = random.sample(range(len(self.sample_points)), num)
                for s in samples:
                    sample = self.sample_points[s]
                    pos = obj.get_actor_location()
                    new_pos = Vector(sample[0], sample[1], pos.z) + jitter_locator()
                    # obj.set_actor_location(new_pos, False, True)
                    # obj.set_actor_hidden_in_game(False)
                    # obj.set_actor_rotation(jitter_rotator_z(360), True)

                    temp = ELL.spawn_actor_from_object(obj, new_pos, jitter_rotator_z(360))
                    scale = temp.get_actor_scale3d() * random.uniform(1.05, 1.10)
                    temp.set_actor_scale3d(scale)
                    temp.set_actor_hidden_in_game(False)
                    self.tmp.append(temp)
            # print(self.tmp)
            # BG Rot
            self.bg.set_actor_rotation(jitter_rotator_z(360), True)
            rot = jitter_rotator_all(2)
            self.cam.add_actor_local_rotation(rot, False, True)
            # self.lbl_cam.add_actor_local_rotation(rot, False, True)
            # self.scene_cap.add_actor_local_rotation(rot, False, True)
            # self.scene_cap_lbl.add_actor_local_rotation(rot, False, True)

    def __tick__(self, deltatime):
        self.take_step(deltatime)
        try:
            if self.n_samplings >= self.i - 1:
                if keyboard.is_pressed('q'):
                    for t in self.tmp:
                        ELL.destroy_actor(t)
                    self.i = 800
                    assert False
                base_path = os.path.join(self.path, 'seg')
                img_path = os.path.join(base_path, "img")
                name = "p_{}_{}.png".format(str(self.i), str(self.cam_fov))
                # path2 = os.path.join(path, "2"+name)

                if not os.path.exists(img_path):
                    os.makedirs(img_path)

                val = IMG_DIM[0] / IMG_DIM[1]
                self.cam.camera_component.set_constraint_aspect_ratio(True)
                self.cam.camera_component.set_aspect_ratio(val)
                # AL.take_high_res_screenshot(1280, 960, path2, self.cam)

                self.scene_cap.capture_component2d.capture_scene()
                ue.RenderingLibrary.export_render_target(self.scene_cap,
                                                         self.scene_cap.capture_component2d.texture_target,
                                                         img_path, name)

                mas_path = os.path.join(base_path, "mask")
                if not os.path.exists(mas_path):
                    os.makedirs(mas_path)
                name = "p_{}_{}.png".format(str(self.i), str(self.cam_fov))
                # path2 = os.path.join(path, "2" + name)
                val = IMG_DIM[0] / IMG_DIM[1]
                self.lbl_cam.camera_component.set_constraint_aspect_ratio(True)
                self.lbl_cam.camera_component.set_aspect_ratio(val)
                # AL.take_high_res_screenshot(1280, 960, path2, self.lbl_cam)
                self.scene_cap_lbl.capture_component2d.capture_scene()
                ue.RenderingLibrary.export_render_target(self.scene_cap_lbl,
                                                         self.scene_cap_lbl.capture_component2d.texture_target,
                                                         mas_path, name)
                # print(path, name)
                avg_exec_time.append(datetime.now() - self.start)
                avg_exec = np.mean(avg_exec_time)
                est_remaining_time = avg_exec * (min(0, self.n_samplings - self.i))
                print("avg exec time: {}, est time remaining: {}".format(avg_exec, est_remaining_time / 60))
                for t in self.tmp:
                    ELL.destroy_actor(t)

                # for obj, scale in zip(self.movable_actors, self.orig_actor_scales):
                #     obj.set_actor_hidden_in_game(True)
                #     obj.set_actor_scale3d(scale)
                # print('t',self.tmp)

                self.tmp = []
                self._set_actor_ini_pos()
                self.i += 1
            else:
                ue.unregister_slate_post_tick_callback(self.on_tick)
        except Exception as error:
            print(error)
            ue.unregister_slate_post_tick_callback(self.on_tick)


if __name__ == '__main__':
    instance = OnTick()
