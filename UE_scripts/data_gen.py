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
IMG_DIM = np.array([1920, 1080])
avg_exec_time = []


# np.random.seed(0)

def get_color(idx):
    r = [0, 235, 104, 60, 6, 224, 197, 71, 55, 122, 212, 19, 77, 233, 206, 22, 39, 100, 52, 230, 42, 35, 107, 103, 17,
         236, 91, 166, 217, 82, 113, 244, 85, 148, 243, 96, 18, 193, 210, 92, 242, 238, 1, 191, 27, 251, 218, 67, 146,
         15, 229, 237, 188, 161, 33, 88, 3, 170, 160, 189, 111, 50, 41, 43, 106, 139, 129, 80, 222, 37, 211, 87, 181,
         232, 112, 5, 69, 151, 208, 57, 94, 95, 61, 154, 204, 9, 54, 150, 192, 248, 30, 97, 239, 10, 254, 59, 70, 46,
         53, 62, 162, 38, 75, 216, 76, 28, 110, 153, 175, 145, 201, 205, 187, 246, 25, 221, 207, 29, 172, 249, 90, 135,
         200, 240, 241, 253, 185, 134, 11, 101, 225, 190, 245, 157, 171, 184, 159, 167, 214, 83, 127, 99, 219, 234, 73,
         198, 156, 130, 108, 14, 120, 247, 137, 4, 177, 143, 116, 152, 115, 169, 32, 178, 142, 12, 195, 34, 78, 182, 65,
         105, 109, 7, 252, 133, 163, 164, 138, 36, 51, 168, 44, 45, 24, 125, 209, 144, 186, 89, 124, 79, 226, 86, 228,
         118, 180, 98, 174, 250, 47, 194, 2, 49, 56, 131, 183, 0, 199, 102, 48, 74, 196, 16, 155, 13, 23, 119, 141, 84,
         20, 40, 114, 223, 165, 227, 147, 128, 179, 215, 202, 136, 72, 117, 220, 8, 81, 149, 68, 173, 121, 231, 140,
         158, 63, 31, 176, 58, 26, 66, 21, 203, 123, 93, 64, 126, 132]

    g = [0, 223, 110, 29, 78, 199, 197, 146, 241, 109, 218, 51, 7, 106, 28, 133, 88, 145, 253, 92, 74, 57, 228, 166,
         248, 73, 167, 175, 94, 242, 128, 186, 151, 101, 190, 45, 103, 138, 40, 140, 230, 123, 134, 60, 247, 43, 130, 5,
         207, 149, 198, 97, 131, 107, 14, 79, 115, 55, 66, 164, 49, 63, 124, 18, 185, 243, 35, 174, 71, 220, 111, 205,
         36, 188, 172, 163, 26, 249, 27, 206, 91, 227, 233, 217, 77, 72, 89, 4, 52, 39, 231, 170, 0, 179, 19, 208, 224,
         251, 168, 121, 209, 142, 87, 65, 159, 143, 195, 24, 70, 160, 82, 56, 53, 41, 201, 100, 112, 210, 240, 104, 225,
         75, 236, 165, 90, 61, 48, 137, 127, 68, 222, 37, 126, 59, 136, 176, 6, 194, 116, 23, 162, 177, 196, 102, 157,
         129, 187, 238, 229, 252, 245, 148, 25, 150, 232, 17, 173, 54, 11, 81, 254, 213, 211, 1, 9, 214, 15, 85, 234,
         117, 125, 31, 22, 32, 237, 154, 139, 10, 212, 221, 113, 21, 86, 226, 120, 219, 12, 69, 108, 239, 246, 67, 93,
         135, 193, 132, 105, 141, 204, 184, 158, 46, 182, 189, 171, 215, 38, 119, 2, 203, 96, 64, 8, 3, 202, 200, 250,
         42, 118, 122, 155, 16, 114, 84, 98, 58, 153, 181, 216, 161, 178, 83, 44, 20, 147, 13, 191, 34, 47, 169, 95,
         183, 33, 62, 235, 192, 80, 76, 99, 156, 244, 50, 180, 152, 144]

    b = [0, 238, 78, 173, 51, 8, 112, 158, 72, 81, 55, 205, 239, 184, 253, 194, 23, 176, 119, 155, 218, 61, 122, 203,
         104, 226, 19, 220, 141, 86, 106, 71, 48, 250, 29, 195, 232, 206, 46, 75, 237, 129, 164, 125, 186, 18, 93, 95,
         147, 204, 124, 229, 28, 32, 53, 221, 241, 76, 102, 175, 85, 58, 80, 254, 178, 94, 79, 153, 143, 99, 188, 146,
         136, 4, 105, 137, 180, 114, 249, 117, 135, 25, 89, 82, 20, 128, 59, 161, 172, 33, 149, 121, 157, 111, 88, 14,
         87, 248, 144, 17, 151, 159, 9, 200, 208, 201, 21, 22, 52, 242, 251, 210, 54, 92, 224, 191, 223, 182, 69, 100,
         24, 192, 217, 64, 31, 90, 49, 148, 0, 5, 35, 156, 30, 62, 142, 107, 67, 96, 177, 7, 240, 139, 56, 138, 231,
         109, 12, 110, 212, 174, 189, 209, 230, 40, 216, 47, 98, 26, 165, 116, 97, 140, 181, 10, 63, 252, 123, 43, 74,
         207, 167, 166, 132, 70, 233, 134, 244, 196, 120, 73, 101, 193, 131, 84, 170, 34, 214, 50, 222, 236, 215, 133,
         65, 11, 234, 225, 202, 228, 247, 171, 243, 57, 199, 6, 45, 154, 66, 145, 152, 44, 197, 108, 118, 160, 77, 219,
         185, 130, 113, 235, 162, 213, 13, 211, 39, 227, 38, 163, 187, 103, 16, 198, 168, 41, 246, 60, 91, 150, 68, 37,
         36, 183, 245, 15, 1, 126, 27, 190, 169, 2, 115, 3, 127, 42, 83]
    return [r[idx], g[idx], b[idx]]


def jitter_rotator_z(val=5):
    jitter_val = val
    jitter = Rotator(0,
                     0,
                     np.random.uniform(-jitter_val, jitter_val))
    return jitter


def buoy_rotator():
    jitter = Rotator(np.random.uniform(0, 10),
                     np.random.uniform(0, 10),
                     np.random.uniform(0, 360),
                     )
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
        self.ocean = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Ocean_BP")[0]

        ### Objects
        self.buoy_g = ue.EditorFilterLibrary.by_actor_label(lst_actors, "BuoyGreen")

        self.buoy_r = ue.EditorFilterLibrary.by_actor_label(lst_actors, "BuoyRed")
        self.sail_u = ue.EditorFilterLibrary.by_actor_label(lst_actors, "SailU")
        self.sail_d = ue.EditorFilterLibrary.by_actor_label(lst_actors, "SailD")
        self.sl_guys = [ue.EditorFilterLibrary.by_actor_label(lst_actors, "Tommo")[0],
                        ue.EditorFilterLibrary.by_actor_label(lst_actors, "Frede")[0]]
        self.kayak = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Kayak")
        self.motors = ue.EditorFilterLibrary.by_actor_label(lst_actors, "MotorBoat")
        self.fishing = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Fishing")
        self.container = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Container")

        self.movable_actors = [(self.kayak, 2, "Kayak"), (self.sl_guys, 5, "Human"), (self.buoy_g, 4, "BuoyGreen"),
                               (self.buoy_r, 4, "BuoyRed"), (self.motors, 3, "MotorBoat"), (self.sail_d, 3, "SailD"),
                               (self.sail_u, 4, "SailU"), (self.fishing, 1, "Fishing"),
                               (self.container, 1, "Container")]

        self.movable_actors = [(self.kayak, 2, "Kayak"), (self.buoy_g, 4, "BuoyGreen"),
                               (self.buoy_r, 4, "BuoyRed"), (self.motors, 3, "MotorBoat"), (self.sail_d, 3, "SailD"),
                               (self.sail_u, 4, "SailU"), (self.fishing, 1, "Fishing")]

        self.tmp = []

    def _set_actor_ini_pos(self):
        self.cam.set_actor_location(Vector(0, 1300, 960), True, True)
        self.scene_cap.set_actor_location(Vector(0, 1300, 960), True, True)
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

        n = 10000
        xy_min = [-1, 4000]
        xy_max = [1, 30000] #in cm so 300 meters away
        self.sample_points = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))
        xy_min = [-1, 3000]
        xy_max = [1, 10000] # 100 meters away
        self.sample_points_close = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))
        xy_min = [-1, 20000]
        xy_max = [1, 100000]
        self.sample_points_far = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))

        self.n_samplings = 10000
        # # Get All possible combinations

        self.on_tick = ue.register_slate_post_tick_callback(self.__tick__)
        # self.on_post_tick = ue.register_slate_post_tick_callback(self.__posttick__)

        self.i, self.theta, self.ticks, self.start = 0, 0, 0, 0
        self.name, self.samples = [], []
        self.path = ue.Paths.screen_shot_dir()
        self.base_path = os.path.join(self.path, 'seg')
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        self.csv_path = os.path.join(self.base_path, "color_to_lbl.csv")
        with open(self.csv_path, "w") as file:
            file.write('')
        self.cam.camera_component.set_field_of_view(110)
        self.cam_fov = self.cam.camera_component.field_of_view
        self.lbl_cam.camera_component.set_field_of_view(110)
        self.lbl_cam_fov = self.cam.camera_component.field_of_view
        self.tock = True


    def take_step(self, deltatime):
        self.curr_sample = ("None", "None")
        self.start = datetime.now()
        object_list = []
        self.sky.set_editor_property("Cloud Coverage", random.uniform(0, 1.5))
        self.sky.set_editor_property("Time Of Day", random.uniform(700, 1650))
        self.sky.set_editor_property("Sun Angle", random.uniform(0, 360))
        # print(dir(self.ocean.water_waves))
        # self.ocean.water_waves.gerstner_wave_generator.min_amplitude = random.uniform(0.01, 8)
        # self.ocean.water_waves.gerstner_wave_generator.max_amplitude = random.uniform(10, 50)
        print(f"n: {self.i + 1} of {self.n_samplings}")

        num = random.randint(1, 3)
        cntr = 0
        obj_possos = []
        objects_in_frame = random.sample(self.movable_actors, num)
        for obj_l, cnt, name in objects_in_frame:
            num = random.randint(1, cnt)
            for _ in range(num):
                num = random.randint(1, len(obj_l)) - 1

                obj = obj_l[num]
                sample = self.random_sample_func(obj_possos, name)
                obj_possos.append(sample)
                pos = obj.get_actor_location()
                new_pos = Vector(sample[0] * sample[1], sample[1], pos.z)
                rot = jitter_rotator_z(360)
                if name in ["BuoyRed", "BuoyGreen"]:
                    rot = buoy_rotator()
                temp = ELL.spawn_actor_from_object(obj, new_pos, rot)

                tt = list(temp.get_components_by_class(ue.StaticMeshComponent))
                gg = list(temp.get_components_by_class(ue.SkeletalMeshComponent))
                j = tt + gg
                cntr += 1
                for i, x in enumerate(j):
                      x.set_editor_property("custom_depth_stencil_value", cntr)

                scale = temp.get_actor_scale3d() * random.uniform(.85, 1.15)
                temp.set_actor_scale3d(scale)
                temp.set_actor_hidden_in_game(False)

                self.tmp.append(temp)
                object_list.append([name, get_color(cntr)])
        # print(self.tmp)
        # BG Rot
        self.bg.set_actor_rotation(jitter_rotator_z(360), True)
        rot = jitter_rotator_all(3)
        self.cam.add_actor_local_rotation(rot, False, True)
        # self.lbl_cam.add_actor_local_rotation(rot, False, True)
        # self.scene_cap.add_actor_local_rotation(rot, False, True)
        # self.scene_cap_lbl.add_actor_local_rotation(rot, False, True)
        self.curr_sample = object_list  # s.append(object_list)
        # np.random.shuffle(self.sample_points)
        # np.random.shuffle(self.sample_points_close)


    def random_sample_func(self, current_pos, name):
        min_rad = 1000
        r = range(len(self.sample_points))
        sample = self.sample_points[random.sample(r, 1)[0]]
        close = ["BuoyRed", "BuoyGreen", "Human", "Kayak"]
        far = ["Container"]
        if name in close:
            sample = self.sample_points_close[random.sample(r, 1)[0]]
        if name in far:
            sample = self.sample_points_far[random.sample(r, 1)[0]]
        # print(sample)
        attempts = 0
        if len(current_pos) != 0:
            for _ in range(10):
                idx = random.sample(r, 1)[0]
                sample = self.sample_points[idx]
                if name in close:
                    sample = self.sample_points_close[idx]
                if name in far:
                    sample = self.sample_points_far[idx]
                all_bigger = 0
                for x in current_pos:
                    dist = np.sqrt(np.sum((sample - x) ** 2))
                    if dist < min_rad:
                        all_bigger += 1
                if all_bigger == 0:
                    break
                if attempts > 10:
                    print("figure something else out")
        return sample

    def __tick__(self, deltatime):
        try:
            if self.n_samplings > self.i:
                if keyboard.is_pressed('q'):
                    for t in self.tmp:
                        ELL.destroy_actor(t)
                    self.i = 800
                    assert False

                name = f"p_{self.i}_{self.cam_fov}.png"
                if self.tock:
                    self.take_step(deltatime)
                    self.tock = False
                    #self.scene_cap.capture_component2d.capture_scene()

                    mas_path = os.path.join(self.base_path, "mask")
                    if not os.path.exists(mas_path):
                        os.makedirs(mas_path)
                    val = IMG_DIM[0] / IMG_DIM[1]
                    self.lbl_cam.camera_component.set_constraint_aspect_ratio(True)
                    self.lbl_cam.camera_component.set_aspect_ratio(val)
                    # AL.take_high_res_screenshot(1280, 960, path2, self.lbl_cam)
                    self.scene_cap_lbl.capture_component2d.capture_scene()
                    ue.RenderingLibrary.export_render_target(self.scene_cap_lbl,
                                                             self.scene_cap_lbl.capture_component2d.texture_target,
                                                             mas_path, name)
                else:
                    img_path = os.path.join(self.base_path, "img")

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
                    avg_exec_time.append(datetime.now() - self.start)
                    avg_exec = np.mean(avg_exec_time)
                    est_remaining_time = avg_exec * (max(0, self.n_samplings - self.i))
                    print("avg exec time: {}, est time remaining: {}".format(avg_exec, est_remaining_time / 60))
                    for t in self.tmp:
                        ELL.destroy_actor(t)
                    self.scene_cap.capture_component2d.capture_scene()
                    with open(self.csv_path, 'a') as file:
                        file.write(f"{name},{self.curr_sample}\n")

                    self.tmp = []
                    self._set_actor_ini_pos()
                    self.i += 1
                    self.tock = True
            else:
                ue.unregister_slate_post_tick_callback(self.on_tick)
        except Exception as error:
            print(error)
            ue.unregister_slate_post_tick_callback(self.on_tick)


if __name__ == '__main__':
    instance = OnTick()
