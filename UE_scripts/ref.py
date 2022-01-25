import unreal as ue
from unreal import Vector, Rotator
from unreal import AutomationLibrary as AL
from unreal import SystemLibrary as SL
from unreal import GameplayStatics as GS

import pandas as pd
import itertools as it
import os
import time
import sys
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
print(os.path.dirname(__file__))
from utils import jitter_locator, fibonacci_sphere, track_target

METERS = 100
np.set_printoptions(suppress=True)
IMG_DIM = np.array([4000, 3000])
IMG_OFFSET = np.array([IMG_DIM[0] / 2, IMG_DIM[1] / 2, 0])
avg_exec_time = []

np.random.seed(0)


def apply_project_cam_matrix(vec, fov):
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


class OnTick(object):

    def _setup_actors(self):
        self.world = ue.EditorLevelLibrary.get_editor_world()
        self.actors = (actor for actor in ue.EditorLevelLibrary.get_selected_level_actors())
        lst_actors = ue.EditorLevelLibrary.get_all_level_actors()
        self.sat = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Soyuz")[0]
        self.cam = ue.EditorFilterLibrary.by_actor_label(lst_actors, "CameraActor")[0]
        self.lbl_cam = ue.EditorFilterLibrary.by_actor_label(lst_actors, "LabelCamera")[0]
        self.stars = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Space_SkyBox")[0]
        self.scene_cap = ue.EditorFilterLibrary.by_actor_label(lst_actors, "SceneCapture_12MP")[0]
        self.scene_cap_lbl = ue.EditorFilterLibrary.by_actor_label(lst_actors, "SceneCapture_label_12MP")[0]
        self.sun = ue.EditorFilterLibrary.by_actor_label(lst_actors, "DirectionalLight")[0]
        self.earth = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Earth")[0]
        self.moon = ue.EditorFilterLibrary.by_actor_label(lst_actors, "Moon")[0]

        lst_comps = self.sat.get_components_by_class(ue.StaticMeshComponent)
        self.anchors = dict(KP1=ue.EditorFilterLibrary.by_id_name(lst_comps, "KP1")[0],
                            KP2=ue.EditorFilterLibrary.by_id_name(lst_comps, "KP2")[0],
                            KP3=ue.EditorFilterLibrary.by_id_name(lst_comps, "KP3")[0],
                            KP4=ue.EditorFilterLibrary.by_id_name(lst_comps, "KP4")[0],
                            KP5=ue.EditorFilterLibrary.by_id_name(lst_comps, "KP5")[0],
                            KP6=ue.EditorFilterLibrary.by_id_name(lst_comps, "KP6")[0],
                            KP7=ue.EditorFilterLibrary.by_id_name(lst_comps, "KP7")[0],
                            KP8=ue.EditorFilterLibrary.by_id_name(lst_comps, "KP8")[0],
                            KP9=ue.EditorFilterLibrary.by_id_name(lst_comps, "KP9")[0],
                            KP10=ue.EditorFilterLibrary.by_id_name(lst_comps, "KP10")[0],
                            KP11=ue.EditorFilterLibrary.by_id_name(lst_comps, "KP11")[0],
                            KP12=ue.EditorFilterLibrary.by_id_name(lst_comps, "KP12")[0],
                            KP13=ue.EditorFilterLibrary.by_id_name(lst_comps, "KP13")[0],
                            KP14=ue.EditorFilterLibrary.by_id_name(lst_comps, "KP14")[0],
                            KP15=ue.EditorFilterLibrary.by_id_name(lst_comps, "KP15")[0], )

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

        self.earth_start_location = Vector(-500000000, 700000000, 6872000 * METERS)
        self.earth.set_actor_location(self.earth_start_location, True, True)
        self.moon_start_location = Vector(0, 0, 384400000 * METERS)
        self.moon.set_actor_location(self.moon_start_location, True, True)
        self.start_location = Vector(-1 * METERS, -1 * METERS, 1000 * METERS)
        self.sat.set_actor_location(self.start_location, True, True)

        self.earth.set_actor_hidden_in_game(False)
        self.moon.set_actor_hidden_in_game(False)

    def __init__(self):
        self._setup_actors()
        self._set_actor_ini_pos()

        steps = range(0, 100, 1)
        xyz = []
        x1, x2, x3, x4 = 0, 0, 0, 0
        y1, y2, y3, y4 = 0, 0, 0, 0
        z1, z2, z3, z4 = 0, 0, 0, 0
        for iter in steps:
            x1 = iter * 0.02 if iter > len(steps) / 2 else (len(steps) - iter) * 0.02
            y1 = iter * 0.15
            z1 = iter * 0.15
            xyz.append((x1, y1, z1))

        for iter in steps:
            x2 = x1 + iter * 0.15
            y2 = (len(steps) - iter) * 0.1 if iter >= len(steps) / 2 else iter * 0.1
            y2 += y1
            z2 = iter * 0.15 + z1
            xyz.append((x2, y2, z2))

        for iter in steps:
            x3 = (len(steps) - iter) * 0.1 if iter > len(steps) / 2 else iter * 0.1
            x3 += x2
            y3 = y2 - iter * 0.2
            z3 = iter * 0.15 + z2
            xyz.append((x3, y3, z3))

        for iter in steps:
            x4 = x3 - iter * 0.25
            y4 = (len(steps) - iter) * 0.15 if iter > len(steps) / 2 else iter * 0.15
            y4 = y3 - y4
            z4 = iter * 0.15 + z3
            xyz.append((x4, y4, z4))

        list(zip(steps, steps, steps))
        self.steppers = xyz
        self.rotation = Rotator(2.5, 1.5, .5)

        self.sun_angles = self.sun.get_actor_rotation()
        # # Get All possible combinations

        self.n_samplings = len(self.steppers)
        sample_rate = self.n_samplings // 5 if self.n_samplings > 10 else 10

        star_sampling = fibonacci_sphere(sample_rate, False)

        star_sampling = np.resize(star_sampling, [self.n_samplings, 3])

        self.star_sampling = [Vector(elem[0], elem[1], elem[2]).rotator() for elem in star_sampling]

        self.on_tick = ue.register_slate_post_tick_callback(self.__tick__)
        self.on_post_tick = ue.register_slate_post_tick_callback(self.__posttick__)
        self.time, self.ticks, self.curr_dist = 0, 0, 10 * METERS

        self.cam.camera_component.set_field_of_view(90)
        self.cam_fov = self.cam.camera_component.field_of_view
        self.base_so_location = self.sat.get_actor_location()
        self.data, self.sat_rot_pos, self.d2 = {}, {}, {}
        self.name, self.sun_angles = [], []
        self.start = datetime.now()
        self.path = ue.Paths.screen_shot_dir()

    def relative_location(self, point, cam_trans):

        loc = cam_trans.inverse_transform_location(point)
        # coordinates in image plane as camera axis are (Y,Z,X) -> to normal (X,Y,Z)
        relative_point = (np.array([loc.y, -loc.z, loc.x]) / METERS)
        return relative_point

    def proj_relative_location(self, point, cam_trans, fov):
        loc = cam_trans.inverse_transform_location(point)
        # coordinates in image plane as camera axis are (Y,Z,X) -> to normal (X,Y,Z)
        relative_point = (np.array([loc.y, -loc.z, loc.x]) / METERS)
        relative_point = apply_project_cam_matrix(relative_point, fov).round(1)
        return relative_point

    def anchors_relative_location(self, cam_trans):
        anchor_dict = {}
        for key in self.anchors:
            anchor = self.anchors[key]
            loc = anchor.get_world_location()
            loc = cam_trans.inverse_transform_location(loc)
            # coordinates in image plane as camera axis are (Y,Z,X) -> to normal (X,Y,Z)
            anchor_dict[key] = (np.array([loc.y, -loc.z, loc.x]) / METERS)
        return anchor_dict

    def proj_anchors_relative_location(self, cam_trans, fov):
        anchor_dict, anchor_dict_proj = {}, {}
        visible_dict = {}
        for key in self.anchors:
            anchor = self.anchors[key]
            loc = anchor.get_world_location()
            loc = cam_trans.inverse_transform_location(loc)
            # coordinates in image plane with camera is (Y,-Z,X) -> to normal (X,Y,Z)
            anchor_dict[key] = (np.array([loc.y, -loc.z, loc.x]) / METERS)
            anchor_dict_proj[key] = apply_project_cam_matrix(anchor_dict[key], fov).round(1)
            xy = []

            ## visibility
            cam_loc = self.cam.get_actor_location()
            line = line_trace(self.world, anchor, cam_loc)
            if line:
                for x in line:
                    tup = x.to_tuple()
                    # 4 is impact point
                    xy.append(np.array([tup[4].x, tup[4].y, tup[4].z]))

                dist = 100
                if len(xy) == 2:
                    dist = np.mean(np.abs(xy[0] - xy[1]))

                visible = 0 if len(line) >= 2 and dist > 2.5 else 1
                anchor_dict_proj[key] = np.concatenate([anchor_dict_proj[key], [visible]])
            else:
                anchor_dict_proj[key] = np.concatenate([anchor_dict_proj[key], [1]]).tolist()
        return anchor_dict_proj, visible_dict, anchor_dict

    def __posttick__(self, deltatime):

        self.start = datetime.now()
        try:
            if self.n_samplings > self.time:

                print("n: {} of {}".format(self.time, self.n_samplings))
                x, y, z = self.steppers[self.time]
                ## Update Postions
                new_sat_pos = self.start_location + Vector(x * METERS, y * METERS, z * METERS)
                tst = self.sat.set_actor_location(new_sat_pos, False, False)
                cam_trans = self.cam.get_actor_transform()
                self.sat.add_actor_local_rotation(self.rotation, True, True)

                ## Get data
                anchor_positions, visible, ap = self.proj_anchors_relative_location(cam_trans, self.cam_fov)
                self.data[self.time] = anchor_positions
                self.d2[self.time] = ap
                r_trans = self.sat.get_actor_transform()
                print(r_trans.to_matrix())
                print(self.sat.get_actor_rotation())
                q = r_trans.rotation
                translation = self.relative_location(r_trans.translation, cam_trans)
                self.sat_rot_pos[self.time] = {"w": q.w, "i": q.x, "j": q.y, "k": q.z,
                                               "x": translation[0], "y": translation[1], "z": translation[2]}

                sun_rot_cam_space = cam_trans.inverse_transform_rotation(self.sun_angles)

                self.data[self.time]['roll'] = sun_rot_cam_space.roll
                self.data[self.time]['pitch'] = sun_rot_cam_space.pitch
                self.data[self.time]['yaw'] = sun_rot_cam_space.yaw
            else:

                data_df = pd.DataFrame.from_dict(self.data, orient='index')
                name = data_df['name']
                data_df.drop(labels=['name'], axis=1, inplace=True)
                data_df.insert(0, 'name', name)
                data_df.to_csv(os.path.join(self.path, 'Data.csv'), index=False,
                               header=False)
                print("data Saved")

                data_df = pd.DataFrame.from_dict(self.d2, orient='index')
                data_df.insert(0, 'name', name)
                data_df.to_csv(os.path.join(self.path, 'Data2.csv'), index=False,
                               header=False)
                # names_df = pd.DataFrame(self.name, columns=['name'])
                # target_df = pd.concat([names_df.reset_index(drop=True), data_df.reset_index(drop=True)], axis=1)
                # target_df.to_csv(os.path.join(self.path, 'Target_{}.csv'.format(self.distances)), index=False,
                #                  header=True)
                # print("saved Target")
                # sun_ang_df = pd.DataFrame(self.sun_angles, columns=['roll', 'pitch', 'yaw'])
                # sun_ang_df = pd.concat([names_df.reset_index(drop=True), sun_ang_df.reset_index(drop=True)], axis=1)
                # sun_ang_df.to_csv(os.path.join(self.path, 'SunAngles.csv'), index=False, header=False)
                # print("Sun Angles Saved")

                # save quats
                r_t_df = pd.DataFrame.from_dict(self.sat_rot_pos, orient='index')
                r_t_df.insert(0, 'name', name)
                # r_t_df = pd.concat([names_df.reset_index(drop=True), r_t_df.reset_index(drop=True)], axis=1)
                r_t_df.to_csv(os.path.join(self.path, 'rot_trans.csv'), index=False,
                              header=False)
                print("quat_saved")

                # all_data_df = pd.concat([sun_ang_df.reset_index(drop=True), data_df.reset_index(drop=True)], axis=1)
                # all_data_df.to_csv(os.path.join(self.path, 'JoinedData.csv'), index=False, header=False)
                # print("AD Saved")
                # print(self.time)
                ue.unregister_slate_post_tick_callback(self.on_post_tick)
        except Exception as error:
            print(error)
            ue.unregister_slate_post_tick_callback(self.on_post_tick)

    def __tick__(self, deltatime):
        try:
            if self.n_samplings > self.time:
                print("Post tick -- Saving Screenshot for n: {}".format(self.time))
                path = os.path.join(self.path, 'video')
                if not os.path.exists(path):
                    os.makedirs(path)

                name = "p_{}_{}.png".format(self.time, self.curr_dist)
                val = IMG_DIM[0] / IMG_DIM[1]
                #self.cam.camera_component.set_constraint_aspect_ratio(True)
                #self.cam.camera_component.set_aspect_ratio(val)
                self.data[self.time]['name'] = name
                # self.sat_rot_pos[self.time]['name'] = name
                self.name.append(name)
                # file = os.path.join('images', name)
                # AL.take_high_res_screenshot(int(IMG_DIM[0]), int(IMG_DIM[1]), file, self.cam)
                self.scene_cap.capture_component2d.capture_scene()
                ue.RenderingLibrary.export_render_target(self.scene_cap,
                                                         self.scene_cap.capture_component2d.texture_target,
                                                         path, name)

                self.time += 1
                avg_exec_time.append(datetime.now() - self.start)
                avg_exec = np.mean(avg_exec_time)
                est_remaining_time = avg_exec * (self.n_samplings - self.time)
                print("avg exec time: {}, est time remaining: {}".format(avg_exec, est_remaining_time))
        except Exception as error:
            print(error)
            ue.unregister_slate_post_tick_callback(self.on_tick)


if __name__ == '__main__':
    instance = OnTick()
