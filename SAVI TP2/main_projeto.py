#!/usr/bin/env python3

import copy
import math
import os
import cv2
import pyttsx3 
import random
import argparse
import open3d as o3d
import numpy as np
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from gtts import gTTS
from Classifier import Classifier
import playsound 
from PIL import Image
from point_cloud_processing_projeto import PointCloudProcessing
from matplotlib import cm
from more_itertools import locate
from colorama import Fore, Back, Style
from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)

# Default view
view = {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 0.69999999999999996, 0.69999999999999996, 0.5 ],
                    "boundingbox_min" : [ -0.69999999999999996, -0.69999999999999996, -0.25 ],
                    "field_of_view" : 60.0,
                    "front" : [ 0.48357815199197129, 0.61548483867363935, 0.62235888704100151 ],
                    "lookat" : [ 0.25470084286906458, 0.23151583259577294, 0.25384666908559167 ],
                    "up" : [ -0.40379961065115821, -0.47397267466848536, 0.78249330866504885 ],
                    "zoom" : 0.87999999999999901
                }
            ],
            "version_major" : 1,
            "version_minor" : 0
        }


# Drawing result
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
    

# Convert color RGB values to color name
def convert_rgb_to_names(rgb_tuple):
    
    # Dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []    
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)    
    distance, index = kdt_db.query(rgb_tuple)
    return f'{names[index]}'


def speak(text):
    tts = gTTS(text=text, lang='en')

    filename = "tts.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)


class PlaneDetection:
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud

    def colorizeInliers(self, r, g, b):
        self.inlier_cloud.paint_uniform_color([r, g, b])

    # Find plane
    def segment(self, distance_threshold=0.05, ransac_n=5, num_iterations=100):

        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model

        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)

        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=False)

        return outlier_cloud

    def __str__(self):
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) + ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0'
        return text
    

def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------

    # Setting up the arguments
    parser = argparse.ArgumentParser(description='PointCloud Scene Processor used to detect items on top of a table and identify them.')  # arguments
    parser.add_argument('-s', '--scene',metavar = 'SCENE_NUMBER', type=int, choices=range(1, 15), 
                        help='Choose a scene from 1 to 14. Do not use argument to use a random scene.\n ')
    parser.add_argument('-tts', '--text_to_speech', action='store_true', default=False, 
                        help='Use text to speech. default=False\n ')
    parser.add_argument('-v', '--show_visualization', action='store_true', default=False, 
                        help='Show visualization windows. default=False\n ')
    args = vars(parser.parse_args())

    scene_number_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14']

    p = PointCloudProcessing()

    # Get scene datapath
    if args['scene'] is None:
        scene_datapath = 'Scenes/rgbd-scenes-v2/pc/' + random.choice(scene_number_list) + '.ply'
    else:
        if args['scene'] < 10:
            scene_datapath = 'Scenes/rgbd-scenes-v2/pc/0' + str(args['scene']) + '.ply'
        else:
            scene_datapath = 'Scenes/rgbd-scenes-v2/pc/' + str(args['scene']) + '.ply'

    # Load PointCloud scene
    p.loadPointCloud(scene_datapath)

    # Resolution of point cloud
    p.preProcess(voxel_size=0.009)

    # ------------------------------------------
    # Find table plane
    # ------------------------------------------

    point_cloud_original = o3d.io.read_point_cloud(scene_datapath)

    # Estimate normals
    point_cloud_original.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=25))
    point_cloud_original.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))

    # Angle tolerance verification
    angle_tolerance = 0.1
    vx, vy, vz = 1, 0, 0
    norm_b = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    horizontal_idxs = []
    for idx, normal in enumerate(point_cloud_original.normals):

        nx, ny, nz = normal
        ab = nx*vx + ny*vy + nz*vz
        norm_a = math.sqrt(nx**2 + ny**2 + nz**2)
        angle = math.acos(ab/(norm_a * norm_b)) * 180/math.pi

        if abs(angle - 90) < angle_tolerance:
            horizontal_idxs.append(idx)

    # Get horizontal pointcloud
    horizontal_cloud = point_cloud_original.select_by_index(horizontal_idxs)
    non_horizontal_cloud = point_cloud_original.select_by_index(horizontal_idxs, invert=True)

    horizontal_cloud.paint_uniform_color([0.5, 0, 1])

    # Remove unwanted points
    (table_point_cloud, lista) = horizontal_cloud.remove_radius_outlier(150, 0.3)

    # Get table plane
    table_plane = PlaneDetection(table_point_cloud)
    table_plane_point_cloud = table_plane.segment()

    # Get table center
    table_center = table_plane_point_cloud.get_center()

    # Positioning coordinate axis in the middle of table
    p.transform(0,0,0,-table_center[0],-table_center[1],-table_center[2])
    p.transform(-120,0,0,0,0,0)
    p.transform(0,0,-120,0,0,0)
    p.transform(0,-7,0,0,0,0)

    # Cropping point cloud
    p.crop(-0.6, -0.5, 0, 0.6, 0.5, 0.5)

    # Find plane
    outliers = p.findPlane()
    
    # ------------------------------------------
    # Clustering
    # ------------------------------------------

    cluster_idxs = list(outliers.cluster_dbscan(eps=0.031, min_points=50, print_progress=True))
    object_idxs = list(set(cluster_idxs))
    object_idxs.remove(-1)

    number_of_objects = len(object_idxs)
    colormap = cm.Pastel1(list(range(0,number_of_objects)))

    # Objects on the table
    objects = []
    for object_idx in object_idxs:

        object_point_idxs = list(locate(cluster_idxs, lambda x: x == object_idx))
        object_points = outliers.select_by_index(object_point_idxs)

        # Create a dictionary to represent the objects
        d = {}
        d['idx'] = str(object_idx)
        d['points'] = object_points
        d['color'] = colormap[object_idx, 0:3]
        d['points'].paint_uniform_color(d['color'])
        d['center'] = d['points'].get_center()

        real_color = [0,0,0]

        for color in np.asarray(object_points.colors):
            
            real_color = real_color + color

        real_color = real_color/np.asarray(object_points.colors).shape[0]
        
        d['real_color'] = convert_rgb_to_names(real_color)

        # Get max and min from objects
        max_bound = d['points'].get_max_bound()
        min_bound = d['points'].get_min_bound()

        # Get length
        if (min_bound[0] < 0) and (max_bound[0] > 0):
            d['length'] = abs(min_bound[0]) + max_bound[0]
        else:
            d['length'] = abs(min_bound[0] + max_bound[0])

        # Get width
        if (min_bound[1] < 0) and (max_bound[1] > 0):
            d['width'] = abs(min_bound[1]) + max_bound[1]
        else:
            d['width'] = abs(min_bound[1] + max_bound[1])

        # Get height
        if (min_bound[2] < 0) and (max_bound[2] > 0):
            d['height'] = abs(min_bound[2]) + max_bound[2]
        else:
            d['height'] = abs(min_bound[2] + max_bound[2])

        # Add the dict of the object to the list
        objects.append(d)
    
    # cereal_box_model = o3d.io.read_point_cloud('cereal_box_2_2_40.pcd')

    # for object_idx, object in enumerate(objects):
    #     print("Apply point-to-point ICP to object " + str(object['idx']) )

    #     trans_init = np.asarray([[1, 0, 0, 0],
    #                              [0,1,0,0],
    #                              [0,0,1,0], 
    #                              [0.0, 0.0, 0.0, 1.0]])
    #     reg_p2p = o3d.pipelines.registration.registration_icp(cereal_box_model, 
    #                                                           object['points'], 2, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
    #     print(reg_p2p.inlier_rmse)
    #     object['rmse'] = reg_p2p.inlier_rmse
    #     # draw_registration_result(cereal_box_model, object['points'], reg_p2p.transformation)

    # # How to classify the object. Use the smallest fitting to decide which object is a "cereal box"
    # minimum_rmse = 10e8 # just a very large number to start
    # cereal_box_object_idx = None

    # for object_idx, object in enumerate(objects):
    #     if object['rmse'] < minimum_rmse: # Found a new minimum
    #         minimum_rmse = object['rmse']
    #         cereal_box_object_idx = object_idx

    # print('The cereal box is object ' + str(cereal_box_object_idx))

    if args['show_visualization'] == True:

        # ------------------------------------------
        # Visualization
        # ------------------------------------------

        # Create a list of entities to draw
        p.inliers.paint_uniform_color([0,1,1])
        entities = []

        frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
        entities.append(frame)

        # Draw bbox
        bbox_to_draw = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(p.bbox)
        entities.append(bbox_to_draw)

        # Draw objects
        for object_idx, object in enumerate(objects):
            entities.append(object['points'])

        # Make a more complex open3D window to show object labels on top of 3d
        app = gui.Application.instance

        # Create an open3d app
        app.initialize() 

        w = app.create_window("Open3D - 3D Text", 1920, 1080)
        widget3d = gui.SceneWidget()
        widget3d.scene = rendering.Open3DScene(w.renderer)
        widget3d.scene.set_background([0,0,0,1])  # Set black background
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 2 * w.scaling

        # Draw entities
        for entity_idx, entity in enumerate(entities):
            widget3d.scene.add_geometry("Entity " + str(entity_idx), entity, material)

        # Draw labels
        for object_idx, object in enumerate(objects):
            label_pos = [object['center'][0], object['center'][1], object['center'][2] + 0.15]

            label_text = object['idx']
            # if object_idx == cereal_box_object_idx:
            #     label_text += ' (Cereal Box)'

            label = widget3d.add_3d_label(label_pos, label_text)
            label.color = gui.Color(object['color'][0], object['color'][1],object['color'][2])
            
        bbox = widget3d.scene.bounding_box
        widget3d.setup_camera(60.0, bbox, bbox.get_center())
        w.add_child(widget3d)

        app.run()

        o3d.visualization.draw_geometries(entities, 
                                        zoom=view['trajectory'][0]['zoom'],
                                        front=view['trajectory'][0]['front'],
                                        lookat=view['trajectory'][0]['lookat'],
                                        up=view['trajectory'][0]['up'],
                                        point_show_normal=False)
    
    # ------------------------------------------
    # Text to Speech
    # ------------------------------------------

    characteristic_list = ['length', 'width', 'height', 'real_color']
    adjective_list = ['long', 'wide', 'tall', ' ']

    info_text = Fore.GREEN + '\nThis room has ' + Fore.RED + str(len(objects)) + Style.RESET_ALL + Fore.GREEN + ' objects on the table.\n' + Style.RESET_ALL
    info_speech ='This room has ' + str(len(objects)) + ' objects on the table. '
    
    for object_idx, object in enumerate(objects):
        if object_idx == 0:
            characteristic_number = random.randint(0, len(characteristic_list)-1)

            if characteristic_number == 3:
                text_to_add = Fore.GREEN + 'Object number ' + Fore.RED + '1 ' + Style.RESET_ALL + Fore.GREEN + 'is ' + Style.RESET_ALL + Fore.RED + str(object.get(characteristic_list[characteristic_number])) + Style.RESET_ALL
                speech_to_add = 'Object number 1 is ' + str(object.get(characteristic_list[characteristic_number]))
            else:
                text_to_add = Fore.GREEN + 'Object number ' + Fore.RED + '1 ' + Style.RESET_ALL + Fore.GREEN + 'is ' + Style.RESET_ALL + Fore.RED + str(round(object.get(characteristic_list[characteristic_number])*100)) + 'cm ' + adjective_list[characteristic_number] + Style.RESET_ALL
                speech_to_add = 'Object number 1 is ' + str(round(object.get(characteristic_list[characteristic_number])*100)) + ' centimeters ' + adjective_list[characteristic_number]
            info_text = info_text + text_to_add
            info_speech = info_speech + speech_to_add

        elif object_idx == len(objects)-1 and object_idx != 0:
            characteristic_number = random.randint(0, len(characteristic_list)-1)

            if characteristic_number == 3:
                text_to_add = Fore.GREEN + ' and object number ' + Fore.RED + str(len(objects)) + Style.RESET_ALL + Fore.GREEN + ' is ' + Style.RESET_ALL + Fore.RED + str(object.get(characteristic_list[characteristic_number])) + Style.RESET_ALL + Fore.GREEN + '.\n' + Style.RESET_ALL
                speech_to_add = ' and object number ' + str(len(objects)) + ' is ' + str(object.get(characteristic_list[characteristic_number])) + '.'
            else:
                text_to_add = Fore.GREEN + ' and object number ' + Fore.RED + str(len(objects)) + Style.RESET_ALL + Fore.GREEN + ' is ' + Style.RESET_ALL + Fore.RED + str(round(object.get(characteristic_list[characteristic_number])*100)) + 'cm ' + adjective_list[characteristic_number] + Style.RESET_ALL + Fore.GREEN + '.\n' + Style.RESET_ALL
                speech_to_add = ' and object number ' + str(len(objects)) + ' is ' + str(round(object.get(characteristic_list[characteristic_number])*100)) + ' centimeters ' + adjective_list[characteristic_number] + '.'
            info_text = info_text + text_to_add
            info_speech = info_speech + speech_to_add

        else:
            characteristic_number = random.randint(0, len(characteristic_list)-1)
            
            if characteristic_number == 3:
                text_to_add = Fore.GREEN + ', object number ' + Fore.RED + str(object_idx+1) + Style.RESET_ALL + Fore.GREEN + ' is ' + Style.RESET_ALL + Fore.RED + str(object.get(characteristic_list[characteristic_number])) + Style.RESET_ALL
                speech_to_add = ', object number ' + str(object_idx+1) + ' is ' + str(object.get(characteristic_list[characteristic_number]))
            else:
                text_to_add = Fore.GREEN + ', object number ' + Fore.RED + str(object_idx+1) + Style.RESET_ALL + Fore.GREEN + ' is ' + Style.RESET_ALL + Fore.RED + str(round(object.get(characteristic_list[characteristic_number])*100)) + 'cm ' + adjective_list[characteristic_number] + Style.RESET_ALL
                speech_to_add = ', object number ' + str(object_idx+1) + ' is ' + str(round(object.get(characteristic_list[characteristic_number])*100)) + ' centimeters ' + adjective_list[characteristic_number]
            info_text = info_text + text_to_add
            info_speech = info_speech + speech_to_add

    print(str(info_text))

    if args['text_to_speech'] == True:
        speak(str(info_speech))

    # Trying to classify an image
    image = Image.open('5.png')

    object_class = Classifier(image)
    print(object_class)


if __name__ == "__main__":
    main()