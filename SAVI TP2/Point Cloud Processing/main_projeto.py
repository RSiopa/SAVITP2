#!/usr/bin/env python3

import copy
import csv
import math
import pickle
from copy import deepcopy
from random import randint
from turtle import color

import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from point_cloud_processing_projeto import PointCloudProcessing
from matplotlib import cm
from more_itertools import locate

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
    

class PlaneDetection:
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud

    def colorizeInliers(self, r, g, b):
        self.inlier_cloud.paint_uniform_color([r, g, b])

    def segment(self, distance_threshold=0.05, ransac_n=3, num_iterations=100):

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
    p = PointCloudProcessing()
    p.loadPointCloud('Scenes/rgbd-scenes-v2/pc/13.ply')

    # ------------------------------------------
    # Execution
    # ------------------------------------------

    # Resolution of point cloud
    p.preProcess(voxel_size=0.01)

    # Find table plane
    point_cloud_original = o3d.io.read_point_cloud('Scenes/rgbd-scenes-v2/pc/13.ply')

    # Estimate normals
    point_cloud_original.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=25))
    point_cloud_original.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))

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

    horizontal_cloud = point_cloud_original.select_by_index(horizontal_idxs)
    non_horizontal_cloud = point_cloud_original.select_by_index(horizontal_idxs, invert=True)

    horizontal_cloud.paint_uniform_color([0.5, 0, 1])

    (table_point_cloud, lista) = horizontal_cloud.remove_radius_outlier(150, 0.3)

    table_plane = PlaneDetection(table_point_cloud)

    table_plane_point_cloud = table_plane.segment()

    table_center = table_plane_point_cloud.get_center()

    print(table_center)

    # Positonatig coordinate axis
    p.transform(0,0,0,-table_center[0],-table_center[1],-table_center[2])
    p.transform(-120,0,0,0,0,0)
    p.transform(0,0,-120,0,0,0)
    p.transform(0,-7,0,0,0,0)

    #Cropping point cloud
    p.crop(-0.7, -0.5, -0.05, 0.7, 0.5, 0.5)

    #Find plane
    outliers = p.findPlane()
    
    # Ex5 - Clustering
   # First, it creates a colormap using the Pastel1 color map from the matplotlib library. The number of colors in the colormap 
   # is determined by the number of clusters found in the previous step.

# Then it iterates over the list of object indexes, which is the set of unique cluster indexes found earlier. For each object 
# index, it finds the indexes of all the points that belong to that cluster, and selects those points from the outliers point
#  cloud using the select_by_index() function.

# Then it creates a dictionary to represent the object. The dictionary contains the following information:

# idx: the index of the cluster
# points: the selected points that belong to the cluster
# color: color of the cluster, which is obtained from the colormap
# center: center of the cluster, which is obtained by calling the get_center() function on the points
# Finally, it paints the object with the selected color and adds the object dictionary to the objects list.

# This code is useful for visualizing the different clusters of points and potentially further processing them based on the object properties.
    cluster_idxs = list(outliers.cluster_dbscan(eps=0.03, min_points=60, print_progress=True))
    object_idxs = list(set(cluster_idxs))
    object_idxs.remove(-1)

    number_of_objects = len(object_idxs)
    colormap = cm.Pastel1(list(range(0,number_of_objects)))

    objects = []
    for object_idx in object_idxs:

        object_point_idxs = list(locate(cluster_idxs, lambda x: x == object_idx))
        object_points = outliers.select_by_index(object_point_idxs)
        # Create a dictionary to represent the objects
        d = {}
        d['idx'] = str(object_idx)
        d['points'] = object_points
        d['color'] = colormap[object_idx, 0:3]
        d['points'].paint_uniform_color(d['color']) # paints the plane in red
        d['center'] = d['points'].get_center()
        objects.append(d) # add the dict of this object to the list

    # Ex6 - ICP
    # This code is using the Iterative Closest Point (ICP) algorithm to align the cereal box model (loaded from a file) with each
    #  of the objects found in the point cloud and then determine which object is a cereal box by comparing the fitting errors
    #  (root-mean-square error (RMSE)) obtained from the ICP registration.

# It iterates over the objects list, and for each object, it applies the registration_icp() function provided by the Open3D 
# library. This function aligns the cereal_box_model and the object points using the ICP algorithm. The initial transformation 
# is set to the identity matrix, and the transformation estimation method is set to o3d.pipelines.registration.
# TransformationEstimationPointToPoint() which is point-to-point method.

# The function returns a registration result object that contains the transformation matrix and the inlier_rmse (root-mean-square error)
#  of the registration. The inlier_rmse represents the fitting error between the cereal box model and the object.

# The code stores the inlier_rmse in the object dictionary as "rmse" for later use.

# After the registration for all objects is done, the code finds the object with the lowest rmse, which is considered 
# as the cereal box and assigns it to cereal_box_object_idx variable. Then it prints the index of the cereal box.

# This code is useful for aligning a 3D model with a point cloud and finding the best fit for a specific object in the point cloud.
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

    # ------------------------------------------
    # Visualization
    # ------------------------------------------

    # Create a list of entities to draw
    p.inliers.paint_uniform_color([0,1,1]) # paints the plane in red
    entities = []

    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    entities.append(frame)

    # Draw bbox
    bbox_to_draw = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(p.bbox)
    entities.append(bbox_to_draw)

    # Draw objects
    for object_idx, object in enumerate(objects):
        # if object_idx == 2: #  show only object idx = 2
        entities.append(object['points'])

    # Make a more complex open3D window to show object labels on top of 3d
    app = gui.Application.instance
    app.initialize() # create an open3d app

    w = app.create_window("Open3D - 3D Text", 1920, 1080)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    widget3d.scene.set_background([0,0,0,1])  # set black background
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
        # label.scale = 2
        
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



if __name__ == "__main__":
    main()
