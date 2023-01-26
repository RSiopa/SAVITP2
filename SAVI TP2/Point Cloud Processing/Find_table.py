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
			"boundingbox_max" : [ 2.7116048336029053, 1.2182252407073975, 3.8905272483825684 ],
			"boundingbox_min" : [ -2.4257750511169434, -1.6397310495376587, -1.3339539766311646 ],
			"field_of_view" : 60.0,
			"front" : [ -0.40177580646913913, -0.79098050787590612, 0.46143909402698702 ],
			"lookat" : [ 0.14291489124298096, -0.21075290441513062, 1.2782866358757019 ],
			"up" : [ 0.1988804210095626, -0.56724133866565107, -0.79917697780145003 ],
			"zoom" : 0.69999999999999996
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

class PlaneDetection():
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud

    def colorizeInliers(self, r,g,b):
        self.inlier_cloud.paint_uniform_color([r,g,b]) # paints the plane in red

    def segment(self, distance_threshold=0.25, ransac_n=3, num_iterations=50):

        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, 
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model

        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)

        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=True)

        return outlier_cloud

    def __str__(self):
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) +  ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0' 
        return text

def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------

    print("Load a ply point cloud, print it, and render it")
    #ply_point_cloud = o3d.data.PLYPointCloud()
    point_cloud = o3d.io.read_point_cloud('Scenes/rgbd-scenes-v2/pc/01.ply')
    print(point_cloud)
    print(np.asarray(point_cloud.points))

    o3d.visualization.draw_geometries([point_cloud],
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])


    # ------------------------------------------
    # Execution
    # ------------------------------------------
    plane_model, inlier_idxs = point_cloud.segment_plane(distance_threshold=0.3, 
                                                    ransac_n=10,
                                                    num_iterations=500)
    [a, b, c, d] = plane_model
    print('Plane equation: ' + str(a) +  ' x + ' + str(b) + ' y + ' + str(c) + ' z + ' + str(d) + ' = 0' )

    inlier_cloud = point_cloud.select_by_index(inlier_idxs)
    inlier_cloud.paint_uniform_color([1.0, 0, 0]) # paints the plane in red
    outlier_cloud = point_cloud.select_by_index(inlier_idxs, invert=True)

    # ------------------------------------------
    # Termination
    # ------------------------------------------

if __name__ == "__main__":
    main()

