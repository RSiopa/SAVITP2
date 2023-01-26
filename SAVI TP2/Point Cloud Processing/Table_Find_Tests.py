#!/usr/bin/env python3
import math
from copy import deepcopy
from pydoc import locate
from more_itertools import locate
from matplotlib import cm
import numpy as np
import open3d as o3d

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


class PlaneDetection:
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud

    def colorizeInliers(self, r, g, b):
        self.inlier_cloud.paint_uniform_color([r, g, b])

    def segment(self, distance_threshold=0.00000001, ransac_n=3, num_iterations=100):

        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model

        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)

        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=True)

        return outlier_cloud

    def __str__(self):
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) + ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0'
        return text


def main():
    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    point_cloud_original = o3d.io.read_point_cloud('Scenes/rgbd-scenes-v2/pc/01.ply')
    print(point_cloud_original.colors)

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    point_cloud = deepcopy(point_cloud_original)

    # Estimate normals
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    point_cloud.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))

    angle_tolerance = 10
    vx, vy, vz = 0, 0, 1
    norm_b = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    vertical_idxs = []
    for idx, normal in enumerate(point_cloud.normals):

        nx, ny, nz = normal
        ab = nx*vx + ny*vy + nz*vz
        norm_a = math.sqrt(nx**2 + ny**2 + nz**2)
        angle = math.acos(ab/(norm_a * norm_b)) * 180/math.pi

        if angle < angle_tolerance:
            vertical_idxs.append(idx)

    vertical_cloud = point_cloud.select_by_index(vertical_idxs)
    non_vertical_cloud = point_cloud.select_by_index(vertical_idxs, invert=True)

    vertical_cloud.paint_uniform_color([0.5, 0, 1])

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    # Create a list of entities to draw
    entities = [vertical_cloud, non_vertical_cloud]

    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=3.0, origin=np.array([0., 0., 0.]))
    entities.append(frame)

    o3d.visualization.draw_geometries(entities, zoom=view['trajectory'][0]['zoom'],
                                      front=view['trajectory'][0]['front'], lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'], point_show_normal=False)


if __name__ == "__main__":
    main()
