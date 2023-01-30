#!/usr/bin/env python3
import math
from copy import deepcopy
from pydoc import locate
from more_itertools import locate
from matplotlib import cm
import numpy as np
import open3d as o3d
import math as m

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
    

def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])


def main():
    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    point_cloud_original = o3d.io.read_point_cloud('Scenes/rgbd-scenes-v2/pc/01.ply')

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    point_cloud = deepcopy(point_cloud_original)

    # Estimate normals
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=25))
    point_cloud.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))

    angle_tolerance = 0.1
    vx, vy, vz = 1, 0, 0
    norm_b = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    horizontal_idxs = []
    for idx, normal in enumerate(point_cloud.normals):

        nx, ny, nz = normal
        ab = nx*vx + ny*vy + nz*vz
        norm_a = math.sqrt(nx**2 + ny**2 + nz**2)
        angle = math.acos(ab/(norm_a * norm_b)) * 180/math.pi

        if abs(angle - 90) < angle_tolerance:
            horizontal_idxs.append(idx)

    horizontal_cloud = point_cloud.select_by_index(horizontal_idxs)
    non_horizontal_cloud = point_cloud.select_by_index(horizontal_idxs, invert=True)

    horizontal_cloud.paint_uniform_color([0.5, 0, 1])

    (table_point_cloud, lista) = horizontal_cloud.remove_radius_outlier(150, 0.3)

    R_matrix = Rx(0) * Ry(0) * Rz(0)

    table_point_cloud.rotate(R_matrix)

    plane = PlaneDetection(table_point_cloud)

    table_point_cloud = plane.segment()

    print(table_point_cloud.get_center())

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    # Create a list of entities to draw
    entities = [table_point_cloud]

    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    entities.append(frame)

    o3d.visualization.draw_geometries(entities, zoom=view['trajectory'][0]['zoom'],
                                      front=view['trajectory'][0]['front'], lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'], point_show_normal=False)

    # o3d.io.write_point_cloud('factory_isolated.ply', cloud_building, write_ascii=False, compressed=False, print_progress=False)


if __name__ == "__main__":
    main()
