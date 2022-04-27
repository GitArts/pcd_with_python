import open3d as o3d
import numpy as np
import shutil

"""
			|== Funkciju saraksts: ==|
1. help_line(string)
2. _get_pcd(sample_name, pcd=, points=, colors=)
3. show_pcd(pcd_or_points, vector=, color=, save=, name=)
4. clustering(pointcloud)
5. Stat_removal(pointcloud, neighbors=, ratio=):
6. display_inlier_outlier(cloud, ind)
7. Find_corner(points)
8. scale_pcd(pcd, n, pcd_center)
9. move_pcd(pcd_move, x_max, y_max, z_min)
10. get_BoBox(obj_pcd)
"""


def help_line(string):
  width, _ = shutil.get_terminal_size()
  print ("\n")
  print (string.center(width, '='))

def _get_pcd(sample_name, pcd=False, points=False, colors=False):
  pcd_dict = {}
  pointcloud = o3d.io.read_point_cloud(sample_name)
  if pcd: 	pcd_dict["pcd"] = pointcloud
  if points: 	pcd_dict["points"] = np.asarray(pointcloud.points)
  if colors: 	pcd_dict["colors"] = np.asarray(pointcloud.colors)

  return pcd_dict

def show_pcd(pcd_or_points, vector=False, color=False, save=False, name = 'show_pcd_save.pcd'):
  if vector:  # Don't use list of points in variable pcd_or_points
    points = pcd_or_points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
  else:
    pcd = pcd_or_points
  if save: o3d.io.write_point_cloud(name, pcd)
  if color: pcd.paint_uniform_color([160/255, 82/255, 45/255])
  if isinstance(pcd_or_points, list):
    o3d.visualization.draw_geometries([elem for elem in pcd_or_points])
  else:
    o3d.visualization.draw_geometries([pcd])

# DBSCAN clustering function
def clustering(pointcloud):
  with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pointcloud.cluster_dbscan(eps=1, min_points=40, print_progress=True)) # eps=3, min_points=100
                # eps - Distance to a neighbors in a claster. In general smaller value - less clasters
                # min_points - the minimum number of points required to form a cluster.

  max_label = labels.max()
  print (f"point cloud has {max_label + 1} clusters")
  labels = np.reshape(labels, (labels.size, 1))
  return labels

# Statisticak outlier removal function
def Stat_removal(pointcloud, neighbors = 20, ratio = 0.5):
  cl, ind = pointcloud.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=ratio)
  return cl, ind

# Helper function which displays deleted (outlier) points and valid (inlier) points
def display_inlier_outlier(cloud, ind):
  inlier_cloud = cloud.select_by_index(ind)
  outlier_cloud = cloud.select_by_index(ind, invert=True)
  outlier_cloud.paint_uniform_color([1, 0, 0])
  inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
  return outlier_cloud
    # select_by_index() - takes a binary mask to output only the selected points.

# Find x, y, z coordinates for corner points. ============================
def Find_corner(points, MM_check):
  MM = list(str(MM_check)) # <-- which Min_Max
  Dict = {}
  if MM[0] == '1':
    Y_max = np.amax(points, where=[True, False, False], initial=-np.inf)
    Y_max = points[np.where((points == Y_max).any(axis=1))][0]
    Dict["y_max"] = Y_max
  if MM[1] == '1':
    Y_min = np.amin(points, where=[True, False, False], initial=np.inf)
    Y_min = points[np.where((points == Y_min).any(axis=1))][0]
    Dict["y_min"] = Y_min
  if MM[2] == '1':
    X_max = np.amax(points, where=[False, True, False], initial=-np.inf)
    X_max = points[np.where((points == X_max).any(axis=1))][0]
    Dict["x_max"] = X_max
  if MM[3] == '1':
    X_min = np.amin(points, where=[False, True, False], initial=np.inf)
    X_min = points[np.where((points == X_min).any(axis=1))][0]
    Dict["x_min"] = X_min
  if MM[4] == '1':
    Z_max = np.amax(points, where=[False, False, True], initial=-np.inf)
    Z_max = points[np.where((points == Z_max).any(axis=1))][0]
    Dict["z_max"] = Z_max
  if MM[5] == '1':
    Z_min = np.amin(points, where=[False, False, True], initial=np.inf)
    Z_min = points[np.where((points == Z_min).any(axis=1))][0]
    Dict["z_min"] = Z_min
  return Dict

def scale_pcd(pcd, n, pcd_center):
  try:
    points = np.asarray(pcd.points)
    MM = Find_corner(points)
    y_max, y_min = MM["y_max"][0], MM["y_min"][0]
    y_diff = y_max-y_min
    n = 360/y_diff
  except:
    pass
  pcd.scale(n, center = pcd_center)
  scaled_by = 1/n
  return pcd, scaled_by

# move pointcloud based on original points
def move_pcd(pcd_move, x_max, y_max, z_min):
  pcd2_coor = np.asarray(pcd_move.points)

  x_max2 = np.amax(pcd2_coor, where=[True, False, False], initial=-1, axis=0)
  x_max2 = x_max2[0]
  y_max2 = np.amax(pcd2_coor, where=[False, True, False], initial=-1, axis=0)
  y_max2 = y_max2[1]
  z_min2 = np.amin(pcd2_coor, where=[False, False, True], initial=610, axis=0)
  z_min2 = z_min2[2]

  x_diff = x_max - x_max2
  y_diff = y_max - y_max2
  z_diff = z_min - z_min2
  Diff = np.array([x_diff, y_diff, z_diff])
  pcd_move.points = o3d.utility.Vector3dVector(np.add(pcd2_coor, Diff))
  return pcd_move

def get_BoBox(obj_pcd):
  BB = obj_pcd.get_axis_aligned_bounding_box()
  BB.color = (0, 1, 0)
  return BB

def From_BB_get_pcd(BB, pcd):
  ob_ind = BB.get_point_indices_within_bounding_box(pcd.points)
  OB = pcd.select_by_index(ob_ind)
  return OB
