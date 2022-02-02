import open3d as o3d
import numpy as np
import shutil

def help_line(string):
  width, _ = shutil.get_terminal_size()
  print ("\n")
  print (string.center(width, '='))

def _get_pcd(sample_name, pcd=False, points=False, colors=False):
  pcd_dict = {}
  pointcloud = o3d.io.read_point_cloud(sample_name)
  if pcd:
    pcd_dict["pcd"] = pointcloud
  if points:
    pcd_dict["points"] = np.asarray(pointcloud.points)
  if colors:
    pcd_dict["colors"] = np.asarray(pointcloud.colors)

  return pcd_dict


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

# Helper function which displays deleted (outlier) points and valid (inlier) points
def display_inlier_outlier(cloud, ind):
  inlier_cloud = cloud.select_by_index(ind)
  outlier_cloud = cloud.select_by_index(ind, invert=True)

  print ("\n=== Show outlier (red) and inlier (gray): ===")
  outlier_cloud.paint_uniform_color([1, 0, 0])
  inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
  o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    # select_by_index() - takes a binary mask to output only the selected points.

# Statisticak outlier removal function
def Stat_removal(pointcloud, neighbors = 20, ratio = 0.5):
  #help_line(" Statistical removal ")
  cl, ind = pointcloud.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=ratio)
  #print ("Points before filtering: ", pointcloud)
  #print ("Points after filtering: ", cl, "\n")
  return cl, ind

# Find x, y, z coordinates for corner points. ============================
def Find_corner(pointcloud):
  pcd1_coor = np.asarray(pointcloud.points)
  x_max = np.amax(pcd1_coor, where=[True, False, False], initial=-1, axis=0)
  x_max = x_max[0]
  y_max = np.amax(pcd1_coor, where=[False, True, False], initial=-1, axis=0)
  y_max = y_max[1]
  z_min = np.amin(pcd1_coor, where=[False, False, True], initial=610, axis=0)
  z_min = z_min[2]
  return x_max, y_max, z_min

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
  #print ("\nx Diff =", x_diff, "\ny Diff =", y_diff, "\nz Diff =", z_diff)
  #o3d.visualization.draw_geometries([pcd["pcd1"], pcd_move, mesh_frame])
  return pcd_move.points
