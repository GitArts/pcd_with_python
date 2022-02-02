import open3d as o3d
import numpy as np
import datetime
import pylab as pl
import shutil
from os import mkdir, path

'''
This file contains 9 functions:
-------------------------------
1. help_line() - function which works similar as a print() function, but for titles;
2. Clustering() - function deletes the base of the samples in pcd;
3. Stat_removal() - filteres out noise points based on neghbors checking;
4. pre_processing() - perferming clustering and filtering of given pcd.
			No need to use Clustering() and Stat_removal() functions one by one;
5. get_one_object() - gets one potential object for later analysis;
6. Is_object() - analyzes object wich cames from get_one_object() function.
			Returns True if an object meets certain criteria and False - if not;
7. get_BoBox() - if Is_object() returned True then this function gets bounding box of the object;
8. del_object() - deletes analized object from pcd, which allows to get next potential object;
9. main() - calls other functions;
At the end of this file (~ line 186) please specify which pointcloud (pcd) to load.
'''

## Function for titles
def help_line(string):
  width, _ = shutil.get_terminal_size()
  print ("\n")
  print (string.center(width, '='))

def Clustering(pointcloud):
  with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pointcloud.cluster_dbscan(eps=1, min_points=40, print_progress=True)) # eps=3, min_points=100
                # eps - Distance to a neighbors in a claster. In general smaller value - less clasters
                # min_points - the minimum number of points required to form a cluster.

  max_label = labels.max()
  print (f"point cloud has {max_label + 1} clusters")
  labels = np.reshape(labels, (labels.size, 1))
  return labels

def Stat_removal(pointcloud, neighbors = 20, ratio = 0.5):
  cl, ind = pointcloud.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=ratio)
  return cl, ind

def pre_processing(head, tail):
  ## load pointclouda ==================================================
  pcd = o3d.io.read_point_cloud(f"{head}/{tail}")
  points = np.asarray(pcd.points)
  colors = np.asarray(pcd.colors)
  #o3d.visualization.draw_geometries([pcd])

  # Clastering. Allows to find the base of the samples ===
  labels = Clustering(pcd)

  # Getting rid of the base ==============================
  Clusterized_points = points[~np.any(labels == 0, axis=1)]
  Clusterized_colors = colors[~np.any(labels == 0, axis=1)]
  pcd.points = o3d.utility.Vector3dVector(Clusterized_points)
  pcd.colors = o3d.utility.Vector3dVector(Clusterized_colors)

  # Filtering out points =================================
  Filtered_pcd, ind = Stat_removal(pcd)

  ## display and save PCD ================================
  #o3d.visualization.draw_geometries([Filtered_pcd])
  try:
    mkdir(f"{head}/no_base_samples")
  except:
    pass
  o3d.io.write_point_cloud(f"{head}/no_base_samples/{tail}", Filtered_pcd)	# <---

def get_one_object(pcd):
  pcd_out = o3d.geometry.PointCloud()
  points = np.asarray(pcd.points)
  Y_max = np.amax(points, where=[True, False, False], initial = -1) # finds the max 'y' value of the whole pcd
  Y_max_row = points[np.where((points == Y_max).any(axis=1))] # finds the point with the max 'y' value in the pcd
  Y_max_row = Y_max_row[0] # Y_max_row = [y, x, z]
  # for loop creates an imaginary square which increases in size until possible object is located
  for r in range(2, 300):
    obj_y = Y_max_row[0] - 2*r
    obj_x_down = Y_max_row[1] - r
    obj_x_up = Y_max_row[1] + r
    New_points = points[(points[:, 0] > obj_y)&(points[:, 1] > obj_x_down)&(points[:, 1] < obj_x_up)]
    if (np.size(New_points) == 
np.size(points[(points[:, 0] > obj_y+1)&(points[:, 1] > obj_x_down+1)&(points[:, 1] < obj_x_up-1)]) and np.size(New_points) != 0):
      print ("radiuss of the object is", r)
      break
    elif r > 290:
      print ("radiuss is bigger then 290")
    else:
      continue

  pcd_out.points = o3d.utility.Vector3dVector(New_points)
  pcd_out.paint_uniform_color([160/255, 82/255, 45/255])

  return pcd_out

def Is_object(pcd):
  points = np.asarray(pcd.points)

  # The object have to be located in coordinates between -1000 and 1000. If not 'initial' veriable value must be changed.
  x_max = np.amax(points, where=[False, True, False], initial = -1000)
  x_min = np.amin(points, where=[False, True, False], initial = 1000)
  y_max = np.amax(points, where=[True, False, False], initial = -1000)
  y_min = np.amin(points, where=[True, False, False], initial = 1000)

  help_line(" Is object function data ")
  print (f" x_max = {x_max};\n x_min = {x_min};\n y_max = {y_max};\n y_min = {y_min};\n")
  x_diff, y_diff = x_max - x_min, y_max - y_min
  print (f"\n x_diff = {x_diff};\n y_diff = {y_diff};\n")
  print ("absolute value:", abs(x_diff/y_diff))

  # Returns True or False based on given criterias
  if abs(x_diff / y_diff) <= 2.85 and y_diff < 65 and y_diff > 5 and x_diff < 65 and x_diff > 5:
    return True
  else:
    return False

def get_BoBox(obj_pcd):
  BB = obj_pcd.get_axis_aligned_bounding_box()
  BB.color = (1, 0, 0)
  return BB

def del_object(pcd, one_object_pcd):
  pcd_points = np.asarray(pcd.points)
  one_object_points = np.asarray(one_object_pcd.points)
  help_line(" del_object info ")
  print ("Shape of pcd_points:", np.shape(pcd_points))
  print ("Shape of one_object_points:", np.shape(one_object_points))
  Next_pcd_values = np.where(np.isin(pcd_points[:, 0], one_object_points[:, 0]), 1, 0)
  Next_pcd_values = np.reshape(Next_pcd_values, (Next_pcd_values.size, 1))

  Next_pcd = pcd_points[np.any(Next_pcd_values == 0, axis=1)]
  print ("Shape of the Next pcd:", np.shape(Next_pcd))
  pcd.points = o3d.utility.Vector3dVector(Next_pcd)
  return pcd

def main(pcd_name):
  begin_time = datetime.datetime.now() # Start time of the main function.
  head, tail = path.split(pcd_name)
  if not path.exists(f"{head}/no_base_samples/{tail}"):
    pre_processing(head, tail) # Comment this line if this line once executed.
  else:
    pass
  pcd = o3d.io.read_point_cloud(f"{head}/no_base_samples/{tail}")
  BB = {}
  BB_nr = 0
  count = 0

  # While true keep looking and analyzing possible objects.
  while True:
    count += 1
    try:
      obj_pcd = get_one_object(pcd)
    except:
      help_line(" No more objects ")
      break
    Filtered_object, ind = Stat_removal(obj_pcd, ratio=0.5)
    Is_an_object = Is_object(Filtered_object)
    print ("Is an object -", Is_an_object)
    if Is_an_object:
      BB_nr += 1  ## There are {BB_nr} Bounding boxes. BB["BB0"] will be 'pcd' variable values.
      BB[f"BB{BB_nr}"] = get_BoBox(Filtered_object)
    else:
      pass

    pcd = del_object(pcd, obj_pcd) # after this line execution pcd contains the same points, 
						# but without an object which has been analyzed.
    help_line(f" End of the part {count} ")
  # ============ visualization ============
  pcd = o3d.io.read_point_cloud(f"{head}/{tail}")	# <---
  BB["BB0"] = pcd
  dict_lengh = len(BB.keys())
  print (f"There are {dict_lengh - 1} objects")
  # === Display script Runtime ====
  print ("Skript runung time:", datetime.datetime.now() - begin_time)

  # Visualization of PCD and bounding boxes ====
  o3d.visualization.draw_geometries([BB[f"{i}"] for i in BB.keys()])


# =============================== main =============================== #

if __name__  ==  '__main__':
  pcd_name = "data_avenes3d/par_14.pcd"
  main(pcd_name)

