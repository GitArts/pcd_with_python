import open3d as o3d
import numpy as np
import datetime
import shutil
from os import path
import sys
sys.path.append("../../../kopigs")
from My_Functions import _get_pcd, display_inlier_outlier, help_line, Stat_removal, Find_corner

'''
This file contains 16 functions:
-------------------------------
1. _get_pcd() - returns dictionary which contains pointcloud it self, pointcloud points and pointcloud colors, 
				if the corresponding function argument is set to True.
2. display_inlier_outlier() - calculates and returns outliers for pointcloud
3. help_line() - function which works similar as a print() function, but for titles;
4. Stat_removal() - filteres out noise points based on neghbors checking;
5. get_colorpoints() - Gets sample points or projection points using color analyse;
6. get_projection() - gets obejct projection. This function uses get_colorpoints function;
7. get_one_object() - gets one potential object for later analysis;
8. _get_one_obejct() - second algorithm wich trys to recognize one object;
9. test_OB() - test if projection slightly looks like object projection or not;
10. In_while_func() - This function is supplement function which is used in _get_one object function;
11. Find_corner() - Finds corners of given points (y_max, y_min, x_max, x_min);
12. Is_object() - analyzes object which cames from get_one_object() function.
			Returns True if an object meets certain criteria and False - if not;
13. get_BoBox() - if Is_object() returned True then this function gets bounding box of the object;
14. From_BB_get_points() - Gets objekt points from known bounding box
15. del_object() - deletes analized object from pcd, which allows to get next potential object;
16. main() - calls other functions;
At the end of this file (~ line 385) please specify which pointcloud (pcd) to load. Use variable 'pcd_name'
'''
def get_colorpoints(pcd_name, reverse=False):
  # Get sample colors for train ================================
  avene_colors = _get_pcd("../OB_def/avene_sample.pcd", colors=True)["colors"]        # Avenes samples
  cidonija_colors = _get_pcd("../OB_def/cidonija_sample.pcd", colors=True)["colors"] # cidonijas samples
  NOTOB_colors4 = _get_pcd("../OB_def/Not_object_main.pcd", colors=True)["colors"] # Not a object samles
  NOTOB_colors2 = _get_pcd("../OB_def/NO.pcd", colors=True)["colors"] # Not a object samples 2

  # Get pointcloud sampels =====================================
  pcd_points = _get_pcd(pcd_name, points=True, colors=True)
  pcd_colors = pcd_points["colors"]
  pcd_points = pcd_points["points"]

  # Get a Mean's of all pointcloud's colors ====================
  avene_mean = np.mean(avene_colors, axis=0)
  cidonija_mean = np.mean(cidonija_colors, axis=0)
  NOTOB_mean4 = np.mean(NOTOB_colors4, axis=0)
  NOTOB_mean2 = np.mean(NOTOB_colors2, axis=0)

  # Get a Norm's to an object mean and not a object sample mean ========
  avene_Norm = np.linalg.norm(pcd_colors - avene_mean, axis=1)
  cidonija_Norm = np.linalg.norm(pcd_colors - cidonija_mean, axis=1)
  NOTOB_Norm4 = np.linalg.norm(pcd_colors - NOTOB_mean4, axis=1)
  NOTOB_Norm2 = np.linalg.norm(pcd_colors - NOTOB_mean2, axis=1)
  # 'for_test' basicly is a test, this is a avenes or cidonijas samples ====
  for_test = (avene_Norm < NOTOB_Norm4) & (avene_Norm < 0.5)
  # Get object points or get rid of object points ======================
  if reverse:
    pcd_points = pcd_points[(avene_Norm > NOTOB_Norm4) & (avene_Norm > 0.5)]
    pcd_colors = pcd_colors[(avene_Norm > NOTOB_Norm4) & (avene_Norm > 0.5)]
    # Get more points if we are looking for cidonijas===================
    if np.size(for_test) < 50:
      p_1 = pcd_points[(cidonija_Norm > NOTOB_Norm4) & (cidonija_Norm > NOTOB_Norm2) & (cidonija_Norm > 0.5)]
      pcd_points = np.vstack((pcd_points, p_1))
      p_1 = pcd_colors[(cidonija_Norm > NOTOB_Norm4) & (cidonija_Norm > NOTOB_Norm2) & (cidonija_Norm > 0.5)]
      pcd_colors = np.vstack((pcd_colors, p_1))
    else:
      pass
  else:
    pcd_points = pcd_points[(avene_Norm < NOTOB_Norm4) & (avene_Norm < 0.5)]
    pcd_colors = pcd_colors[(avene_Norm < NOTOB_Norm4) & (avene_Norm < 0.5)]
    if np.size(for_test) < 50:
      p_1 = pcd_points[(cidonija_Norm < NOTOB_Norm4) & (cidonija_Norm < NOTOB_Norm2) & (cidonija_Norm < 0.5)]
      pcd_points = np.vstack((pcd_points, p_1))
      p_1 = pcd_colors[(cidonija_Norm < NOTOB_Norm4) & (cidonija_Norm < NOTOB_Norm2) & (cidonija_Norm < 0.5)]
      pcd_colors = np.vstack((pcd_colors, p_1))
    else:
      pass

  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(pcd_points)
  pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
  return pcd

def get_projection(pcd_name):
  # Get projection of the pointcloud ===================================
  projection = get_colorpoints(pcd_name, reverse=True)
  _, ind = Stat_removal(projection)
  projection = display_inlier_outlier(projection, ind)
  projection_points = np.asarray(projection.points)  # <-- 2D projection
  return projection_points

def get_one_object(pcd, pcd_name, projection_points):
  pcd_out = o3d.geometry.PointCloud() # New empty pcd
  points = np.asarray(pcd.points)
  # find the max 'y' value of the whole pointcloud =====================
  Y_max = np.amax(points, where=[True, False, False], initial = -1)
  # find the point with the max 'y' value in the pointcloud ============
  Y_max_row = points[np.where((points == Y_max).any(axis=1))]
  Y_max_row = Y_max_row[0] # -> Y_max_row = [y, x, z]

  # for loop creates an imaginary square which increases in size until possible object is located
  for r in range(2, 300):
    obj_y = Y_max_row[0] - 2*r
    obj_x_down = Y_max_row[1] - r
    obj_x_up = Y_max_row[1] + r
    New_points = points[(points[:, 0] > obj_y)&(points[:, 1] > obj_x_down)&(points[:, 1] < obj_x_up)]

    if (np.size(New_points) == 
np.size(points[(points[:, 0] > obj_y+1)&(points[:, 1] > obj_x_down+1)&(points[:, 1] < obj_x_up-1)]) and np.size(New_points) != 0):
      break
    elif r > 30:
      if projection_points is None:
        projection_points = get_projection(pcd_name)
      else:
        pass
      OB_BB = _get_one_object(New_points, Y_max_row, projection_points)
      New_points = From_BB_get_points(OB_BB, points)
      break
    else:
      continue

  pcd_out.points = o3d.utility.Vector3dVector(New_points)
  pcd_out.paint_uniform_color([160/255, 82/255, 45/255])
  #o3d.visualization.draw_geometries([pcd_out])
  # return one possible object and projection points================
  return pcd_out, projection_points

# =================== second algorithm =============================
def _get_one_object(points, Y_max_samples, projection_points):
  # Find projection points in 'points' region ======================
  Max_min_points = Find_corner(points)
  # tmp projection points to find a Y_max_projection ===============
  projection_points = projection_points[(projection_points[:, 0] < Max_min_points["y_max"][0]+5)&(projection_points[:, 0] > Max_min_points["y_min"][0]-5)&(projection_points[:, 1] < Max_min_points["x_max"][1]+5)&(projection_points[:, 1] > Max_min_points["x_min"][1]-5)]
  # while loop tries to find max 'y' value of the object projection using test_OB function =====
  while True:
    if np.size(projection_points) == 0:
      points = points[(points[:, 0] <= Max_min_points["y_max"][0])&(points[:, 0] >= Max_min_points["y_min"][0])&(points[:, 1] <= Max_min_points["x_max"][1])&(points[:, 1] >= Max_min_points["x_min"][1])]
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(points)
      # Get bounding box object projection ================================
      BB = get_BoBox(pcd)
      return BB

    Y_max_projection = Find_corner(projection_points)['y_max']
    # Projection region for analysis =================================
    projection_points = projection_points[
   	# |=== for Y axis ===|
	(projection_points[:, 0] < Max_min_points["y_max"][0]+5)&(projection_points[:, 0] > Max_min_points["y_min"][0]-5)
   	# |=== for X axis ===|
	&(projection_points[:, 1] < Max_min_points["x_max"][1]+5)&(projection_points[:, 1] > Max_min_points["x_min"][1]-5)
   	# |=== for Z axis ===|
	&(projection_points[:, 2] < Y_max_projection[2]+5)&(projection_points[:, 2] > Y_max_projection[2]-5)]

    # Def a point into object projection for analyses ================
    ob_dot = Y_max_projection - np.asarray([5, 0, 0]) # point from where object will be analyzed
    # set boundres for analyses ======================================
    x_up, x_down = ob_dot[1]+1, ob_dot[1]-1;
    p_1 = projection_points[(projection_points[:,1] > x_down)&(projection_points[:, 1] < x_up)&(projection_points[:, 0] < ob_dot[0])]
    if np.size(p_1) == 0:
      projection_points = np.delete(projection_points, np.where(projection_points[:, 0] == Y_max_projection[0]), axis=0)
      continue
    # Distance from 'ob_dot' to all points ('p_1') ===================
    distance = np.linalg.norm(p_1 - ob_dot, axis=1)
    # Distance to closest point from 'ob_dot' ========================
    distance = np.where(distance==np.amin(distance), -1, distance)
    p_1 = p_1[distance == -1]
    # Find ob_dot in the middle (ob_dot1) ============================
    ob_dot = np.mean(np.vstack((Y_max_projection, p_1)), axis=0)
    # Test are there points on y and x axis at 'ob_dot' location =====
    projection_points, Done = test_OB(projection_points, ob_dot, p_1, Y_max_samples, Y_max_projection)
    # If test_OB() shows that this is not an object projection then delete max 'y' point and try again =====
    if Done:
      break
    else:
      projection_points = np.delete(projection_points, np.where(projection_points[:, 0] == Y_max_projection[0]), axis=0)
  #========== All while loops tries to get boundries of an object projection ===========================
  while True: # move right
    p_1, Done = In_while_func(projection_points, ob_dot, p_1, Y_max_projection)
    if Done:
      break
    else:
      ob_dot = ob_dot - [0.5, 0, 0]
  while True: # move left
    p_1, Done = In_while_func(projection_points, ob_dot, p_1, Y_max_projection)
    if Done:
      break
    else:
      ob_dot = ob_dot + [0.5, 0, 0]

  while True: # move up
    p_1, Done = In_while_func(projection_points, ob_dot, p_1, Y_max_projection, rl=False)
    if Done:
      break
    else:
      ob_dot = ob_dot + [0, 0.5, 0]

  while True: # move down
    p_1, Done = In_while_func(projection_points, ob_dot, p_1, Y_max_projection, rl=False)
    if Done:
      break
    else:
      ob_dot = ob_dot - [0, 0.5, 0]

  #==================================================================================
  # Get all points of the object projection ============================
  # Delay ====
  p_1_max_min = Find_corner(p_1)
  p_2, p_3, p_4, p_5 = p_1_max_min["y_min"] + [-1, 0, -40], p_1_max_min["y_max"] + [1, 0, 1], p_1_max_min["x_min"] + [0, -1, 0], p_1_max_min["x_max"] + [0, 1, 0]
  all_points = np.vstack((p_1, p_2, p_3, p_4, p_5))
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(all_points)
  # Get bounding box object projection ================================
  BB = get_BoBox(pcd)
  return BB

# @@@@@@@@ Test function for object projection test ==============
def test_OB(points, ob_dot, p_1, Y_max_samples, Y_max_projection):
  y_left_dot, y_right_dot = ob_dot[0]+1, ob_dot[0]-1;
  p_2 = points[(points[:,0] < y_left_dot)&(points[:, 0] > y_right_dot)&(points[:, 1] > ob_dot[1]+1)]
  p_3 = points[(points[:,0] < y_left_dot)&(points[:, 0] > y_right_dot)&(points[:, 1] < ob_dot[1]-1)]
  if (np.size(p_1)==0 or np.size(p_2)==0 or np.size(p_3)==0):
    Done = False
  elif (Y_max_projection[1] < Y_max_samples[1]+3)&(Y_max_projection[1] > Y_max_samples[1]-3):
    Done = True
  else:
    Done = False
  return points, Done

#@@@@@ For while loops in the function _get_one_object()
def In_while_func(points, ob_dot, p_1, Y_max_row, rl = True):
  if rl:
    y_left, y_right = ob_dot[0]+2, ob_dot[0]-2;
  else:
    x_up, x_down = ob_dot[1]+2, ob_dot[1]-2;

  # Find min x untop of the ob_dot && bellow of the ob_dot =======
  # ====================== _p_2_ = _p_3_ ==========================
  if rl:
    p_2 = points[(points[:,0] < y_left)&(points[:, 0] > y_right)&(points[:, 1] > ob_dot[1])]
    p_3 = points[(points[:,0] < y_left)&(points[:, 0] > y_right)&(points[:, 1] < ob_dot[1])]
  else:
    p_2 = points[(points[:,1] < x_up)&(points[:, 1] > x_down)&(points[:, 0] < ob_dot[0])]
    p_3 = points[(points[:,1] < x_up)&(points[:, 1] > x_down)&(points[:, 0] > ob_dot[0])]
  # There can be option when p_2=[], then distance won't work ====
  if np.size(p_2)==0 or np.size(p_3)==0:
    return p_1, True

  distance = np.linalg.norm(p_2 - ob_dot, axis=1)
  distance = np.where(distance==np.amin(distance), -1, distance)
  p_2 = p_2[distance == -1]

  distance = np.linalg.norm(p_3 - ob_dot, axis=1)
  distance = np.where(distance==np.amin(distance), -1, distance)
  p_3 = p_3[distance == -1]
  # ====================== ^p_2^ = ^p_3^ ==========================

  # If p_2 is x_max of the object and p_3 is a min of the object then Done ===========
  try:
    if rl:
      Done = (p_1[-2][1] > p_2[0][1])&(p_1[-1][1] < p_3[0][1])
    else:
      Done = (p_1[-2][0] < p_2[0][0])&(p_1[-1][0] > p_3[0][0])
  except:
    pass
  # stack all points ======================================
  if np.size(p_1) == 3:
    p_1 = np.vstack((Y_max_row, p_1, p_2, p_3))
    Done = False
  else:
    p_1 = np.vstack((p_1, p_2, p_3))
  return p_1, Done

def Is_object(pcd):
  points = np.asarray(pcd.points)

  x_max = np.amax(points, where=[False, True, False], initial = -np.inf)
  x_min = np.amin(points, where=[False, True, False], initial = np.inf)
  y_max = np.amax(points, where=[True, False, False], initial = -np.inf)
  y_min = np.amin(points, where=[True, False, False], initial = np.inf)
  x_diff, y_diff = x_max - x_min, y_max - y_min

  # Returns True or False based on given criterias
  if abs(x_diff / y_diff) <= 2.85 and y_diff < 65 and y_diff > 5 and x_diff < 65 and x_diff > 5:
    return True
  else:
    return False

def get_BoBox(obj_pcd):
  BB = obj_pcd.get_axis_aligned_bounding_box()
  BB.color = (1, 0, 0)
  return BB

def From_BB_get_points(BB, points):
  proj = o3d.geometry.PointCloud()
  proj.points = o3d.utility.Vector3dVector(points)
  ob_ind = BB.get_point_indices_within_bounding_box(proj.points)
  obj = proj.select_by_index(ob_ind)
  points = np.asarray(obj.points)
  return points

def del_object(pcd, one_object_pcd):
  pcd_points = np.asarray(pcd.points)
  one_object_points = np.asarray(one_object_pcd.points)
  Next_pcd_values = np.where(np.isin(pcd_points[:, 0], one_object_points[:, 0]), 1, 0)
  Next_pcd_values = np.reshape(Next_pcd_values, (Next_pcd_values.size, 1))

  Next_pcd = pcd_points[np.any(Next_pcd_values == 0, axis=1)]
  pcd.points = o3d.utility.Vector3dVector(Next_pcd)
  return pcd

def main(pcd_name):
  begin_time = datetime.datetime.now() # Start time of the main function.
  head, tail = path.split(pcd_name)
  # Get samples by colors ============================
  pcd = get_colorpoints(pcd_name)
  pcd, ind = Stat_removal(pcd, ratio=0.5)
  projection_points = None # Projection_points activates if second algorithm is in use.
  # Def bounding box variables =======================
  BB = {}
  BB_nr = 0
  count = 0

  # While 'True' -> keep looking and analyzing possible objects.
  while True:
    try:
      # If the second algorithm activates then there are values in projection_points variable.
      obj_pcd, projection_points = get_one_object(pcd, pcd_name, projection_points)
    except:
      help_line(" No more objects ")
      break
    #Filtered_object, ind = Stat_removal(obj_pcd, ratio=0.5)
    Is_an_object = Is_object(obj_pcd)
    if Is_an_object:
      BB_nr += 1  ## There are {BB_nr} Bounding boxes. BB["BB0"] will be 'pcd' variable values.
      BB[f"BB{BB_nr}"] = get_BoBox(obj_pcd)
    else:
      pass
    pcd = del_object(pcd, obj_pcd) # after this line execution pcd contains the same points, 
						# except an object which has been analyzed.
  # ============ visualization ============
  pcd = o3d.io.read_point_cloud(f"{head}/{tail}")	# <---
  BB["BB0"] = pcd

  # ============ Get np array from bounding boxes ======

  dict_lengh = len(BB.keys())
  # Information out =====================================
  print (f"{dict_lengh - 1} objects were detected")
  print ("Skript runung time:", datetime.datetime.now() - begin_time)

  # PCD and bounding boxes visuzalization ====
  o3d.visualization.draw_geometries([BB[f"{i}"] for i in BB.keys()])

# ========================= End of the main() ============================= #

if __name__  ==  '__main__':
  # use ./file.pcd for current directory
  pcd_name = "../../../samples_3D/avenes/sample_19.pcd"
  exit(main(pcd_name))

