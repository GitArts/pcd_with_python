import open3d as o3d
import numpy as np
import datetime
import shutil
from os import mkdir, path
from My_Functions import _get_pcd, display_inlier_outlier
'''
This file contains 14 functions:
-------------------------------
1. help_line() - function which works similar as a print() function, but for titles;
2. Stat_removal() - filteres out noise points based on neghbors checking;
3. get_colorpoints() - Gets sample points or projection points using color analyse;
4. get_projection() - gets obejct projection. This function uses get_colorpoints function;
5. get_one_object() - gets one potential object for later analysis;
6. _get_one_obejct() - second algorithm wich trys to recognize one object;
7. test_OB() - test if projection slightly looks like object projection or not;
8. In_while_func() - This function is supplement function which is used in _get_one object function;
9. Find_corner() - Finds corners of given points (y_max, y_min, x_max, x_min);
10. Is_object() - analyzes object which cames from get_one_object() function.
			Returns True if an object meets certain criteria and False - if not;
11. get_BoBox() - if Is_object() returned True then this function gets bounding box of the object;
12. From_BB_get_points() - Gets objekt points from known bounding box
13. del_object() - deletes analized object from pcd, which allows to get next potential object;
14. main() - calls other functions;
At the end of this file (~ line 385) please specify which pointcloud (pcd) to load. Use variable 'pcd_name'
'''

## Function for titles
def help_line(string):
  width, _ = shutil.get_terminal_size()
  print ("\n")
  print (string.center(width, '='))

def Stat_removal(pointcloud, neighbors = 20, ratio = 0.5):
  cl, ind = pointcloud.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=ratio)
  return cl, ind

def get_colorpoints(pcd_name, reverse=False):
  # Get avenes sample colors ===================================
  OB_colors = _get_pcd("../samples/sample_1.pcd", colors=True)["colors"]        # Finction _get_pcd() returns dict
  NOTOB_colors4 = _get_pcd("../samples/Not_object_main.pcd", colors=True)["colors"]

  # Test pointcloud ============================================
  pcd_points = _get_pcd(pcd_name, points=True, colors=True)
  pcd_colors = pcd_points["colors"]
  pcd_points = pcd_points["points"]

  # Get a Mean's of all pointcloud's colors ============
  OB_mean = np.mean(OB_colors, axis=0)
  NOTOB_mean4 = np.mean(NOTOB_colors4, axis=0)

  # Get a Norm's to an object mean and not a object sample mean ========
  OB_Norm = np.linalg.norm(pcd_colors - OB_mean, axis=1)
  NOTOB_Norm4 = np.linalg.norm(pcd_colors - NOTOB_mean4, axis=1)

  # Get object points ==================================================
  if reverse:
    pcd_points = pcd_points[(OB_Norm > NOTOB_Norm4) & (OB_Norm > 0.5)]
    pcd_colors = pcd_colors[(OB_Norm > NOTOB_Norm4) & (OB_Norm > 0.5)]
  else:
    pcd_points = pcd_points[(OB_Norm < NOTOB_Norm4) & (OB_Norm < 0.5)]
    pcd_colors = pcd_colors[(OB_Norm < NOTOB_Norm4) & (OB_Norm < 0.5)]
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
  pcd_out = o3d.geometry.PointCloud()
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
      print ("radiuss of the object is", r)
      break
    elif r > 15:
      print ("radiuss is bigger then 15")
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

def In_while_func(points, ob_dot, p_1, Y_max_row, rl = True):
  if rl:
    y_left, y_right = ob_dot[0]+2, ob_dot[0]-2;
  else:
    x_up, x_down = ob_dot[1]+2, ob_dot[1]-2;

  # Find min x untop of the ob_dot && bellow of the ob_dot ======
  if rl:
    p_2 = points[(points[:,0] < y_left)&(points[:, 0] > y_right)&(points[:, 1] > ob_dot[1])]
  else:
    p_2 = points[(points[:,1] < x_up)&(points[:, 1] > x_down)&(points[:, 0] < ob_dot[0])]
  # There can be option when p_2=[], then distance won't work ====
  if np.size(p_2)==0:
    return p_1, True

  distance = np.linalg.norm(p_2 - ob_dot, axis=1)
  distance = np.where(distance==np.amin(distance), -1, distance)
  p_2 = p_2[distance == -1]

  if rl:
    p_3 = points[(points[:,0] < y_left)&(points[:, 0] > y_right)&(points[:, 1] < ob_dot[1])]
  else:
    p_3 = points[(points[:,1] < x_up)&(points[:, 1] > x_down)&(points[:, 0] > ob_dot[0])]
  if np.size(p_3)==0:
    return p_1, True

  distance = np.linalg.norm(p_3 - ob_dot, axis=1)
  distance = np.where(distance==np.amin(distance), -1, distance)
  p_3 = p_3[distance == -1]

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

#@@@@@@@
def Find_corner(points):
  Dict = {}
  y_max = np.amax(points, where=[True, False, False], initial=-np.inf)
  y_max = points[np.where((points == y_max).any(axis=1))][0]
  print ("My print:", y_max)

  y_min = np.amin(points, where=[True, False, False], initial=np.inf)
  y_min = points[np.where((points == y_min).any(axis=1))][0]

  x_max = np.amax(points, where=[False, True, False], initial=-np.inf)
  x_max = points[np.where((points == x_max).any(axis=1))][0]

  x_min = np.amin(points, where=[False, True, False], initial=np.inf, axis=0)
  x_min = points[np.where((points == x_min).any(axis=1))][0]

  Dict["y_max"], Dict["y_min"], Dict["x_max"], Dict["x_min"] = y_max, y_min, x_max, x_min
  return Dict
# =========================================================

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
  # Get only samples by colors =======================
  pcd = get_colorpoints(pcd_name)
  pcd, ind = Stat_removal(pcd, ratio=0.5)
  projection_points = None # Projection_points activates if second algorithm is in use.
  # Def bounding box variables =======================
  BB = {}
  BB_nr = 0
  count = 0

  # While 'True' -> keep looking and analyzing possible objects.
  while True:
    count += 1
    try:
      # If the second algorithm activates then there are values in projection_points variable.
      obj_pcd, projection_points = get_one_object(pcd, pcd_name, projection_points)
    except:
      help_line(" No more objects ")
      break
    #Filtered_object, ind = Stat_removal(obj_pcd, ratio=0.5)
    Is_an_object = Is_object(obj_pcd)
    print ("Is an object -", Is_an_object)
    if Is_an_object:
      BB_nr += 1  ## There are {BB_nr} Bounding boxes. BB["BB0"] will be 'pcd' variable values.
      BB[f"BB{BB_nr}"] = get_BoBox(obj_pcd)
    else:
      pass
    pcd = del_object(pcd, obj_pcd) # after this line execution pcd contains the same points, 
						# but without an object which has been analyzed.
    help_line(f" End of the part {count} ")
  # ============ visualization ============
  pcd = o3d.io.read_point_cloud(f"{head}/{tail}")	# <---
  BB["BB0"] = pcd

  # ============ Get np array from bounding boxes ======

  dict_lengh = len(BB.keys())
  print (f"There are {dict_lengh - 1} objects")
  # === Display script Runtime ====
  print ("Skript runung time:", datetime.datetime.now() - begin_time)

  # Visualization of PCD and bounding boxes ====
  o3d.visualization.draw_geometries([BB[f"{i}"] for i in BB.keys()])

# =============================== main =============================== #

if __name__  ==  '__main__':
  pcd_name = "../sample_14.pcd"
  exit(main(pcd_name))
  # Ja p_1 = []
  # cidonijas
