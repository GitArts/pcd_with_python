import open3d as o3d
import numpy as np
from helper import get_pcd, Stat_removal, display_inlier_outlier, Find_corner
from Object_func import get_colorpoints, get_BoBox, From_BB_get_points

# |=== This function tries to get potential object ===|
def get_one_object(pcd, pcd_name, projection_points):
  pcd_out = o3d.geometry.PointCloud() # New empty pcd
  points = np.asarray(pcd.points)
  # find the max 'y' value of the whole pointcloud =====================
  Y_max = np.amax(points, where=[True, False, False], initial = -np.inf)
  # find the point with the max 'y' value in the pointcloud ============
  Y_max_row = points[np.where((points == Y_max).any(axis=1))]
  Y_max_row = Y_max_row[0] # -> Y_max_row = [y, x, z]

  # for loop creates an imaginary square which increases in size until possible object is located
  for r in range(2, 300):
    obj_y = Y_max_row[0] - 2*r
    obj_x_down = Y_max_row[1] - r
    obj_x_up = Y_max_row[1] + r
    New_points = points[(points[:, 0] > obj_y)&(points[:, 1] > obj_x_down)&(points[:, 1] < obj_x_up)]

    if (np.size(New_points) == np.size(points[(points[:, 0] > obj_y+1)&
                                        (points[:, 1] > obj_x_down+1)&
                                        (points[:, 1] < obj_x_up-1)]) and np.size(New_points) != 0):
      break
    elif r > 30:
      if projection_points is None:
        projection_points = get_projection(pcd_name)
      else:
        pass
      # |=== The second algorithm activation ===|
      OB_BB = _get_one_object(New_points, Y_max_row, projection_points)
      Size_could_be_0 = From_BB_get_points(OB_BB, points)
      New_point = Size_could_be_0 if np.size(Size_could_be_0)!=0 else New_points
      break
    else:
      continue

  pcd_out.points = o3d.utility.Vector3dVector(New_points)
  pcd_out.paint_uniform_color([160/255, 82/255, 45/255])
  # return one possible object and projection points================
  return pcd_out, projection_points

# =================== second algorithm =============================
def _get_one_object(points, Y_max_samples, projection_points):
  # Find projection points in 'points' region ======================
  Max_min_points = Find_corner(points, 111100)
  # tmp projection points to find a Y_max_projection ===============
  projection_points = projection_points[(projection_points[:, 0] < Max_min_points["y_max"][0]+5)&
                                        (projection_points[:, 0] > Max_min_points["y_min"][0]-5)&
                                        (projection_points[:, 1] < Max_min_points["x_max"][1]+5)&
                                        (projection_points[:, 1] > Max_min_points["x_min"][1]-5)]
  # while loop tries to find max 'y' value of the object projection using test_OB function =====
  while True:
    # If no projection then return bounding box of the points. It means it's not an object.
    if np.size(projection_points) == 0:
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(points)
      # Get bounding box object projection ================================
      BB = get_BoBox(pcd)
      return BB
    Y_max_projection = Find_corner(projection_points, 100000)['y_max']
    # Projection region for analysis. Filter noises =================================
    projection_points = projection_points[(projection_points[:, 2] < Y_max_projection[2]+5)&
                                                (projection_points[:, 2] > Y_max_projection[2]-5)]
    # Def a point into object projection for analyses ================
    ob_dot = Y_max_projection - np.asarray([5, 0, 0]) # point from where object will be analyzed
    # set boundres for analyses ======================================
    x_up, x_down = ob_dot[1]+1, ob_dot[1]-1;
    p_1 = projection_points[(projection_points[:,1] > x_down)&
                                (projection_points[:, 1] < x_up)&
                                (projection_points[:, 0] < ob_dot[0])]
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
    # Test are there points on 'y' and 'x' axis at 'ob_dot' location =
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
  p_2, p_3, p_4, p_5 = (p_1_max_min["y_min"] + [-1, 0, -40], 
                        p_1_max_min["y_max"] + [1, 0, 1], 
                        p_1_max_min["x_min"] + [0, -1, 0], 
                        p_1_max_min["x_max"] + [0, 1, 0])
  all_points = np.vstack((p_1, p_2, p_3, p_4, p_5))
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(all_points)
  # Get bounding box object projection ================================
  BB = get_BoBox(pcd)
  return BB
#|=== Next function is for while loops in the function _get_one_object(). It gets 4 points of the object projection ===|
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

# |=== Test function for object projection test ===|
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

# |=== get objects projections on the base ===|
def get_projection(pcd_name):
  # Get projection of the pointcloud ===================================
  pcd = get_pcd(pcd_name, pcd=True)["pcd"]
  projection = get_colorpoints(pcd, reverse=True)
  _, ind = Stat_removal(projection)
  projection = display_inlier_outlier(projection, ind)
  projection_points = np.asarray(projection.points)  # <-- 2D projection
  return projection_points
