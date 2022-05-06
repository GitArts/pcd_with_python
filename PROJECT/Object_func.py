import open3d as o3d
import numpy as np
from helper import get_pcd, Find_corner

# |=== Get potencional object in pcd for analyze ===|
def get_colorpoints(pcd, reverse=False):
  # Get sample colors for train ================================
  avene_colors = get_pcd("../../OB_def/avene_sample.pcd", colors=1)["colors"]        # Avenes samples
  cidonija_colors = get_pcd("../../OB_def/cidonija_sample.pcd", colors=1)["colors"] 	# cidonijas samples
  NOTOB_colors4 = get_pcd("../../OB_def/Not_object_main.pcd", colors=1)["colors"] 	# Not a object samles
  NOTOB_colors2 = get_pcd("../../OB_def/NO.pcd", colors=1)["colors"] 		# Not a object samples 2

  # Get pointcloud samples =====================================
  pcd_points = np.asarray(pcd.points)
  pcd_colors = np.asarray(pcd.colors)

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

# |=== Determines if pcd value is an object ===|
def Is_object(pcd):
  points = np.asarray(pcd.points)
  MM = Find_corner(points, 111100) # <-- Min Max dictionary
  x_diff, y_diff = MM["x_max"][1]-MM["x_min"][1], MM["y_max"][0]-MM["y_min"][0]

  # Returns True or False based on given criterias
  if y_diff == 0 or x_diff == 0: return False;
  if abs(x_diff / y_diff) <= 2.85 and y_diff < 65 and y_diff > 5 and x_diff < 65 and x_diff > 5:
    return True
  else:
    return False
# |=== Get bounding box from pcd ===|
def get_BoBox(obj_pcd):
  BB = obj_pcd.get_axis_aligned_bounding_box()
  BB.color = (1, 0, 0)
  return BB

# |=== Get pcd.points from baunding box ===|
def From_BB_get_points(BB, points):
  proj = o3d.geometry.PointCloud()
  proj.points = o3d.utility.Vector3dVector(points)
  ob_ind = BB.get_point_indices_within_bounding_box(proj.points)
  obj = proj.select_by_index(ob_ind)
  points = np.asarray(obj.points)
  return points

# |=== Delete some points (one_object pcd) from big pcd ===|
def del_object(pcd, one_object_pcd):
  pcd_points = np.asarray(pcd.points)
  one_object_points = np.asarray(one_object_pcd.points)
  Next_pcd_values = np.where(np.isin(pcd_points[:, 0], one_object_points[:, 0]), 1, 0)
  Next_pcd_values = np.reshape(Next_pcd_values, (Next_pcd_values.size, 1))
  Next_pcd = pcd_points[np.any(Next_pcd_values == 0, axis=1)]
  pcd.points = o3d.utility.Vector3dVector(Next_pcd)
  return pcd

# |=== print object information Height, Width, Depth ===|
def print_BB_info(BB_dict):
  for BB_nr in BB_dict.keys():
    H = BB_dict[f"{BB_nr}"].get_max_bound()[2] - BB_dict[f"{BB_nr}"].get_min_bound()[2]
    W = BB_dict[f"{BB_nr}"].get_max_bound()[0] - BB_dict[f"{BB_nr}"].get_min_bound()[0]
    D = BB_dict[f"{BB_nr}"].get_max_bound()[1] - BB_dict[f"{BB_nr}"].get_min_bound()[1]

    print ("     |===", BB_nr, "===|")
    print ("OB_Height:", H, "points")   # z
    print ("OB_Width:", W, "points")    # y
    print ("OB_Depth:", D, "points")    # x
