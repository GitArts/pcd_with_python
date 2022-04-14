import open3d as o3d
import numpy as np
import sys; sys.path.append("../../../kopigs")
import heapq
from My_Functions import _get_pcd, show_pcd, Find_corner
from My_Functions2 import Align_base


def get_colorpoints(pcd, reverse=False):
  # Get sample colors for train ================================
  avene_colors = _get_pcd("../OB_def/avene_sample.pcd", colors=True)["colors"]        	# Avenes samples
  cidonija_colors = _get_pcd("../OB_def/cidonija_sample.pcd", colors=True)["colors"] 	# cidonijas samples
  NOTOB_colors4 = _get_pcd("../OB_def/Not_object_main.pcd", colors=True)["colors"] 	# Not a object samles
  NOTOB_colors2 = _get_pcd("../OB_def/NO.pcd", colors=True)["colors"]			# Not a object samples 2

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
  return pcd_points, pcd_colors

def get_points(points, Z_VALUE):
  max_z = np.amax(points, where=[False, False, True], initial = -np.inf)
  max_z_point = points[np.where((points == max_z).any(axis=1))][0]
  # |== +Y ==|
  for r in range(1, 100):
    pluss_y = max_z_point[0]+r
    P1 = points[(points[:, 0] < pluss_y)&
			(points[:, 0] > max_z_point[0]-1)&
			(points[:, 1] < max_z_point[1]+1)&
			(points[:, 1] > max_z_point[1]-1)]
    if (P1[-2, 2] < P1[-1, 2]+0.05)and(P1[-1, 2] > Z_VALUE-2):  # Z_VALUE = base_Z
      show_pcd(P1, vector=True)
      print ("P1[-1, 2] =", P1[-1, 2], "\nZ_VALUE-0.05 =", Z_VALUE-2)
      break
  # |== -Y ==|
  for r in range(1, 100):
    minus_y = max_z_point[0]-r
    P2 = points[(points[:, 0] < max_z_point[0]+1)&
			(points[:, 0] > minus_y)&
			(points[:, 1] < max_z_point[1]+1)&
			(points[:, 1] > max_z_point[1]-1)]
    if (P2[-2, 2] < P2[-1, 2]+0.05)and(P2[-1, 2] < Z_VALUE-2):  # Z_VALUE = base_Z
      show_pcd(P2, vector=True)
      print ("P2[-1, 2] =", P2[-1, 2], "\nZ_VALUE-0.05 =", Z_VALUE-2)
      break
  # |== +X ==|
  # |== -X ==|
  New_points = np.vstack((P1, P2))
  return New_points

def Is_object(points):

  x_max = np.amax(points, where=[False, True, False], initial = -np.inf)
  x_min = np.amin(points, where=[False, True, False], initial = np.inf)
  y_max = np.amax(points, where=[True, False, False], initial = -np.inf)
  y_min = np.amin(points, where=[True, False, False], initial = np.inf)
  z_max = np.amax(points, where=[False, False, True], initial = -np.inf)
  z_min = np.amin(points, where=[False, False, True], initial = np.inf)
  x_diff, y_diff, z_diff = x_max-x_min, y_max-y_min, z_max-z_min

  # Returns True or False based on given criterias
  if (abs(x_diff / y_diff) <= 2.85 and 
		y_diff < 65 and y_diff > 5 and 
		x_diff < 65 and x_diff > 5):
    return True
  else:
    return False

def get_BoBox(obj_points):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(obj_points)
  BB = pcd.get_axis_aligned_bounding_box()
  BB.color = (1, 0, 0)
  return BB

def main():
  pcd = _get_pcd("../par_10.pcd", pcd=True, points=True, colors=True)
  # Get pcd, pcd points and pcd colors ===============
  points = pcd["points"]
  colors = pcd["colors"]
  pcd = pcd["pcd"]  # <-- 4. rotate this pcd.
  # pcd rotation =====================================
  pcd = Align_base(pcd)
  # Def bounding box variables =======================
  BB = {}
  BB_nr = 0
  count = 0
  """
  1. Atrast max_z pcd;							<-- 1. YEP
  2. "Imaginārā kvadrāta" princips; 					<-- 2. YEP
  3. Ja visos četros virzienos nākamajā solī 'z' vērtība ir mazāka
		turpini atlasīt punktus;				<-- 3. YEP
  4. Rotēt pcd, lai visiem stūriem būtu max_z vienāds.			<-- 4. YEP kind of (y_max[2]=y_min[2]=x_max[2])
  5. no augšas analizēt uz leju līdz brītim, kad max_z saglabājas
  6. Modificēt get_points(), lai meklē 4 stūrus -- 4 reizes lēnāk.
  6. Ja nē izmantot funkciju Is_object;
  7. Modificēt funkciju Is_objekt -->
		analīze pēc augstuma
  """
  # Get object points ===============================
  Z_VALUE = Find_corner(np.asarray(pcd.points), y_max=True)["y_max"][2]
  OB = get_points(points, Z_VALUE)
  # Is it an object? ================================
  Is = Is_object(OB)
  # If it is an object, get bounding box ============
  if Is:
    print ("IS OBJECT!")
    BB_nr += 1
    BB[f"BB{BB_nr}"] = get_BoBox(OB)
  # Visualize pcd and bounding boxes ================
  BB["BB0"] = pcd
  o3d.visualization.draw_geometries([BB[f"{i}"] for i in BB.keys()])

if __name__ == "__main__":
  main()

#show_pcd(points, vector=1)
