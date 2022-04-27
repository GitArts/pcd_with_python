import open3d as o3d
import numpy as np
import sys; sys.path.append("../../../kopigs")
import heapq
from My_Functions import _get_pcd, show_pcd, Find_corner, get_BoBox, From_BB_get_pcd, scale_pcd
from My_Functions2 import Align_base, get_points

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

def main():
  # 		|=== load pcd ===|
  pcd = _get_pcd("../par_10.pcd", pcd=True)["pcd"]
  # 		|=== pcd rotation ===|
  pcd, Base_z = Align_base(pcd)
  points = np.asarray(pcd.points)
  # 		|=== scale pcd ===|
  pcd_center = pcd.get_center()
  pcd, scale_num = scale_pcd(pcd, -1, pcd_center) # (-1) <-- This value doesn't matter
  # 		|=== remove the base ===|
  points = points[points[:, 2] > Base_z+1]
  """
  1. Atrast max_z pcd;							<-- 1. YEP
  2. "Imaginārā kvadrāta" princips; 					<-- 2. YEP
  3. Ja visos četros virzienos nākamajā solī 'z' vērtība ir mazāka
		turpini atlasīt punktus;				<-- 3. YEP
  4. Rotēt pcd, lai visiem stūriem būtu max_z vienāds.			<-- 4. YEP kind of (y_max[2]=y_min[2]=x_max[2])
  5. noņemt pamatu							<-- 5. YEP
  6. Modificēt get_points(), lai meklē 4 stūrus -- 4 reizes lēnāk.	<-- 6. YEP
  7. dabūt objektu no BB
  8. Ja nē, izmantot funkciju Is_object;
  9. Modificēt funkciju Is_objekt -->
		analīze pēc augstuma
  10. Noņemt objektu no pointcloud-a
  """
  # 	   |=== Def bounding box variables ===|
  BB = {}
  BB_nr = 0
  count = 0
  # 	       |=== Get object points ===|
  while True:
    OB = get_points(points, Base_z)
    OB_BB = get_BoBox(OB)
    OB = From_BB_get_pcd(OB_BB, pcd) # <-- object bounding box is aplayed to big pcd
    #		  |=== Object test ===|
    Is_an_object = Is_object(np.asarray(OB.points))
    #|=== If it's an object then get bounding box ===|
    if Is_an_object:
      BB_nr += 1
      BB[f"BB{BB_nr}"] = get_BoBox(OB)
      # |=== anti-scale for bounding boxes ===|
      BB[f"BB{BB_nr}"], _ = scale_pcd(BB[f"BB{BB_nr}"], scale_num, pcd_center)
    show_pcd(BB["BB1"])
    breakpoint()
    show_pcd([pcd, BB["BB1"]])
    quit()
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
