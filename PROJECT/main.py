import open3d as o3d
import numpy as np
import datetime
from os import path

from OD_FUNC import *
from groovedness import *
from Object_func import *
from helper import get_pcd, show_pcd, Find_corner, scale_pcd, Stat_removal
'''
This project contains 24 functions:
-------------------------------
OD_FUNC:
  1. get_one_object() - gets one potential object for later analysis;
  2. _get_one_obejct() - second algorithm which trys to recognize one object by it's projection;
  3. In_while_func() - This function is supplement function which is used in _get_one object function;
  4. test_OB() - test if projection slightly looks like object projection or not;
  5. get_projection() - gets obejct projection. This function uses get_colorpoints function;

Object_func:
  1. get_colorpoints() - Gets sample points or projection points using color analyse;
  2. Is_object() - analyzes object which cames from get_one_object() function.
			Returns True if an object meets certain criteria and False - if not;
  3. get_BoBox() - if Is_object() returned True then this function gets bounding box of the object;
  4. From_BB_get_points() - Gets objekt points from known bounding box
  5. del_object() - deletes analized object from pcd, which allows to get next potential object;
  6. print_BB_info() - print object information Height, Width, Depth

grooviedness:
  1. OB_cut_points() - get object analyses points
  2. merge_points() - merge closest points together
  3. Grad_extremes() - locate object max and min values in local areas
  4. Analyse() - Analyzes object groovedness

helper:
  1. get_pcd() - returns dictionary which contains pointcloud it self, pointcloud points and pointcloud colors, 
				if the corresponding function argument is set to True.
  2. show_pcd() - visualizes pcd ar pcd.points
  3. Find_corner() - Finds corners of points (y_max, y_min, x_max, x_min, z_max, z_min);
  4. Vector() - converts points into pcd
  5. Stat_removal() - filteres out noise points based on neghbors checking;
  6. display_inlier_outlier() - calculates and returns outliers for pointcloud
  7. scale_pcd() - calculete and scale pcd into needed size
  8. Align_base() - calculate and rotate pcd so all base corners have the same z value

This file (main.py):
  1. main() - calls all functions;

At the end of this file (~ line 106) please specify which pointcloud (pcd) to load. Use variable 'pcd_name'
'''

def main(pcd_name):
  begin_time = datetime.datetime.now() # Start time of the main function.
  head, tail = path.split(pcd_name)
  # Get samples by colors ============================
  pcd = get_pcd(pcd_name, pcd=True)["pcd"]
  # Scale pcd for analysis ===========================
  pcd_center = pcd.get_center()
  pcd, scale_num = scale_pcd(pcd, -1, pcd_center) # (-1) <-- This value doesn't matter
  pcd = get_colorpoints(pcd)
  for _ in range(3):
    pcd, ind = Stat_removal(pcd, ratio=0.5)
  projection_points = None # <-- Projection_points activates if second algorithm is in use.
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
      #help_line(" No more objects ")
      break
    #Filtered_object, ind = Stat_removal(obj_pcd, ratio=0.5)
    Is_an_object = Is_object(obj_pcd)
    if Is_an_object:
      BB_nr += 1  ## There are {BB_nr} Bounding boxes. BB["BB0"] will be 'pcd'.
      BB[f"BB{BB_nr}"] = get_BoBox(obj_pcd)
      # |=== Groovedness ===|
      if True:
        groove_main(obj_pcd)
      # anti-scale for bounding boxes ===============
      BB[f"BB{BB_nr}"], _ = scale_pcd(BB[f"BB{BB_nr}"], scale_num, pcd_center)
    else:
      pass
    pcd = del_object(pcd, obj_pcd) # after this line execution pcd contains the same points, 
						# except an object which has been analyzed.
  # |=== BB print() ===|
  print_BB_info(BB)
  # ============ load pointcloud for Viz ============
  pcd = o3d.io.read_point_cloud(f"{head}/{tail}")	# <---
  BB["BB0"] = pcd

  # ============ Get np array from bounding boxes ======
  dict_lengh = len(BB.keys())
  # Information out =====================================
  print (f"\n{dict_lengh - 1} objects were detected")
  # PCD and bounding boxes visuzalization ====
  o3d.visualization.draw_geometries([BB[f"{i}"] for i in BB.keys()])

# ========================= End of the main() ============================= #

if __name__  ==  '__main__':
  # use ./file.pcd for current directory
  pcd_name = "../../data_cidonijas3d/a1.pcd"
  # problēma ar brūnu fonu ========
  #pcd_name = "../../../samples_3D/avenes/sample_19.pcd"
  exit(main(pcd_name))

