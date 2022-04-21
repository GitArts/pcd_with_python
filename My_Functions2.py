import open3d as o3d
import numpy as np
from My_Functions import Find_corner, show_pcd
# mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=z_max, size=20)


# |== align base on z axis ==|
def Align_base(pcd):
  # Rotate around Y axis ============================================
  MM = Find_corner(np.asarray(pcd.points), '001100')
  diff_z_x = MM["x_max"][2] - MM["x_min"][2]    # <-- get z value diff platumā
  diff_x = MM["x_max"][1] - MM["x_min"][1]		# <-- get x value diff platumā

  rot_y = np.arcsin(diff_z_x/diff_x) # <-- rad
  R = pcd.get_rotation_matrix_from_xyz((rot_y, 0, 0))
  pcd.rotate(R, center = MM["x_max"])

  # Rotate around X axis ============================================
  MM = Find_corner(np.asarray(pcd.points), 110000)	# <-- New max_min points after first rotation
  diff_z_y = MM["y_max"][2] - MM["y_min"][2]
  diff_y = MM["y_max"][0] - MM["y_min"][0]
  rot_x = np.arcsin(diff_z_y/diff_y) # <-- rad
  R = pcd.get_rotation_matrix_from_xyz((0, rot_x, 0))
  pcd.rotate(R, center = MM["y_max"])

  MM = Find_corner(np.asarray(pcd.points), 111100)
  y_max, y_min, x_max, x_min = MM["y_max"], MM["y_min"], MM["x_max"], MM["x_min"]
  print ("\n\ny_max =", y_max, "\ny_min =", y_min, "\nx_max =", x_max, "\nx_min =", x_min)
  quit()
  return pcd
