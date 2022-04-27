import open3d as o3d
import numpy as np
from My_Functions import Find_corner, show_pcd
# mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=z_max, size=20)


# |== align base on z axis ==|
def Align_base(pcd):
  # Rotate around Y axis ============================================
  MM = Find_corner(np.asarray(pcd.points), '001100')
  diff_z_x = MM["x_max"][2] - MM["x_min"][2]    # <-- get z value diff platumā
  diff_x = MM["x_max"][1] - MM["x_min"][1]	# <-- get x value diff platumā

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
  return pcd, MM["y_max"][2]

def Vector(points):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  return pcd

# if points have to be showed with mesh ===|
def test_where(where, points):
  mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=where, size=20)
  tmp_pcd = Vector(points)
  show_pcd([tmp_pcd, mesh])

def get_points(points, Z_VALUE):
  first_point = Find_corner(points, '000010')["z_max"] # <-- Find min z value. pcd is upside down.
  # 			|== +Y ==|
  for r in range(1, 100):
    pluss_y = first_point[0]+r
    New_points_PlussY = points[(points[:, 0] < pluss_y)&
                        (points[:, 0] > first_point[0]-1)&
                        (points[:, 1] < first_point[1]+1)&
                        (points[:, 1] > first_point[1]-1)]

    Test_points = points[(points[:, 0] < pluss_y+1)&
                        (points[:, 0] > first_point[0]-1)&
                        (points[:, 1] < first_point[1]+1)&
			(points[:, 1] > first_point[1]-1)]

    if (np.size(New_points_PlussY) == np.size(Test_points) and np.size(New_points_PlussY) != 0):
      print ("Pluss Y r =", r)
      show_pcd(New_points_PlussY, vector=True)
      break
    elif r > 20:
      print ("r > 20")
      show_pcd(New_points_PlussY, vector=True)
      break
    else:
      continue

  # 			|== -Y ==|
  for r in range(1, 100):
    minus_y = first_point[0]-r
    New_points_MinusY = points[(points[:, 0] < first_point[0]+1)&
                        (points[:, 0] > minus_y)&
                        (points[:, 1] < first_point[1]+1)&
                        (points[:, 1] > first_point[1]-1)]

    Test_points = points[(points[:, 0] < first_point[0]+1)&
                        (points[:, 0] > minus_y-1)&
                        (points[:, 1] < first_point[1]+1)&
			(points[:, 1] > first_point[1]-1)]

    if (np.size(New_points_MinusY) == np.size(Test_points) and np.size(New_points_MinusY) != 0):
      print ("Minus Y r =", r)
      show_pcd(New_points_MinusY, vector=True)
      break
    elif r > 20:
      print ("r > 20")
      show_pcd(New_points_MinusY, vector=True)
      break
    else:
      continue

  # 			|== +X ==|
  for r in range(1, 100):
    pluss_x = first_point[1]+r
    New_points_PlussX = points[(points[:, 0] < first_point[0]+1)&
                        (points[:, 0] > first_point[0]-1)&
                        (points[:, 1] < pluss_x)&
                        (points[:, 1] > first_point[1]-1)]

    Test_points = points[(points[:, 0] < first_point[0]+1)&
                        (points[:, 0] > first_point[0]-1)&
                        (points[:, 1] < pluss_x+1)&
			(points[:, 1] > first_point[1]-1)]

    if (np.size(New_points_PlussX) == np.size(Test_points) and np.size(New_points_PlussX) != 0):
      print ("Pluss X r =", r)
      show_pcd(New_points_PlussX, vector=True)
      break
    elif r > 20:
      print ("r > 20")
      show_pcd(New_points_PlussX, vector=True)
      break
    else:
      continue
  # 			|== -X ==|
  for r in range(1, 100):
    minus_x = first_point[1]-r
    New_points_MinusX = points[(points[:, 0] < first_point[0]+1)&
                        (points[:, 0] > first_point[0]-1)&
                        (points[:, 1] < first_point[1]+1)&
                        (points[:, 1] > minus_x)]

    Test_points = points[(points[:, 0] < first_point[0]+1)&
                        (points[:, 0] > first_point[0]-1)&
                        (points[:, 1] < first_point[1]+1)&
			(points[:, 1] > minus_x-1)]

    if (np.size(New_points_MinusX) == np.size(Test_points) and np.size(New_points_MinusX) != 0):
      print ("Minus X r =", r)
      show_pcd(New_points_MinusX, vector=True)
      break
    elif r > 20:
      print ("r > 20")
      show_pcd(New_points_MinusX, vector=True)
      break
    else:
      continue
  New_points = np.vstack((New_points_PlussY, New_points_MinusY, New_points_PlussX, New_points_MinusX))
  OBJECT = Vector(New_points)
  print ("RESULT POINTS!")
  show_pcd(OBJECT)
  return OBJECT

