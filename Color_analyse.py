import numpy as np
import open3d as o3d
from My_Functions import _get_pcd
import datetime


def main():
  begin_time = datetime.datetime.now()

  # Get avenes colors ==========================================
  OB_colors = _get_pcd("../samples/sample_1.pcd", colors=True)["colors"]	# Finction _get_pcd() returns dict
  NOTOB_colors4 = _get_pcd("../samples/Not_object4.pcd", colors=True)["colors"]

  # Test pointcloud ============================================
  pcd_points = _get_pcd("../par_1.pcd", points=True, colors=True)
  pcd_colors = pcd_points["colors"]
  pcd_points = pcd_points["points"]

  # Print pcd loading time =============================
  print ("pcd loading time:", datetime.datetime.now() - begin_time)

  # Get a Mean's of all pointcloud's colors ============
  OB_mean = np.mean(OB_colors, axis=0)
  NOTOB_mean4 = np.mean(NOTOB_colors4, axis=0)

  # Get a Norm's to an object mean and not a object sample mean ========
  OB_Norm = np.linalg.norm(pcd_colors - OB_mean, axis=1)
  NOTOB_Norm4 = np.linalg.norm(pcd_colors - NOTOB_mean4, axis=1)

  # Atstaat punktus ar objekta krasam =========
  pcd_points = pcd_points[(OB_Norm < NOTOB_Norm4) & (OB_Norm < 0.5)]
  pcd_colors = pcd_colors[(OB_Norm < NOTOB_Norm4) & (OB_Norm < 0.5)]

  # define pcd ================================
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(pcd_points)
  pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

  # Print script running time =================
  print ("Skript runung time:", datetime.datetime.now() - begin_time)

  # show pcd ==================================
  o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
  exit(main())


