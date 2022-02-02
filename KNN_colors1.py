import numpy as np
import open3d as o3d
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from My_Functions import _get_pcd


def main():
  # Get avenes colors ==========================================
  Kruze_colors = _get_pcd("Kruze/Analysis/Train_x.ply", colors=1)["colors"]	# Finction _get_pcd() returns dict
  Glaze_colors = _get_pcd("Glaze/Analysis/Train_x1.ply", colors=1)["colors"]
  Kruze_colors_T = _get_pcd("Kruze/Analysis/Test_x.ply", colors=1)["colors"]
  Glaze_colors_T = _get_pcd("Glaze/Analysis/sample_0.ply", colors=1)["colors"]
  assert np.shape(Glaze_colors_T) != ()
  assert np.shape(Kruze_colors_T) != ()
  assert np.shape(Glaze_colors) != ()
  assert np.shape(Kruze_colors) != ()

  # stack avenes and cidonijas colors in one training set and label them =====
  train_X = np.vstack((Kruze_colors, Glaze_colors))
  Kruze_train_labels = np.ones(np.shape(Kruze_colors)[0])  # label for Burka colors is 1
  Glaze_train_labels = np.zeros(np.shape(Glaze_colors)[0])  # label for Glaze colors is 0
  train_y = np.append(Kruze_train_labels , Glaze_train_labels)

  # Define and train KNN model ===============================================
  knn = KNeighborsClassifier(n_neighbors=5)
  knn.fit(train_X, train_y)

  # Get avenes and cidonijas testing data (colors) and label tham ============
  test_X = np.vstack((Kruze_colors_T, Glaze_colors_T))
  Kruze_test_labels = np.ones(np.shape(Kruze_colors_T)[0])
  Glaze_test_labels = np.zeros(np.shape(Glaze_colors_T)[0])
  test_y = np.append(Kruze_test_labels , Glaze_test_labels)

  # Test the model and print and Accuracy of the model =======================
  y_pred = knn.predict(test_X)
  print ("Accuracy:", metrics.accuracy_score(test_y, y_pred))
  '''
  # define pcd
  pcd = o3d.geometry.PointCloud()
  pcd_points = pcd_points[y_pred == 1]
  pcd_colors = pcd_colors[y_pred == 1]
  pcd.points = o3d.utility.Vector3dVector(pcd_points)
  pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

  # show pcd
  o3d.visualization.draw_geometries([pcd])
  '''

if __name__ == "__main__":
  exit(main())

