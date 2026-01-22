import numpy as np
import open3d as o3d
import os
from matplotlib import pyplot as plt
import pyransac3d as pyrsc

def RANSAC_plane(points: np.ndarray, thresh=0.003, minPoints=1000, maxIteration=500):
    plane1 = pyrsc.Plane()
    plane_eq, inliers_indices = plane1.fit(points, thresh=thresh, minPoints=minPoints, maxIteration=maxIteration)
    return plane_eq, inliers_indices

def create_plane_mesh(points: np.ndarray, plane_eq):
    points_min = points.min(axis=0)*1.2
    points_max = points.max(axis=0)*1.2
    points_xy = np.stack((points_min[:2], points_max[:2], [points_min[0], points_max[1]], [points_max[0], points_min[1]]), axis=0)

    # On applique l'équation de plan
    plane_points = -(plane_eq[0]*points_xy[:,0] + plane_eq[1]*points_xy[:,1] + plane_eq[3]) / plane_eq[2]
    plane_points = np.concatenate((points_xy, np.expand_dims(plane_points, axis=1),), axis=1)
    plane_pcd = points2pcd(plane_points)
    plane_mesh, _ = o3d.geometry.TetraMesh.create_from_point_cloud(plane_pcd)
    return plane_mesh

def points2pcd(points: np.ndarray):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

def segment_point_cloud(point_cloud: o3d.geometry.PointCloud, step_size = 0.008, neigh_size = 1.3, subsampling_rate = 5):
    points = np.asarray(point_cloud.points)
    plane_eq, inliers_indices = RANSAC_plane(points)

    mask = np.ones(points.shape[0], dtype=np.bool8)
    mask[inliers_indices] = 0

    # Algo de ligne de partage des eaux
    outliers3d = points[mask, :][::subsampling_rate]
    outliers4d = np.concatenate([outliers3d, np.ones((outliers3d.shape[0], 1))], axis=1)
    print("Outliers4D: ", outliers4d.shape)

    dist2plane = np.abs(np.sum(plane_eq*outliers4d, axis=1))/np.sqrt(plane_eq[0]**2 + plane_eq[1]**2 + plane_eq[2]**2)
    sorted_indices = np.argsort(dist2plane, axis=0)[::-1]
    dist2plane_max = np.max(dist2plane)

    outliers3d = outliers3d[sorted_indices, :]
    dist2plane = dist2plane[sorted_indices]

    print("Dist2plane: ", dist2plane.shape, dist2plane_max, dist2plane)
    step_size = 0.008
    neigh_size = step_size*np.sqrt(2)
    # Init des labels
    next_label = 1
    current_level = dist2plane_max
    treated_points = np.zeros((0, 3), dtype=np.double)
    treated_labels = np.zeros((0, 1), dtype=np.int64)
    iteration = 0
    while treated_points.shape[0] < outliers3d.shape[0]:
        # if iteration == 3:
        #     break
        iteration += 1
        print("Progress: ", treated_points.shape[0],  outliers3d.shape[0])
        new_mask = np.logical_and(current_level <= dist2plane, dist2plane < current_level+step_size)
        # new_mask = np.logical_and(current_level >= dist2plane, dist2plane > current_level-step_size)
        new_indices = np.arange(0, len(new_mask), dtype=np.int64)[new_mask]
        new_points = outliers3d[new_mask, :]
        new_labels = np.zeros_like(new_indices)
        for i in range(len(new_indices)):
            distance2point = np.sqrt(np.sum((treated_points-new_points[i, :])**2,axis=1))
            # print("distance2point:" , distance2point.shape, distance2point)
            if np.any((treated_labels[distance2point < neigh_size] > 0) + (treated_labels[distance2point < neigh_size] == -3)):
                new_labels[i] = -1 # S
            else:
                new_labels[i] = -2 # M
        S_points: list = new_points[new_labels == -1, :].tolist()
        current_points = treated_points.copy()
        current_labels = treated_labels.copy()
        # print("Current labels: ", current_labels.shape)
        while len(S_points):
            distance2point = np.sqrt(np.sum((current_points-np.array(S_points[0]))**2,axis=1))
            neigh_labels = current_labels[distance2point < neigh_size]
            unique_neigh_labels = np.unique(neigh_labels)
            if len(unique_neigh_labels) > 1:
                current_labels = np.concatenate([current_labels, np.array([[-3,],])], axis=0) # LPE
            else:
                # print("Unique Neigh Labels: ", unique_neigh_labels.shape, unique_neigh_labels)
                current_labels = np.concatenate([current_labels, unique_neigh_labels[0:1, np.newaxis]], axis=0) # Propagation du cluster
            distance2point = np.sqrt(np.sum((new_points-np.array(S_points[0]))**2,axis=1))
            S_points += new_points[np.logical_and(new_labels == -2, distance2point < neigh_size), :].tolist()
            new_labels[np.logical_and(new_labels == -2, distance2point < neigh_size)] = -1
            current_points = np.concatenate([current_points, np.array(S_points[0])[np.newaxis, :]], axis=0)
            S_points.pop(0)
            # print("Spoints: ", len(S_points))

        M_points: list = new_points[new_labels == -2, :].tolist()
            
        while len(M_points):
            # print("M_points: ", M_points[0], type(M_points[0]))
            distance2point = np.sqrt(np.sum((current_points-np.array(M_points[0]))**2,axis=1))
            neigh_labels = current_labels[distance2point < neigh_size]
            if len(neigh_labels > 0):
                current_labels = np.concatenate([current_labels, neigh_labels[0:1, :]], axis=0) # Propagation du cluster
            else:
                current_labels = np.concatenate([current_labels, np.array([next_label,])[:, np.newaxis]], axis=0) # Nouveau cluster
                next_label +=1
            current_points = np.concatenate([current_points, np.array(M_points[0])[np.newaxis, :]], axis=0)
            M_points.pop(0)
            # print("Mpoints: ", len(M_points))

        current_level = current_level - step_size
        # treated_points = np.concatenate([treated_points, new_points], axis=0)
        treated_labels = current_labels.copy()
        treated_points = current_points.copy()
    
    inliers_pcd = points2pcd(points[inliers_indices])
    outliers_pcd = points2pcd(treated_points)
    labels = treated_labels[:, 0]

    return inliers_pcd, outliers_pcd, labels

def main(filename: str):
    pcd = o3d.io.read_point_cloud(filename)
    plane_eq, inliers_indices = RANSAC_plane(np.asarray(pcd.points))
    plane_mesh = create_plane_mesh(np.asarray(pcd.points), plane_eq)
    inliers_pcd, outliers_pcd, labels = segment_point_cloud(pcd)

    # Coloration
    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")
    colors = np.zeros((labels.shape[0], 3), dtype=np.double)
    for i in range(max_label):
        colors[labels == i, :] = np.random.rand(1, 3)
    colors[labels < 0, :] = 1
    colors[labels == -3, :] = 0
    outliers_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([inliers_pcd, plane_mesh, outliers_pcd], left=0, top=100, window_name=filename)
