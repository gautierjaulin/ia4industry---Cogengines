import numpy as np
import open3d as o3d
import os
from matplotlib import pyplot as plt
import pyransac3d as pyrsc
from pcdiff import knn_graph, estimate_basis, build_grad_div

PLY_ROOT = "./data"
filenames = [f"data{i}.ply" for i in range(2, 3)]

for filename in filenames:
    # Lecture du fichier PLY sous la forme d'un nuage de points
    plyfile_path = os.path.join(PLY_ROOT, filename)
    pcd = o3d.io.read_point_cloud(plyfile_path)

    pcd.estimate_normals()
    pcd_normals = np.asarray(pcd.normals)
    print("PCD normals: ", pcd_normals)

    # On récupère les points (sans la couleur)
    points = np.asarray(pcd.points)
    # print(points.min(), points.max())
    # mask = np.abs(points[:, 2]) < 0.4
    # print("Mask: ", np.sum(mask))
    # points = points[mask, :]
    print("Points: ", points.shape)

    # # Generate kNN graph
    # edge_index = knn_graph(points, 20)
    # # Estimate normals and local frames
    # basis = estimate_basis(points, edge_index)
    # # Build gradient and divergence operators (Scipy sparse matrices)
    # grad, div = build_grad_div(points, *basis, edge_index)

    # normals, x_basis, y_basis = basis

    # print(grad.shape, normals.shape, div.shape)
    # indices = np.zeros([grad.shape[0]-1, ], dtype=np.int64)
    # indices[0::2] = np.arange(0,  grad.shape[0]-1, step=2)
    # indices[1::2] = np.arange(2,  grad.shape[0], step=2)
    # tmp = grad @ normals
    # courbure = np.add.reduceat(tmp, indices, axis=0)[::2]
    # courbure = np.sqrt(np.sum(courbure*courbure, axis=1))

    # print("Courbure: ", courbure.shape, courbure)


    # Algorithme de régression de l'équation du plan via RANSAC
    # On obtient les coefficient de l'équation et l'index des points qui sont considérés "plateau"
    plane1 = pyrsc.Plane()
    best_eq, best_inliers = plane1.fit(points, thresh=0.003, minPoints=1000, maxIteration=500)
    print("Plane Equation: ", best_eq)

    # On cherche les angles du plan
    points_min = points.min(axis=0)*1.2
    points_max = points.max(axis=0)*1.2
    points_xy = np.stack((points_min[:2], points_max[:2], [points_min[0], points_max[1]], [points_max[0], points_min[1]]), axis=0)

    print(points_xy.shape)

    # On applique l'équation de plan
    plane_points = -(best_eq[0]*points_xy[:,0] + best_eq[1]*points_xy[:,1] + best_eq[3]) / best_eq[2]
    plane_points = np.concatenate((points_xy, np.expand_dims(plane_points, axis=1),), axis=1)
    # print(plane_points.shape)

    # print(best_inliers.shape)
    # On récupère les points "plateau"
    best_inliers_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[best_inliers, :]))
    # best_inliers_pcd.paint_uniform_color([1, 0.706, 0])
    # best_inliers_pcd.paint_uniform_color([0.0, 0.0, 0.0])

    # On récupère les points "potentiel objet" et on les colore en noir
    mask = np.ones(points.shape[0], dtype=np.bool8)
    mask[best_inliers] = 0
    outliers_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[mask, :]))
    print(outliers_pcd.has_colors())
    # outliers_pcd.paint_uniform_color([0.0, 0.0, 1.0])

    # Algo de ligne de partage des eaux
    outliers3d = points[mask, :][::5]
    outliers4d = np.concatenate([outliers3d, np.ones((outliers3d.shape[0], 1))], axis=1)
    print("Outliers4D: ", outliers4d.shape)

    dist2plane = np.abs(np.sum(best_eq*outliers4d, axis=1))/np.sqrt(best_eq[0]**2 + best_eq[1]**2 + best_eq[2]**2)
    sorted_indices = np.argsort(dist2plane, axis=0)[::-1]
    dist2plane_max = np.max(dist2plane)

    outliers3d = outliers3d[sorted_indices, :]
    dist2plane = dist2plane[sorted_indices]

    print("Dist2plane: ", dist2plane.shape, dist2plane_max, dist2plane)
    STEP_SIZE = 0.008
    NEIGH_SIZE = STEP_SIZE*np.sqrt(2)
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
        new_mask = np.logical_and(current_level <= dist2plane, dist2plane < current_level+STEP_SIZE)
        # new_mask = np.logical_and(current_level >= dist2plane, dist2plane > current_level-STEP_SIZE)
        new_indices = np.arange(0, len(new_mask), dtype=np.int64)[new_mask]
        new_points = outliers3d[new_mask, :]
        new_labels = np.zeros_like(new_indices)
        for i in range(len(new_indices)):
            distance2point = np.sqrt(np.sum((treated_points-new_points[i, :])**2,axis=1))
            # print("distance2point:" , distance2point.shape, distance2point)
            if np.any((treated_labels[distance2point < NEIGH_SIZE] > 0) + (treated_labels[distance2point < NEIGH_SIZE] == -3)):
                new_labels[i] = -1 # S
            else:
                new_labels[i] = -2 # M
        S_points: list = new_points[new_labels == -1, :].tolist()
        current_points = treated_points.copy()
        current_labels = treated_labels.copy()
        # print("Current labels: ", current_labels.shape)
        while len(S_points):
            distance2point = np.sqrt(np.sum((current_points-np.array(S_points[0]))**2,axis=1))
            neigh_labels = current_labels[distance2point < NEIGH_SIZE]
            unique_neigh_labels = np.unique(neigh_labels)
            if len(unique_neigh_labels) > 1:
                current_labels = np.concatenate([current_labels, np.array([[-3,],])], axis=0) # LPE
            else:
                # print("Unique Neigh Labels: ", unique_neigh_labels.shape, unique_neigh_labels)
                current_labels = np.concatenate([current_labels, unique_neigh_labels[0:1, np.newaxis]], axis=0) # Propagation du cluster
            distance2point = np.sqrt(np.sum((new_points-np.array(S_points[0]))**2,axis=1))
            S_points += new_points[np.logical_and(new_labels == -2, distance2point < NEIGH_SIZE), :].tolist()
            new_labels[np.logical_and(new_labels == -2, distance2point < NEIGH_SIZE)] = -1
            current_points = np.concatenate([current_points, np.array(S_points[0])[np.newaxis, :]], axis=0)
            S_points.pop(0)
            # print("Spoints: ", len(S_points))

        M_points: list = new_points[new_labels == -2, :].tolist()
            
        while len(M_points):
            # print("M_points: ", M_points[0], type(M_points[0]))
            distance2point = np.sqrt(np.sum((current_points-np.array(M_points[0]))**2,axis=1))
            neigh_labels = current_labels[distance2point < NEIGH_SIZE]
            if len(neigh_labels > 0):
                current_labels = np.concatenate([current_labels, neigh_labels[0:1, :]], axis=0) # Propagation du cluster
            else:
                current_labels = np.concatenate([current_labels, np.array([next_label,])[:, np.newaxis]], axis=0) # Nouveau cluster
                next_label +=1
            current_points = np.concatenate([current_points, np.array(M_points[0])[np.newaxis, :]], axis=0)
            M_points.pop(0)
            # print("Mpoints: ", len(M_points))

        current_level = current_level - STEP_SIZE
        # treated_points = np.concatenate([treated_points, new_points], axis=0)
        treated_labels = current_labels.copy()
        treated_points = current_points.copy()
    
    outliers_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(treated_points))
    labels = treated_labels[:, 0]

    print("Total de points: ", np.asarray(best_inliers_pcd.points).shape, np.asarray(outliers_pcd.points).shape, np.asarray(best_inliers_pcd.points).shape[0] + np.asarray(outliers_pcd.points).shape[0], points.shape)
    
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     labels = np.array(outliers_pcd.cluster_dbscan(eps=0.003, min_points=50, print_progress=True))
    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(np.double((labels+1)) / np.double((max_label+1 if max_label > 0 else 2)))
    colors = np.zeros((labels.shape[0], 3), dtype=np.double)
    for i in range(max_label):
        colors[labels == i, :] = np.random.rand(1, 3)
    colors[labels < 0, :] = 1
    colors[labels == -3, :] = 0
    outliers_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # On trace le plan
    plane_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(plane_points))
    plane_mesh, _ = o3d.geometry.TetraMesh.create_from_point_cloud(plane_pcd)

    plane_normal = np.asarray(best_eq[:3])
    plane_normal = plane_normal / np.sqrt(np.sum(plane_normal*plane_normal))
    print("Plane Normal: ", plane_normal)

    dot_products = np.sum(plane_normal[:, np.newaxis] * pcd_normals.T, axis=0)
    print("Dot product: ", dot_products.shape, dot_products)

    # dot_product_colors = np.stack([np.abs(dot_products), *((np.zeros_like(dot_products),)*2)], axis=1)
    dot_product_colors = np.stack([np.abs(dot_products),]*3, axis=1)
    print(dot_product_colors.shape)
    # outliers_pcd.colors = o3d.utility.Vector3dVector(dot_product_colors[mask])
    best_inliers_pcd.colors = o3d.utility.Vector3dVector(dot_product_colors[best_inliers])

    # Coloration basée une info de pseudo courbure
    # courbure_colors = np.stack([np.abs(courbure),]*3, axis=1)
    # outliers_pcd.colors = o3d.utility.Vector3dVector(courbure_colors[mask])
    # best_inliers_pcd.colors = o3d.utility.Vector3dVector(courbure_colors[best_inliers])
    print(np.asarray(outliers_pcd.colors))

    # On affiche le tout (plan, points jaune et bleu)
    o3d.visualization.draw_geometries([best_inliers_pcd, plane_mesh, outliers_pcd], left=0, top=100, window_name=filename)
    print("Mean: ", points.mean(axis=0))
    print("Std: ", points.std(axis=0))

    # # Calcul de dérivées (travaux en cours, ne fonctionne pas pour le moment)
    # num_elem = points.shape[0]
    # dGdx, t_size = derivative_gaussian_kernel_3d(0.1)
    # scale = 0.01 # échelle d'une case du noyau dans le domaine 3d
    # neighbours = np.zeros((2*t_size+1, 2*t_size+1, 2*t_size+1,), dtype=np.double)
    # floored_points = np.floor(points).astype(np.int32)
    # for idx in range(num_elem):
    #     floored_point = floored_points[idx, :]
    #     diff_points = floored_points - floored_point
    #     diff_points
    

    # Affiche des histogrammes en z
    # plt.figure()
    # plt.hist(points[:, 2], np.linspace(-0.3, -0.2, 256))

# plt.show()

## Petit test sur le format de la donnée
# from plyfile import PlyData, PlyElement

# plydata = PlyData().read("data/data1.ply")
# print(plydata)
# vertex_list = plydata.elements[0].data.tolist()
# vertex_img = np.array(vertex_list)
# print(vertex_img.shape)
# # print(640*480)
# # vertex_img = vertex_img.reshape(480, 640, 6)