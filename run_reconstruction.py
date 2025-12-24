import torch
import numpy as np
import open3d as o3d
import nksr
from pycg import vis
from kiss_icp.filtering import filter_statistical_outlier_removal
# --- Set device ---
device = torch.device("cpu")

# --- Load point cloud ---
data = np.load('pointcloud_raw.npz')
# Assuming the point cloud is saved under 'xyz'
xyz = data['xyz']  # shape [N, 3]
xyz=filter_statistical_outlier_removal(xyz)
# --- Convert to Open3D point cloud ---
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# --- Estimate normals ---
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
# Orient normals consistently
pcd.orient_normals_consistent_tangent_plane(k=30)

# --- Convert to torch tensors on GPU ---
input_xyz = torch.from_numpy(np.asarray(pcd.points)).float().to(device)
input_normal = torch.from_numpy(np.asarray(pcd.normals)).float().to(device)

# --- Initialize NKSR reconstructor ---
reconstructor = nksr.Reconstructor(device=device,
                                    config='ks')

# --- Run reconstruction ---
field = reconstructor.reconstruct(input_xyz, 
                                    input_normal, 
                                    detail_level=0,#is by default 0, and you can tune it from 0.0 to 1.0, 
                                    # where 0.0 contains the least detail but may be more robust towards noise, 
                                    # and 1.0 has the most details but could overfit to noise and leads to memory overflow.

                                    )

# --- Extract mesh ---
mesh = field.extract_dual_mesh()

# --- Visualize ---
vis.show_3d([vis.mesh(mesh.v, mesh.f)])