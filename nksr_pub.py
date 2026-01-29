import torch
import numpy as np
import nksr
from copy import deepcopy
import time

# PyTorch3D
from pytorch3d.ops import knn_points

# ROS2 imports
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class NKSRMapPublisher:
    def __init__(self, node: Node, device="cuda:0", knn=30):
        self.node = node
        self.device = torch.device(device)
        self.knn = knn

        self.stream = torch.cuda.Stream(device=self.device)
        self.reconstructor = nksr.Reconstructor(device=self.device, config="ks")

        self._pending_field = None
        self._has_pending = False

        self.mesh_pub = self.node.create_publisher(Marker, "/nksr_mesh", 1)
        self.node.create_timer(0.2, self._publish_if_ready)

        # Timings
        self.timing = {
            "normals": [],
            "reconstruction": [],
            "mesh_extract": []
        }

    def __call__(self, map_pc: np.ndarray):
        if map_pc.shape[0] < 1000:
            self.node.get_logger().warn("Pointcloud too small, skipping NKSR")
            return

        pts = torch.from_numpy(map_pc).float().to(self.device, non_blocking=True)
        
        t0 = time.time()
        nrm = self._estimate_normals_gpu(pts)
        t1 = time.time()
        self._pending_field = self.reconstructor.reconstruct(
            pts, nrm, 
            detail_level=0.1,
            fused_mode=False
        )
        t2 = time.time()

        self._has_pending = True

        # Record timings
        self.timing["normals"].append(t1 - t0)
        self.timing["reconstruction"].append(t2 - t1)
        self.node.get_logger().info(
            f"[TIMING] Normals: {t1-t0:.4f}s, Reconstruction enqueue: {t2-t1:.4f}s"
        )

    def _estimate_normals_gpu(self, pts: torch.Tensor) -> torch.Tensor:
        pts_b = pts.unsqueeze(0)
        knn_res = knn_points(pts_b, pts_b, K=self.knn)
        neighbors = knn_res.idx[0]

        nbr_pts = pts[neighbors]
        diffs = nbr_pts - pts[:, None, :]

        cov = torch.matmul(diffs.transpose(-2, -1), diffs)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        normals = eigvecs[..., :, 0]
        normals = torch.nn.functional.normalize(normals, dim=-1)
        return normals

    def _publish_if_ready(self):
        if not self._has_pending or self._pending_field is None:
            return

        torch.cuda.synchronize(self.stream)

        t0 = time.time()
        field = self._pending_field
        self._pending_field = None
        self._has_pending = False

        mesh = field.extract_dual_mesh()
        t1 = time.time()

        msg = self._mesh_to_marker(mesh)
        self.mesh_pub.publish(msg)
        t2 = time.time()

        self.timing["mesh_extract"].append(t1 - t0)
        self.node.get_logger().info(
            f"[TIMING] Mesh extract: {t1-t0:.4f}s, Publish: {t2-t1:.4f}s, "
            f"Vertices: {len(mesh.v)}"
        )

    def _mesh_to_marker(self, mesh):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "nksr_mesh"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0

        verts = mesh.v.cpu().numpy()
        faces = mesh.f.cpu().numpy() if torch.is_tensor(mesh.f) else mesh.f

        z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
        z_norm = (verts[:, 2] - z_min) / (z_max - z_min + 1e-8)

        for f in faces:
            for vi in f:
                v = verts[vi]
                pt = Point()
                pt.x, pt.y, pt.z = float(v[0]), float(v[1]), float(v[2])
                marker.points.append(pt)

        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.2
        marker.color.a = 1.0

        return marker
#
# import torch
# import numpy as np
# import nksr
# from copy import deepcopy
# import open3d as o3d

# # ROS2 imports
# from rclpy.node import Node
# from visualization_msgs.msg import Marker
# from sensor_msgs.msg import PointCloud2, PointField
# from std_msgs.msg import Header
# from geometry_msgs.msg import Point


# class NKSRMapPublisher2:
#     def __init__(self, node: Node, device="cuda:0", use_chunking=True, chunk_size=50000):
#         """
#         node: ROS2 Node that owns this publisher
#         device: CUDA device string
#         use_chunking: whether to split large point clouds into chunks
#         chunk_size: number of points per chunk if chunking is enabled
#         """
#         self.node = node
#         self.device = torch.device(device)
#         self.use_chunking = use_chunking
#         self.chunk_size = chunk_size

#         # CUDA stream for asynchronous work
#         self.stream = torch.cuda.Stream(device=self.device)

#         # Initialize reconstructor once
#         self.reconstructor = nksr.Reconstructor(device=self.device, config="ks")

#         # Pending field(s)
#         self._pending_fields = []
#         self._has_pending = False

#         # ROS2 publishers
#         self.mesh_pub = self.node.create_publisher(Marker, "/nksr_mesh", 1)

#         # Timer to check if pending mesh is ready and publish it
#         self.node.create_timer(0.2, self._publish_if_ready)

#     def __call__(self, map_pc: np.ndarray):
#         """
#         Accepts a numpy point cloud, triggers reconstruction asynchronously.
#         """
#         if map_pc.shape[0] < 1000:
#             self.node.get_logger().warn("Pointcloud too small, skipping NKSR")
#             return

#         if self.use_chunking and map_pc.shape[0] > self.chunk_size:
#             chunks = self._chunk_pointcloud(map_pc)
#             self._pending_fields = []

#             # Launch reconstruction for each chunk in its own CUDA stream
#             self.node.get_logger().info(f"Reconstructing in {len(chunks)} chunks...")
#             for chunk in chunks:
#                 pts = torch.from_numpy(chunk).float().to(self.device, non_blocking=True)
#                 nrm = self._estimate_normals_fast(pts)
#                 with torch.cuda.stream(self.stream):
#                     field = self.reconstructor.reconstruct(pts, nrm, detail_level=0.7)
#                     self._pending_fields.append(field)
#         else:
#             pts = torch.from_numpy(map_pc).float().to(self.device, non_blocking=True)
#             nrm = self._estimate_normals_fast(pts)
#             with torch.cuda.stream(self.stream):
#                 field = self.reconstructor.reconstruct(pts, nrm, detail_level=0.7)
#                 self._pending_fields = [field]

#         self._has_pending = True

#     def _chunk_pointcloud(self, points: np.ndarray):
#         n_points = points.shape[0]
#         chunks = []
#         for i in range(0, n_points, self.chunk_size):
#             chunks.append(points[i:i + self.chunk_size])
#         return chunks

#     def _estimate_normals_fast(self, pts: torch.Tensor) -> torch.Tensor:
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
#         pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
#         pcd.orient_normals_consistent_tangent_plane(k=30)
#         normals = torch.from_numpy(np.asarray(pcd.normals)).float().to(self.device, non_blocking=True)
#         return normals

#     def _publish_if_ready(self):
#         if not self._has_pending or len(self._pending_fields) == 0:
#             return

#         # Synchronize before extracting meshes
#         torch.cuda.synchronize(self.stream)

#         # Merge all meshes (if chunked)
#         all_vertices = []
#         all_faces = []
#         offset = 0
#         for field in self._pending_fields:
#             mesh = field.extract_dual_mesh()
#             v = mesh.v.cpu().numpy() if torch.is_tensor(mesh.v) else np.asarray(mesh.v)
#             f = mesh.f.cpu().numpy() if torch.is_tensor(mesh.f) else np.asarray(mesh.f)
#             all_vertices.append(v)
#             all_faces.append(f + offset)
#             offset += len(v)

#         vertices = np.vstack(all_vertices)
#         faces = np.vstack(all_faces)

#         # Reset pending
#         self._pending_fields = []
#         self._has_pending = False

#         # Publish mesh as Marker
#         marker = self._vertices_faces_to_marker(vertices, faces)
#         self.mesh_pub.publish(marker)
#         self.node.get_logger().info(f"Published NKSR mesh with {len(vertices)} vertices")

#     def _vertices_faces_to_marker(self, vertices: np.ndarray, faces: np.ndarray):
#         """
#         Convert vertices and faces to a Marker message with height-based coloring.
#         """
#         marker = Marker()
#         marker.header.frame_id = "map"
#         marker.header.stamp = self.node.get_clock().now().to_msg()
#         marker.ns = "nksr_mesh"
#         marker.id = 0
#         marker.type = Marker.TRIANGLE_LIST
#         marker.action = Marker.ADD
#         marker.pose.orientation.w = 1.0
#         marker.scale.x = 1.0
#         marker.scale.y = 1.0
#         marker.scale.z = 1.0

#         z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
#         z_range = max(z_max - z_min, 1e-5)

#         for f in faces:
#             for vi in f:
#                 v = vertices[vi]
#                 pt = Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))
#                 marker.points.append(pt)
#                 # height-based coloring (blue at min, red at max)
#                 z_norm = (v[2] - z_min) / z_range
#                 from std_msgs.msg import ColorRGBA
#                 color = ColorRGBA()
#                 color.r = float(z_norm)
#                 color.g = 0.2
#                 color.b = float(1.0 - z_norm)
#                 color.a = 1.0
#                 marker.colors.append(color)

#         return marker



# import torch
# import numpy as np
# import nksr
# from pycg import vis
# from copy import deepcopy
# import open3d as o3d

# # ROS2 imports
# from rclpy.node import Node
# from visualization_msgs.msg import Marker
# from geometry_msgs.msg import Point

# class NKSRMapPublisher:
#     #this is the original version
#     def __init__(self, node: Node, device="cuda:0"):
#         """
#         node: ROS2 Node that owns this publisher
#         device: CUDA device string
#         """
#         self.node = node
#         self.device = torch.device(device)

#         # CUDA stream to enqueue work asynchronously
#         self.stream = torch.cuda.Stream(device=self.device)

#         # Initialize reconstructor once
#         self.reconstructor = nksr.Reconstructor(device=self.device, config="ks")

#         # Pending field (GPU work not yet extracted)
#         self._pending_field = None
#         self._has_pending = False

#         # ROS2 publisher for meshes (as Marker for RViz)
#         self.mesh_pub = self.node.create_publisher(Marker, "/nksr_mesh", 1)

#         # Timer to check if pending mesh is ready and publish it
#         self.node.create_timer(0.2, self._publish_if_ready)

#     def __call__(self, map_pc: np.ndarray):
#         """
#         Accepts a numpy pointcloud, triggers reconstruction asynchronously.
#         """
#         if map_pc.shape[0] < 1000:  # skip tiny clouds
#             self.node.get_logger().warn("Pointcloud too small, skipping NKSR")
#             return

#         # Convert to torch tensor on GPU
#         pts = torch.from_numpy(map_pc).float().to(self.device, non_blocking=True)

#         # Estimate normals (Open3D CPU → tensor)
#         nrm = self._estimate_normals_fast(pts)

#         # Enqueue reconstruction in CUDA stream
#         self._pending_field = self.reconstructor.reconstruct(
#             pts, nrm, detail_level=0.1
#         )

#         self._has_pending = True

#     def _estimate_normals_fast(self, pts: torch.Tensor) -> torch.Tensor:
#         """
#         Estimate normals for the pointcloud quickly using Open3D.
#         Returns a torch.Tensor on the same device as pts.
#         """
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
#         pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
#         pcd.orient_normals_consistent_tangent_plane(k=30)
#         normals = torch.from_numpy(np.asarray(pcd.normals)).float().to(self.device, non_blocking=True)
#         return normals

#     def _publish_if_ready(self):
#         """
#         Check if reconstruction is done; extract mesh and publish.
#         Runs on ROS2 timer.
#         """
#         if not self._has_pending or self._pending_field is None:
#             return

#         # Synchronize stream before extracting mesh
#         torch.cuda.synchronize(self.stream)

#         field = self._pending_field
#         self._pending_field = None
#         self._has_pending = False

#         # Extract mesh from NKSR field
#         mesh = field.extract_dual_mesh()  # mesh.v, mesh.f

#         # Convert mesh to Marker for RViz
#         msg = self._mesh_to_marker(mesh)
#         self.mesh_pub.publish(msg)
#         self.node.get_logger().info(f"Published NKSR mesh with {len(mesh.v)} vertices")

#     def _mesh_to_marker(self, mesh):
#         """
#         Convert NKSR mesh to a ROS2 Marker message (triangle list),
#         coloring vertices based on height (z-axis).
#         """
#         marker = Marker()
#         marker.header.frame_id = "map"
#         marker.header.stamp = self.node.get_clock().now().to_msg()
#         marker.ns = "nksr_mesh"
#         marker.id = 0
#         marker.type = Marker.TRIANGLE_LIST
#         marker.action = Marker.ADD
#         marker.pose.orientation.w = 1.0
#         marker.scale.x = 1.0
#         marker.scale.y = 1.0
#         marker.scale.z = 1.0

#         # Move vertices/faces to CPU and convert to NumPy
#         verts = mesh.v.cpu().numpy()  # <-- FIX HERE
#         faces = mesh.f.cpu().numpy() if torch.is_tensor(mesh.f) else mesh.f

#         # Compute per-vertex height color
#         z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
#         z_norm = (verts[:, 2] - z_min) / (z_max - z_min + 1e-8)

#         # Color map: red = high, blue = low
#         colors = np.zeros_like(verts)
#         colors[:, 0] = z_norm      # Red
#         colors[:, 1] = 0.2         # Slight green
#         colors[:, 2] = 1 - z_norm  # Blue

#         # Attach vertices to marker (RViz TRIANGLE_LIST)
#         for f in faces:
#             for vi in f:
#                 v = verts[vi]
#                 # c = colors[vi]  # per-vertex color (not fully supported in TRIANGLE_LIST)
#                 pt = Point()
#                 pt.x, pt.y, pt.z = float(v[0]), float(v[1]), float(v[2])
#                 marker.points.append(pt)

#         # Overall fallback color
#         marker.color.r = 1.0
#         marker.color.g = 0.5
#         marker.color.b = 0.2
#         marker.color.a = 1.0

#         return marker

