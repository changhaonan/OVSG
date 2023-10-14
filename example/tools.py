"""Tools that preprocess the data."""
import open3d as o3d
import os


def convert_ply2pcd(ply_path, pcd_path):
    """Convert .ply to .pcd."""
    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.io.write_point_cloud(pcd_path, pcd)
    return pcd


if __name__ == "__main__":
    root_dir = os.path.join(os.path.dirname(__file__), "..")
    demo_ply_file = os.path.join(root_dir, "test_data", "cus_scene0001_00", "cus_scene0001_00_vh_clean_2.ply")
    demo_pcd_file = os.path.join(root_dir, "test_data", "cus_scene0001_00", "scan-00.pcd")  # fit with OVIR-3D format
    convert_ply2pcd(demo_ply_file, demo_pcd_file)
