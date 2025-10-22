import open3d as o3d
import time

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Point Cloud Sequence")

pcd = None
for t in range(21):
    filename = f"pointcloud_frame_{t}.ply"
    print(f"Loading {filename}...")
    new_pcd = o3d.io.read_point_cloud(filename)
    if len(new_pcd.points) == 0:
        print(f"No points in {filename}, skipping.")
        continue
    if pcd is None:
        pcd = new_pcd
        vis.add_geometry(pcd)
    else:
        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors
        vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    print(f"Displaying {filename} for 1 second...")
    time.sleep(1)

vis.destroy_window()