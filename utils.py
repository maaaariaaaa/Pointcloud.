import numpy as np
import open3d as o3d
import os

def display_pointcloud(xyz, point_size=1) -> None:
    xyz = np.nan_to_num(xyz).reshape(-1, 3)
    point_cloud_open3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))

    #rgb = rgb_cropped.reshape(-1, 3)
    #point_cloud_open3d.colors = o3d.utility.Vector3dVector(rgb / 255)

    visualizer = o3d.visualization.Visualizer()  # pylint: disable=no-member
    visualizer.create_window()
    #visualizer.set_full_screen(True)
    visualizer.add_geometry(point_cloud_open3d)
    #visualizer.get_render_option().background_color = (0, 0, 0)
    visualizer.get_render_option().point_size = point_size
    visualizer.get_render_option().show_coordinate_frame = True
    visualizer.get_view_control().set_front([0, 1, 0])
    visualizer.get_view_control().set_up([0, 0, 1])
    visualizer.run()
    visualizer.destroy_window()
    return

def read_off(file):
    off_header = file.readline().strip()
    if 'OFF' == off_header:
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    else:
        n_verts, n_faces, __ = tuple([int(s) for s in off_header[3:].split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def create_dirs(dirs):
    if isinstance(dirs, list):
        for cur_dir in dirs:
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)
                print(f'Directory {cur_dir} created')
    else:
        if not os.path.exists(dirs):
            os.makedirs(dirs)
            print(f'Directory {dirs} created')