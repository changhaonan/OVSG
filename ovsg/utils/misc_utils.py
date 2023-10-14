from typing import List, Tuple, Union, Any
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import heapq
import math
from collections import deque
from scipy.ndimage import label
import copy
import time
from contextlib import contextmanager
from datetime import datetime
from PIL import Image
import pickle
import os

#### Safe exec ####


def merge_dicts(dicts):
    return {k: v for d in dicts for k, v in d.items()}


def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ["import", "__"]
    for phrase in banned_phrases:
        assert phrase not in code_str

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([gvars, {"exec": empty_fn, "eval": empty_fn}])
    exec(code_str, custom_gvars, lvars)


#### Navigation Utils ####


def point_to_grid_index_2d(point, grid_resolution):
    return np.floor(point[:2] / grid_resolution).astype(int)


def pcd_to_occupancy_map(pcd, voxel_size, z_min, z_max, post_process=True):
    """Convert point cloud to occupancy map

    Args:
        pcd (open3d.geometry.PointCloud): point cloud
        voxel_size (float, optional): voxel size. Defaults to 0.1.
        z_min (float, optional): min z value. Defaults to 0.
        z_max (float, optional): max z value. Defaults to 2.
        post_process (bool, optional): whether to post process the occupancy map. Defaults to True.

    Returns:
        np.ndarray: occupancy map
    """
    # get voxel grid
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_down, voxel_size=voxel_size)
    # get occupancy map
    grid_resolution = voxel_size  # in meters
    grid_size_2d = np.array([10, 10])  # X, Y dimensions in meters
    grid_shape_2d = np.ceil(grid_size_2d / grid_resolution).astype(int)
    occupancy_grid_2d = np.zeros(grid_shape_2d, dtype=np.uint8)
    for voxel in voxel_grid.get_voxels():
        center = voxel.grid_index.astype(float) * voxel_size
        z = center[2]
        if z_min <= z <= z_max:
            grid_index_2d = point_to_grid_index_2d(center, grid_resolution)
            if np.all(grid_index_2d >= 0) and np.all(grid_index_2d < grid_shape_2d):
                # occupancy_grid_2d[tuple(grid_index_2d)] += 1
                # (y, x)
                occupancy_grid_2d[grid_index_2d[1], grid_index_2d[0]] = 1
    # post processing
    # Only left one connected component with start
    start_index = point_to_grid_index_2d(np.array([0, 0, 0]), grid_resolution)
    occupancy_grid_2d = np.asarray(mark_connected_cells(occupancy_grid_2d, start_index), dtype=np.uint8)
    return occupancy_grid_2d, voxel_grid.origin, voxel_size


def astar(maze: List, start: Tuple, end: Tuple, enable_vis=False):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def valid_coordinate(coord, maze):
        if 0 <= coord[0] < len(maze) and 0 <= coord[1] < len(maze[0]) and maze[coord[0]][coord[1]] != 1:
            return True
        return False

    if not valid_coordinate(start, maze) or not valid_coordinate(end, maze):
        raise ValueError("Invalid start or end coordinates.")

    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def neighbors(node):
        dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        result = []
        for dir in dirs:
            neighbor = (node[0] + dir[0], node[1] + dir[1])
            if valid_coordinate(neighbor, maze):
                result.append(neighbor)
        return result

    def visualize(maze, path):
        h, w = len(maze), len(maze[0])
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[np.array(maze) == 1] = (0, 0, 255)  # Wall color (blue)
        for step in path:
            if step == start:
                img[step] = (0, 255, 0)  # Start color (green)
            elif step == end:
                img[step] = (255, 0, 0)  # End color (red)
            else:
                img[step] = (255, 255, 255)  # Path color (white)

        img = cv2.resize(img, (w * 10, h * 10), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("A* Pathfinding", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    visited = set()
    frontier = []
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    heapq.heappush(frontier, (f_score[start], start))

    while frontier:
        current = heapq.heappop(frontier)[1]
        if current == end:
            path = reconstruct_path(came_from, current)
            if enable_vis:
                visualize(maze, path)
            return path
        visited.add(current)

        for neighbor in neighbors(current):
            tentative_g_score = g_score[current] + 1
            if neighbor in visited and tentative_g_score >= g_score.get(neighbor, math.inf):
                continue
            if tentative_g_score < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                if neighbor not in [i[1] for i in frontier]:
                    heapq.heappush(frontier, (f_score[neighbor], neighbor))

    # if failed, visualize start and goal
    if enable_vis:
        visualize(maze, [start, end])
    return None


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    points = []
    dx = x1 - x0
    dy = y1 - y0
    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1
    dx = abs(dx)
    dy = abs(dy)
    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0
    D = 2 * dy - dx
    y = 0
    for x in range(dx + 1):
        px, py = x0 + x * xx + y * yx, y0 + x * xy + y * yy
        points.append((px, py))
        if D > 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy
    return points


def find_nearest_free(maze: List[List[int]], start: Tuple[int, int], pos: Tuple[int, int]) -> Tuple[int, int]:
    def is_valid(maze: List[List[int]], x: int, y: int) -> bool:
        if 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[x][y] == 0:
            return True
        return False

    line_points = bresenham_line(start[0], start[1], pos[0], pos[1])
    # reverse the line points so that the nearest free point is returned
    line_points.reverse()
    for point in line_points:
        if is_valid(maze, point[0], point[1]):
            return point
    return None  # No free position found along the line


def mark_connected_cells(occupancy_map, start):
    def neighbors(x, y):
        return [(x + i, y + j) for i in range(-1, 2) for j in range(-1, 2) if i != 0 or j != 0]

    def bfs(x, y):
        q = deque([(x, y)])
        connected = []
        visited = [[0 for _ in range(len(occupancy_map[0]))] for _ in range(len(occupancy_map))]
        while q:
            x, y = q.popleft()
            if 0 <= x < len(occupancy_map) and 0 <= y < len(occupancy_map[0]) and visited[x][y] == 0 and occupancy_map[x][y] == 0:
                visited[x][y] = 1
                connected.append((x, y))
                for nx, ny in neighbors(x, y):
                    q.append((nx, ny))

        return connected

    connected = bfs(start[0], start[1])
    marked_occ_map = [[1 for _ in range(len(occupancy_map[0]))] for _ in range(len(occupancy_map))]
    for x, y in connected:
        marked_occ_map[x][y] = 0

    return marked_occ_map


#### User interaction ####

def user_input(end_token: List[str]) -> str:
    """Talk with LLM"""
    print(">> User: ", end="")
    lines = []
    while True:
        line = input()
        # if this line is endwith one of the end token, stop
        for token in end_token:
            if line.endswith(token):
                content = line.split(token)[0]
                if content:
                    lines.append(content)
                return ("\n".join(lines)).strip()
        lines.append(line)


def print_type_indicator(agent_name="LLM"):
    # print typing indicator
    print(f"* {agent_name} is typing...", end="", flush=True)
    for i in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print()


@contextmanager
def action_verbose(action, verbose: bool, save: bool, **kwargs):
    if verbose:
        print("--------- Code start from here ---------")
        print(action)
        print("--------- Code end here ----------------")
        if save:
            # save the action file for further debug
            now = datetime.now()
            current_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
            log_path = kwargs.get("log_path", None)
            env_name = kwargs.get("env_name", None)
            if log_path is not None and env_name is not None:
                action.save(os.path.join(log_path, env_name, "program", f"{current_time_str}.txt"))
        # run the action
        print("========= Execution start from here ====")
    yield
    if verbose:
        print("--------- Code end here ----------------")


def reply_verbose(reply: Any, verbose: bool, save: bool, **kwargs):
    """Print or save the reply from the system"""
    if verbose:
        print("========= Reply from the sys ============")
        print(reply)
        print("=========================================")
    if save:
        # save the reply for further debug
        now = datetime.now()
        current_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
        log_path = kwargs.get("log_path", None)
        env_name = kwargs.get("env_name", None)
        if log_path is not None and env_name is not None:
            save_path = os.path.join(log_path, env_name, "reply")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # save the readable reply
            with open(os.path.join(save_path, f"{current_time_str}.txt"), "w") as f:
                if isinstance(reply, str):
                    f.write(reply)
                elif isinstance(reply, list):
                    f.write("\n".join(reply))
                elif isinstance(reply, dict):
                    f.write("\n".join([f"{k}: {v}" for k, v in reply.items()]))
            # save the pickle reply
            with open(os.path.join(save_path, f"{current_time_str}.pkl"), "wb") as f:
                pickle.dump(reply, f)


#### Visualize Utils ####

def draw_bbox(
    input_image: Union[Image.Image, np.ndarray],
    scores: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    queries: Union[List[str], Image.Image, np.ndarray],
    format="coco",
):
    """Draw bounding boxes on the input image"""
    if isinstance(input_image, np.ndarray):
        input_image_vis = input_image
    else:
        input_image_vis = np.array(input_image)
    if format == "coco":
        # size
        width, height = input_image_vis.shape[1], input_image_vis.shape[0]
        # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plt.imshow(input_image_vis, extent=(0, 1, 1, 0), origin="upper")
        # plt.set_axis_off()
        for score, box, label in zip(scores, boxes, labels):
            x1, y1, x2, y2 = box
            x1, x2 = x1 / width, x2 / width
            y1, y2 = y1 / height, y2 / height
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "r")
            if isinstance(queries, List):
                plt.text(
                    x1,
                    y2 + 0.015,
                    f"{queries[label]}: {score:1.2f}",
                    ha="left",
                    va="top",
                    color="red",
                    bbox={"facecolor": "white", "edgecolor": "red", "boxstyle": "square,pad=.3"},
                )
            elif isinstance(queries, np.ndarray) or isinstance(queries, Image.Image):
                if isinstance(queries, np.ndarray):
                    queries_vis = queries
                else:
                    queries_vis = np.array(queries)
                # draw query image
                query_img_size = 0.1
                plt.imshow(queries_vis, extent=(x1, x1 + query_img_size, y1, y1 - query_img_size), alpha=0.8, origin="upper")
                plt.text(
                    x1,
                    y2 + 0.015,
                    f"{score:1.2f}",
                    ha="left",
                    va="top",
                    color="red",
                    bbox={"facecolor": "white", "edgecolor": "red", "boxstyle": "square,pad=.3"},
                )
                plt.xlim([0, max(1, x1 + query_img_size)])
                plt.ylim([max(1, y2 + query_img_size), 0])
        plt.show()


def project_3d_points_to_2d(points_3d, extrinsic_matrix, intrinsic_matrix):
    """Project 3D points to 2D points in the image plane"""
    num_points = points_3d.shape[0]
    points_3d_homogeneous = np.hstack((points_3d, np.ones((num_points, 1))))
    points_camera_frame = np.dot(extrinsic_matrix, points_3d_homogeneous.T).T
    points_2d_homogeneous = np.dot(intrinsic_matrix, points_camera_frame[:, :3].T).T
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2][:, np.newaxis]
    return points_2d


def draw_occupancy_map(occupancy_map, vis_scale=1.0):
    """Draw the occupancy map"""
    # color for the occupied cells
    color = (0, 0, 255)  # Blue color for occupied cells
    image_vis = np.zeros((occupancy_map.shape[0], occupancy_map.shape[1], 3), dtype=np.uint8)
    image_vis[occupancy_map == 1] = color

    # resize the image
    image_vis = cv2.resize(image_vis, (image_vis.shape[1] * vis_scale, image_vis.shape[0] * vis_scale), interpolation=cv2.INTER_NEAREST)

    grid_color = (50, 50, 50)  # Gray color for grid
    for i in range(0, image_vis.shape[0], vis_scale):
        cv2.line(image_vis, (0, i), (image_vis.shape[1], i), grid_color, 1)
    for j in range(0, image_vis.shape[1], vis_scale):
        cv2.line(image_vis, (j, 0), (j, image_vis.shape[0]), grid_color, 1)

    return image_vis
