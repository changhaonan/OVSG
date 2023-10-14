from typing import List, Union, Any, Dict, Tuple
import numpy as np
import abc
from abc import ABC, abstractmethod
import cv2
from collections import deque
from enum import Enum
import copy
import open3d as o3d


class Direction(Enum):
    """Pre-defined Direction"""

    CTR = 0  # origin
    TOP = 1  # top
    BOT = 2  # bottom
    LFT = 3  # left
    RIT = 4  # right
    TOPLFT = 5  # top left
    TOPRIT = 6  # top right
    BOTLFT = 7  # bottom left
    BOTRIT = 8  # bottom right


# Helper functions
def draw_rectangle(event, x, y, flags, params, img, rectangles):
    """Draw a rectangle on the image"""
    if event == cv2.EVENT_LBUTTONDOWN:
        rectangles.append([x, y])
    elif event == cv2.EVENT_LBUTTONUP:
        rectangles[-1].extend([x, y])
        cv2.rectangle(
            img,
            (rectangles[-1][0], rectangles[-1][1]),
            (rectangles[-1][2], rectangles[-1][3]),
            (255, 0, 255),
            2,
        )


def draw_occupancy_map(occupancy_map, vis_scale=1.0):
    """Draw the occupancy map"""
    # color for the occupied cells
    color = (0, 0, 255)  # Blue color for occupied cells
    image_vis = np.zeros((occupancy_map.shape[0], occupancy_map.shape[1], 3), dtype=np.uint8)
    image_vis[occupancy_map == 1] = color
    # flip the image because occupancy map is saved as (x+, y+), but image is (y-, x+)
    image_vis = np.flipud(image_vis.transpose(1, 0, 2))
    # resize the image
    image_vis = cv2.resize(
        image_vis,
        (image_vis.shape[1] * vis_scale, image_vis.shape[0] * vis_scale),
        interpolation=cv2.INTER_NEAREST,
    )

    grid_color = (50, 50, 50)  # Gray color for grid
    for i in range(0, image_vis.shape[0], vis_scale):
        cv2.line(image_vis, (0, i), (image_vis.shape[1], i), grid_color, 1)
    for j in range(0, image_vis.shape[1], vis_scale):
        cv2.line(image_vis, (j, 0), (j, image_vis.shape[0]), grid_color, 1)

    return image_vis


def draw_color_map(color_map, vis_scale=1.0):
    """Draw the color map"""
    image_vis = np.copy(color_map)
    # resize the image
    # flip the image because occupancy map is saved as (x+, y+), but image is (y-, x+)
    image_vis = np.flipud(image_vis.transpose(1, 0, 2))
    # resize the image
    image_vis = cv2.resize(
        image_vis,
        (image_vis.shape[1] * vis_scale, image_vis.shape[0] * vis_scale),
        interpolation=cv2.INTER_NEAREST,
    )

    grid_color = (0, 0, 0)  # Gray color for grid
    for i in range(0, image_vis.shape[0], vis_scale):
        cv2.line(image_vis, (0, i), (image_vis.shape[1], i), grid_color, 1)
    for j in range(0, image_vis.shape[1], vis_scale):
        cv2.line(image_vis, (j, 0), (j, image_vis.shape[0]), grid_color, 1)

    return image_vis


class Region(ABC):
    """Region in physical space"""

    grid_size: Union[List[int], None]  # size of the grid
    resolution: float  # resolution of the grid
    region_map: np.ndarray  # region map in xy plane
    world2region: np.ndarray  # tf from world to region, region origin is the position of (0, 0)
    connected: bool  # if region is connected
    region_pcd: Union[o3d.geometry.PointCloud, None]  # point cloud of the region

    def __init__(
        self,
        resolution: float,
        grid_size: Union[List[int], None] = None,
        world2region: np.ndarray = np.eye(4, dtype=np.float32),
        name: str = "region",
        **kwargs,
    ):
        """Initialize a region in physical space"""
        self.grid_size = grid_size
        self.resolution = resolution
        self.name = name
        # tf from world to region
        self.world2region = world2region
        # region map
        self.region_map = None  # reflecting occupancy and depth
        self.color_map = None  # top-down image
        self.connected = True
        # pcd
        self.region_pcd = None

    @abstractmethod
    def __contains__(self, item):
        """Check if a point is inside the region"""
        raise NotImplementedError

    @abstractmethod
    def iou(self, other: "Region") -> float:
        """Compute the intersection over union between two regions"""
        raise NotImplementedError

    @abstractmethod
    def visualize(self, **kwargs):
        """Visualize the region"""
        raise NotImplementedError

    @abstractmethod
    def grid_points_3d(self, **kwargs):
        """Get the grid points in the region"""
        raise NotImplementedError

    def save(self, path: str):
        """Save the region"""
        np.savez(
            path,
            grid_size=self.grid_size,
            resolution=self.resolution,
            region_map=self.region_map,
            color_map=self.color_map,
            world2region=self.world2region,
            name=self.name,
            connected=self.connected,
        )

    def load(self, path: str):
        """Load the region"""
        data = np.load(path)
        self.grid_size = data["grid_size"]
        self.resolution = data["resolution"]
        self.region_map = data["region_map"]
        self.color_map = data["color_map"]
        self.world2region = data["world2region"]
        self.name = data["name"]
        self.connected = data["connected"]

    def check_connected(self):
        """Check if the region is mono connected"""
        assert self.region_map is not None
        # function to perform BFS

        def bfs(grid, i, j, visited):
            n_rows, n_cols = grid.shape
            queue = deque([(i, j)])
            visited[i, j] = True
            while queue:
                i, j = queue.popleft()
                if i > 0 and grid[i - 1, j] and not visited[i - 1, j]:
                    queue.append((i - 1, j))
                    visited[i - 1, j] = True
                if i < n_rows - 1 and grid[i + 1, j] and not visited[i + 1, j]:
                    queue.append((i + 1, j))
                    visited[i + 1, j] = True
                if j > 0 and grid[i, j - 1] and not visited[i, j - 1]:
                    queue.append((i, j - 1))
                    visited[i, j - 1] = True
                if j < n_cols - 1 and grid[i, j + 1] and not visited[i, j + 1]:
                    queue.append((i, j + 1))
                    visited[i, j + 1] = True

        # check if the grid is connected
        n_rows, n_cols = self.region_map.shape
        visited = np.zeros((n_rows, n_cols), dtype=bool)
        visited[self.region_map == 0] = True  # All free occupany are visited
        for i in range(n_rows):
            for j in range(n_cols):
                if self.region_map[i, j] and not visited[i, j]:
                    bfs(self.region_map, i, j, visited)
                    if not visited.all():
                        print("The region map is not connected.")
                        self.connected = False
                        return self.connected

        print("The region map is connected.")
        self.connected = True
        return self.connected


class Region2D(Region):
    """2.5D region in physical space"""

    def __init__(
        self,
        resolution: float,
        grid_size: Union[List[int], None] = None,
        world2region: np.ndarray = np.eye(4, dtype=np.float32),
        name: str = "region",
        **kwargs,
    ):
        super().__init__(
            resolution=resolution,
            grid_size=grid_size,
            world2region=world2region,
            name=name,
            **kwargs,
        )

    def __contains__(self, item):
        if isinstance(item, np.ndarray) or isinstance(item, List) or isinstance(item, str):
            if isinstance(item, List):
                pos = np.array(item)
            elif isinstance(item, str):
                try:
                    pos = np.array(
                        [
                            float(item.split(",")[0]),
                            float(item.split(",")[1]),
                            float(item.split(",")[2]),
                        ]
                    )
                except ValueError as exc:
                    raise ValueError("The input string should be in the format of x,y,z.") from exc
            else:
                pos = item
            if pos.shape[0] >= 3:
                pos = pos[:3]
                # check if the point is inside the region
                point_region = self.world2region @ np.append(pos, 1)
                point_region_grid = [
                    int(point_region[0] / self.resolution),
                    int(point_region[1] / self.resolution),
                ]
                if point_region_grid[0] < 0 or point_region_grid[0] >= self.grid_size[0]:
                    return False
                if point_region_grid[1] < 0 or point_region_grid[1] >= self.grid_size[1]:
                    return False
                if self.region_map[point_region_grid[0], point_region_grid[1]] == 1:
                    return True
                else:
                    return False
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def iou(self, other: "Region") -> float:
        """Compute the intersection over union between two regions"""
        return 0.0

    def visualize(self, **kwargs):
        """Visualize the region"""
        title = kwargs.get("title", self.name)
        # check
        # image_vis = draw_occupancy_map(self.region_map, vis_scale=5)
        image_vis = draw_color_map(self.color_map, vis_scale=5)
        cv2.imshow(title, image_vis)
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()

    def grid_points_3d(self, **kwargs):
        """Get the 3d grid points in the region"""
        # get the grid points
        idx = np.indices(self.grid_size)
        region_map_3d = np.broadcast_to(self.region_map[..., np.newaxis], self.grid_size)

        mask = region_map_3d == 1
        grid_points = np.column_stack((idx[0][mask], idx[1][mask], idx[2][mask]))

        # transform to 3d
        grid_points = grid_points * self.resolution
        grid_points = np.concatenate([grid_points, np.ones((grid_points.shape[0], 1))], axis=1)
        grid_points = np.matmul(grid_points, np.linalg.inv(self.world2region).T)

        return grid_points[:, :3]

    def bbox(self, **kwargs):
        """Get the o3d bounding box of the region"""
        color = kwargs.get("color", (0.0, 1.0, 0.0))
        grid_points = self.grid_points_3d()
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(grid_points)
        )
        bbox.color = color
        return bbox

    # create interface
    def create(self, method: str, **kwargs):
        """Create region from different methods"""
        if method == "image":
            self.create_from_image(**kwargs)
        elif method == "param":
            self.create_from_param(**kwargs)
        else:
            raise NotImplementedError

    def create_from_param(
        self, width: int, height: int, cam2world: np.ndarray, intrinsic: np.ndarray, **kwargs
    ):
        """Create region from camera parameters"""
        # start building the region
        depth_est = cam2world[2, 3]
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        # update grid size
        region_width = width / fx * depth_est
        region_height = height / fy * depth_est
        if self.grid_size is None:
            self.grid_size = [
                int(region_width / self.resolution),
                int(region_height / self.resolution),
                100,
            ]
        else:
            self.grid_size = [
                int(region_width / self.resolution),
                int(region_height / self.resolution),
                self.grid_size[2],
            ]
        # make sure the grid size is even
        if self.grid_size[0] % 2 != 0:
            self.grid_size[0] += 1
        if self.grid_size[1] % 2 != 0:
            self.grid_size[1] += 1

        # update tf_world2region
        region_origin = np.array(
            [(0 - cx) / fx * depth_est, (height - cy) / fy * depth_est, depth_est]
        )
        region_x_axis = np.array([1.0, 0.0, 0.0])
        region_y_axis = np.array([0.0, -1.0, 0.0])
        region_z_axis = np.array([0.0, 0.0, -1.0])
        region2camera = np.eye(4, dtype=np.float32)
        region2camera[:3, 0] = region_x_axis
        region2camera[:3, 1] = region_y_axis
        region2camera[:3, 2] = region_z_axis
        region2camera[:3, 3] = region_origin
        self.world2region = np.linalg.inv(cam2world @ region2camera)

        # capture the color
        x_grid = np.array(range(self.grid_size[0] + 1)) * self.resolution
        y_grid = np.array(range(self.grid_size[1] + 1)) * self.resolution
        grid_points_rgn = np.zeros((self.grid_size[0] + 1, self.grid_size[1] + 1, 4))
        grid_points_rgn[:, :, 0] = x_grid.reshape(-1, 1)
        grid_points_rgn[:, :, 1] = y_grid.reshape(1, -1)
        grid_points_rgn[:, :, 2] = 0
        grid_points_rgn[:, :, 3] = 1.0
        grid_points_rgn = grid_points_rgn.reshape(-1, 4)
        grid_points_cam = region2camera @ grid_points_rgn.T
        # map grid to image space
        grid_points_img = np.zeros((grid_points_cam.shape[1], 3))
        grid_points_img[:, 0] = grid_points_cam[0, :] / grid_points_cam[2, :] * fx + cx
        grid_points_img[:, 1] = grid_points_cam[1, :] / grid_points_cam[2, :] * fy + cy
        grid_points_img[:, 2] = grid_points_cam[2, :]
        # get the color
        self.color_map = np.zeros((self.grid_size[0], self.grid_size[1], 3))
        image = kwargs.get("image", np.ones((height, width, 3)) * 255.0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                x, y = grid_points_img[i * (self.grid_size[1] + 1) + j, :2]
                x = int(x)
                y = int(y)
                if x < 0 or x >= width or y < 0 or y >= height:
                    continue
                self.color_map[i, j, :] = image[y, x, :] / 255.0

        # update occupancy
        self.region_map = np.zeros((self.grid_size[0], self.grid_size[1]))
        resolution_pixel_x = float(width) / float(self.grid_size[0])
        resolution_pixel_y = float(height) / float(self.grid_size[1])
        # convert the coordinates of the marked rectangles to occupancy values
        rectangles = kwargs.get("rectangles", None)
        if rectangles is not None:
            for rect in rectangles:
                x1, y1, x2, y2 = rect
                y1 = height - y1
                y2 = height - y2
                self.region_map[
                    int(x1 / resolution_pixel_x): int(x2 / resolution_pixel_x),
                    int(y2 / resolution_pixel_y): int(y1 / resolution_pixel_y),
                ] = 1
            # update
            # find min/max x and y values of occupied cells
            occupied_indices = np.where(self.region_map == 1)
            min_x = np.min(occupied_indices[0])
            max_x = np.max(occupied_indices[0])
            min_y = np.min(occupied_indices[1])
            max_y = np.max(occupied_indices[1])
            self.crop_to(bbox=(min_x, max_x, min_y, max_y))
        else:
            self.region_map = np.ones((self.grid_size[0], self.grid_size[1]))

        enable_vis = kwargs.get("enable_vis", False)
        if enable_vis:
            self.visualize()

    def create_from_image(
        self,
        color_image: np.ndarray,
        cam2world: np.ndarray,
        intrinsic: np.ndarray,
        top_down: bool = True,
        depth_image: Union[np.ndarray, None] = None,
        require_draw: bool = False,
        **kwargs,
    ):
        """Create region from camera image"""
        if not top_down:
            raise NotImplementedError

        # copy the image
        img_vis = np.copy(color_image)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        # bind callback function
        if require_draw:
            # create window
            cv2.namedWindow(self.name)
            cv2.imshow(self.name, img_vis)
            cv2.waitKey(1)
            if cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("Window is not created")

            rectangles = []
            cv2.setMouseCallback(
                self.name,
                lambda event, x, y, flags, params: draw_rectangle(
                    event, x, y, flags, params, img=img_vis, rectangles=rectangles
                ),
            )
            # Wait for the user to mark occupancy
            while True:
                cv2.imshow(self.name, img_vis)
                key = cv2.waitKey(1) & 0xFF
                # If the 'Enter' key is pressed, exit the loop
                if key == 13:
                    break
            # close the window
            cv2.destroyAllWindows()
        else:
            rectangles = [[0, 0, img_vis.shape[1], img_vis.shape[0]]]
        # build region_pcd
        if depth_image is not None:
            depth_scale = kwargs.get("depth_scale", 1.0)
            depth_trunc = kwargs.get("depth_trunc", 1.0)
            # create pcd from rgbd using o3d
            color = o3d.geometry.Image(color_image)
            depth = o3d.geometry.Image(depth_image)
            # create Open3D RGBD image from color and depth images
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=depth_scale,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
            # create Open3D point cloud from RGBD image
            intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=color_image.shape[1],
                height=color_image.shape[0],
                fx=intrinsic[0, 0],
                fy=intrinsic[1, 1],
                cx=intrinsic[0, 2],
                cy=intrinsic[1, 2],
            )
            self.region_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
            # transform to world frame
            self.region_pcd.transform(cam2world)

        print("debug", cam2world)
        # start building the region
        self.create_from_param(
            width=img_vis.shape[1],
            height=img_vis.shape[0],
            cam2world=cam2world,
            intrinsic=intrinsic,
            rectangles=rectangles,
            image=color_image,
            **kwargs,
        )

    def crop_to(self, bbox: Tuple[int]):
        """Adapt the size of region"""
        min_x, max_x, min_y, max_y = bbox
        if max_x < min_x:
            max_x = min_x - 1
        if max_y < min_y:
            max_y = min_y - 1
        # update size
        self.grid_size = (max_x - min_x + 1, max_y - min_y + 1, self.grid_size[2])
        self.region_map = self.region_map[min_x: max_x + 1, min_y: max_y + 1]
        self.color_map = self.color_map[min_x: max_x + 1, min_y: max_y + 1]
        self.world2region[:3, 3] -= np.array(
            [min_x * self.resolution, min_y * self.resolution, 0.0]
        )
        return min_x, max_x, min_y, max_y

    def sub_region(self, direction: Direction, origin: Union[np.ndarray, None] = None) -> "QRegion":
        """sub-region corresponding to pre-defined action"""
        if origin is None and direction is None:
            return self
        assert self.connected, "Region is not connected"
        if origin is None:
            origin = np.array(
                [self.grid_size[0] / 2.0, self.grid_size[1] / 2.0, 0], dtype=np.float32
            )
        if direction == Direction.CTR:
            min_x = max(0, int(origin[0] - self.grid_size[0] * 0.25))
            max_x = min(self.grid_size[0] - 1, int(origin[0] + self.grid_size[0] * 0.25))
            min_y = max(0, int(origin[1] - self.grid_size[1] * 0.25))
            max_y = min(self.grid_size[1] - 1, int(origin[1] + self.grid_size[1] * 0.25))
            region_copy = copy.deepcopy(self)
            region_copy.name = f"{self.name}-CTR"
            region_copy.crop_to(bbox=(min_x, max_x, min_y, max_y))
            return region_copy
        elif direction == Direction.TOP:
            min_x = 0
            max_x = self.grid_size[0] - 1
            min_y = int(origin[1])
            max_y = self.grid_size[1] - 1
            region_copy = copy.deepcopy(self)
            region_copy.name = f"{self.name}-TOP"
            region_copy.crop_to(bbox=(min_x, max_x, min_y, max_y))
            return region_copy
        elif direction == Direction.BOT:
            min_x = 0
            max_x = self.grid_size[0] - 1
            min_y = 0
            max_y = int(origin[1])
            region_copy = copy.deepcopy(self)
            region_copy.name = f"{self.name}-BOT"
            region_copy.crop_to(bbox=(min_x, max_x, min_y, max_y))
            return region_copy
        elif direction == Direction.LFT:
            min_x = 0
            max_x = int(origin[0])
            min_y = 0
            max_y = self.grid_size[1] - 1
            region_copy = copy.deepcopy(self)
            region_copy.name = f"{self.name}-LFT"
            region_copy.crop_to(bbox=(min_x, max_x, min_y, max_y))
            return region_copy
        elif direction == Direction.RIT:
            min_x = int(origin[0])
            max_x = self.grid_size[0] - 1
            min_y = 0
            max_y = self.grid_size[1] - 1
            region_copy = copy.deepcopy(self)
            region_copy.name = f"{self.name}-RIT"
            region_copy.crop_to(bbox=(min_x, max_x, min_y, max_y))
            return region_copy
        elif direction == Direction.TOPLFT:
            min_x = 0
            max_x = int(origin[0])
            min_y = int(origin[1])
            max_y = self.grid_size[1] - 1
            region_copy = copy.deepcopy(self)
            region_copy.name = f"{self.name}-TOPLFT"
            region_copy.crop_to(bbox=(min_x, max_x, min_y, max_y))
            return region_copy
        elif direction == Direction.TOPRIT:
            min_x = int(origin[0])
            max_x = self.grid_size[0] - 1
            min_y = int(origin[1])
            max_y = self.grid_size[1] - 1
            region_copy = copy.deepcopy(self)
            region_copy.name = f"{self.name}-TOPRIT"
            region_copy.crop_to(bbox=(min_x, max_x, min_y, max_y))
            return region_copy
        elif direction == Direction.BOTLFT:
            min_x = 0
            max_x = int(origin[0])
            min_y = 0
            max_y = int(origin[1])
            region_copy = copy.deepcopy(self)
            region_copy.name = f"{self.name}-BOTLFT"
            region_copy.crop_to(bbox=(min_x, max_x, min_y, max_y))
            return region_copy
        elif direction == Direction.BOTRIT:
            min_x = int(origin[0])
            max_x = self.grid_size[0] - 1
            min_y = 0
            max_y = int(origin[1])
            region_copy = copy.deepcopy(self)
            region_copy.name = f"{self.name}-BOTRIT"
            region_copy.crop_to(bbox=(min_x, max_x, min_y, max_y))
            return region_copy
        else:
            raise ValueError("direction not support.")


if __name__ == "__main__":
    import os
    import open3d as o3d

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    scene_name = "scene0011_00"
    # Test region
    test_img_file = os.path.join(project_dir, "data/notion", f"{scene_name}.png")
    region = Region2D(grid_size=None, resolution=0.1, name="living room")
    test_image = cv2.imread(test_img_file)
    camera_info = np.load(test_img_file.replace("png", "npz"))
    region.create_from_param(
        width=test_image.shape[1],
        height=test_image.shape[0],
        cam2world=camera_info["extrinsic"],
        intrinsic=camera_info["intrinsic"],
    )
    o3d.visualization.draw_geometries([region.bbox(color=(0, 0, 1))])
