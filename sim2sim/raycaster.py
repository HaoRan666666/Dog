"""深度相机 Python 薄封装，桥接 raycaster_ext（vendor 自 mujoco_ray_caster）。"""

import numpy as np

import raycaster_ext


class DepthCamera:
    """包一层 raycaster_ext.RayCasterCamera，直接读取 MuJoCo model/data 指针。"""

    def __init__(
        self,
        model,
        data,
        cam_name="depth_cam",
        h_ray_num=64,
        v_ray_num=48,
        focal_length=24.0,
        horizontal_aperture=20.955,
        dis_range=(0.1, 2.0),
        debug_vis=False,
    ):
        self.h_ray_num = h_ray_num
        self.v_ray_num = v_ray_num
        self.debug_vis = debug_vis
        self._cam = raycaster_ext.RayCasterCamera(
            model_addr=model._address,
            data_addr=data._address,
            cam_name=cam_name,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            h_ray_num=h_ray_num,
            v_ray_num=v_ray_num,
            dis_min=dis_range[0],
            dis_max=dis_range[1],
        )
        if debug_vis:
            import cv2  # noqa: F401 仅在需要可视化时才要求装了 opencv

            self._cv2 = cv2
            cv2.namedWindow("depth_cam", cv2.WINDOW_NORMAL)

    def step(self) -> np.ndarray:
        """推进一次射线计算，返回 (v_ray_num, h_ray_num) 的 distance_to_image_plane 深度图（米）。"""
        self._cam.compute_distance()
        flat = self._cam.get_distance_to_image_plane_vec(False, True)
        image = flat.reshape(self.v_ray_num, self.h_ray_num)
        if self.debug_vis:
            self._visualize(image)
        return image

    def get_ray_geometry(self):
        """返回 (cam_pos:(3,), hit_points:(nray,3))，世界坐标系，未命中为 NaN。

        必须在 step() 之后调用（复用其 compute_distance() 结果，不重复触发射线计算）。
        """
        cam_pos = self._cam.get_cam_pos()
        hits = self._cam.get_hit_points_world()
        return cam_pos, hits

    def _visualize(self, image: np.ndarray) -> None:
        near, far = 0.1, 2.0
        clipped = np.clip(image, near, far)
        normalized = (clipped - near) / (far - near)
        vis = (normalized * 255).astype(np.uint8)
        vis = self._cv2.resize(vis, (self.h_ray_num * 6, self.v_ray_num * 6),
                                interpolation=self._cv2.INTER_NEAREST)
        self._cv2.imshow("depth_cam", vis)
        self._cv2.waitKey(1)
