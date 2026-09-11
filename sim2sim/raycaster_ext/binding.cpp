// 把 mujoco_ray_caster 的 RayCasterCamera 包一层 pybind11，接进现有 Python
// sim2sim 主循环。只暴露部署需要的最小接口：构造 + compute_distance +
// get_distance_to_image_plane_vec + enable_sensor，不绑定 noise/绘制相关接口
// (绘制需要 mjvScene*，部署用的 launch_passive viewer 不方便桥接内部 scene 指针；
// noise 部署时本来就不加)。
//
// mjModel*/mjData* 通过 Python 侧 `model._address`/`data._address` 拿到的原始
// 指针地址传入，再 reinterpret_cast 回指针 —— 前提是本扩展必须和 Python 里
// `import mujoco` 用的是同一份 libmujoco.so（见 CMakeLists.txt 里的路径固定），
// 否则 mjModel/mjData 的内存布局可能对不上。

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "RayCasterCamera.h"

namespace py = pybind11;

namespace {

RayCasterCameraCfg make_cfg(std::uintptr_t model_addr, std::uintptr_t data_addr,
                             const std::string &cam_name, mjtNum focal_length,
                             mjtNum horizontal_aperture,
                             mjtNum vertical_aperture, int h_ray_num,
                             int v_ray_num, mjtNum dis_min, mjtNum dis_max,
                             bool is_detect_parentbody, mjtNum baseline,
                             mjtNum loss_angle, mjtNum min_energy) {
  RayCasterCameraCfg cfg;
  cfg.m = reinterpret_cast<const mjModel *>(model_addr);
  cfg.d = reinterpret_cast<mjData *>(data_addr);
  cfg.cam_name = cam_name;
  cfg.focal_length = focal_length;
  cfg.horizontal_aperture = horizontal_aperture;
  cfg.vertical_aperture = vertical_aperture;
  cfg.h_ray_num = h_ray_num;
  cfg.v_ray_num = v_ray_num;
  cfg.dis_range = {dis_min, dis_max};
  cfg.is_detect_parentbody = is_detect_parentbody;
  cfg.baseline = baseline;
  cfg.loss_angle = loss_angle;
  cfg.min_energy = min_energy;
  return cfg;
}

py::array_t<double> get_distance_to_image_plane_vec(RayCasterCamera &self,
                                                     bool is_noise,
                                                     bool is_inf_max) {
  std::vector<double> data =
      self.get_distance_to_image_plane_vec(is_noise, is_inf_max);
  // 注意：py::array_t<double>(count) 这个单参数构造不会分配正确 stride 的
  // 缓冲区(实测 strides=(0,)，导致所有元素都读到同一个地址)。显式传 shape +
  // 源指针 走拷贝构造，pybind11 才会分配一段连续内存并算对 stride。
  std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(data.size())};
  return py::array_t<double>(shape, data.data());
}

} // namespace

PYBIND11_MODULE(raycaster_ext, m) {
  m.doc() = "mujoco_ray_caster RayCasterCamera 的最小 pybind11 绑定";

  py::class_<RayCasterCamera>(m, "RayCasterCamera")
      .def(py::init([](std::uintptr_t model_addr, std::uintptr_t data_addr,
                        const std::string &cam_name, mjtNum focal_length,
                        mjtNum horizontal_aperture, mjtNum vertical_aperture,
                        int h_ray_num, int v_ray_num, mjtNum dis_min,
                        mjtNum dis_max, bool is_detect_parentbody,
                        mjtNum baseline, mjtNum loss_angle,
                        mjtNum min_energy) {
             RayCasterCameraCfg cfg = make_cfg(
                 model_addr, data_addr, cam_name, focal_length,
                 horizontal_aperture, vertical_aperture, h_ray_num, v_ray_num,
                 dis_min, dis_max, is_detect_parentbody, baseline, loss_angle,
                 min_energy);
             return new RayCasterCamera(cfg);
           }),
           py::arg("model_addr"), py::arg("data_addr"), py::arg("cam_name"),
           py::arg("focal_length") = 24.0,
           py::arg("horizontal_aperture") = 20.955,
           py::arg("vertical_aperture") = 0.0, py::arg("h_ray_num") = 64,
           py::arg("v_ray_num") = 48, py::arg("dis_min") = 0.1,
           py::arg("dis_max") = 2.0, py::arg("is_detect_parentbody") = false,
           py::arg("baseline") = 0.0, py::arg("loss_angle") = 0.0,
           py::arg("min_energy") = 0.0)
      .def("compute_distance", &RayCasterCamera::compute_distance)
      .def("enable_sensor", &RayCasterCamera::enable_sensor, py::arg("enable"))
      .def("get_distance_to_image_plane_vec", &get_distance_to_image_plane_vec,
           py::arg("is_noise") = false, py::arg("is_inf_max") = true)
      .def_readonly("h_ray_num", &RayCasterCamera::h_ray_num)
      .def_readonly("v_ray_num", &RayCasterCamera::v_ray_num)
      .def_readonly("nray", &RayCasterCamera::nray);
}
