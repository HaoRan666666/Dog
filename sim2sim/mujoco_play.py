import mujoco
import mujoco.viewer
import os
import time

# 加载 BPX MuJoCo 模型（基于脚本自身路径，不受 cwd 影响）
xml_path = os.path.join(os.path.dirname(__file__),
                        "../source/Dog/Dog/assets/BPX/mujoco/bpx.xml")
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(m, d)
        viewer.sync()
        # 保持实时仿真
        time_until_next = m.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)
