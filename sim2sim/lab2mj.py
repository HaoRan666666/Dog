import mujoco 

m=mujoco.MjModel.from_xml_path("/home/xhr/BPX-master/mujoco/bpx.xml")
d=mujoco.MjData(m)

