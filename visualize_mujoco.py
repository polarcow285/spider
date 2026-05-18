import mujoco
import mujoco.viewer

# model = mujoco.MjModel.from_xml_path("C:\\Users\\mmoop\\Code\\spider\\example_datasets\\processed\\custom\\assets\\robots\\wuji\\right.xml")
# model = mujoco.MjModel.from_xml_path("C:\\Users\\mmoop\\Code\\spider\\example_datasets\\processed\\custom\\leap\\right\\screwdriver\\scene.xml")
# model = mujoco.MjModel.from_xml_path("C:\\Users\\mmoop\\Code\\spider\\example_datasets\\processed\\custom\\assets\\robots\\leap\\right.xml")

model = mujoco.MjModel.from_xml_path("C:\\Users\\mmoop\\Code\\spider\\example_datasets\\processed\\arctic\\wuji\\right\\scissors\\scene.xml")
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)
