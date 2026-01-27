import mujoco
import mujoco.viewer

# Load model
model = mujoco.MjModel.from_xml_path('/home/nl455/spider/spider/assets/robots/leap/right.xml')
data = mujoco.MjData(model)

# Launch viewer
mujoco.viewer.launch(model, data)
