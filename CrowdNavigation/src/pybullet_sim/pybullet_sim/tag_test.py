# aruco_double_sided_test.py
import os, math, time, pybullet as p, pybullet_data

# --------------------------------------------------------------------- #
# PARAMETERS (edit to taste)
# --------------------------------------------------------------------- #
TAG_PNG   = "aruco_rgb.png"      # 3-channel or 4-channel PNG
MESH_FILE = "aruco_plane.obj"    # your pre-made quad (UV 0-1)
SIZE      = 0.12                 # physical tag size [m]
THICK     = 0.001                # collision thickness [m]
GAP       = 0.0005               # visual gap for the back plate [m]
half      = SIZE / 2

# --------------------------------------------------------------------- #
# CONNECT
# --------------------------------------------------------------------- #
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# --------------------------------------------------------------------- #
# LOAD TEXTURE
# --------------------------------------------------------------------- #
tex_id = p.loadTexture(os.path.abspath(TAG_PNG))

# --------------------------------------------------------------------- #
# COLLISION SHAPE (thin box)
# --------------------------------------------------------------------- #
col_id = p.createCollisionShape(
    p.GEOM_BOX, halfExtents=[THICK / 2, half, half]
)

# --------------------------------------------------------------------- #
# VISUAL SHAPES
# --------------------------------------------------------------------- #
front_vis = p.createVisualShape(
    p.GEOM_MESH, fileName=MESH_FILE, meshScale=[SIZE, SIZE, 1]
)
back_vis  = p.createVisualShape(
    p.GEOM_MESH, fileName=MESH_FILE, meshScale=[SIZE, SIZE, 1]
)

# --------------------------------------------------------------------- #
# FRONT PLATE  (collision + texture)
# --------------------------------------------------------------------- #
tag_body = p.createMultiBody(
    baseMass                = 0,
    baseCollisionShapeIndex = col_id,
    baseVisualShapeIndex    = front_vis,
    basePosition            = [0, 0, half],        # sits on floor
    baseOrientation         = [0, 0, 0, 1]         # faces +X
)
p.changeVisualShape(tag_body, -1, textureUniqueId=tex_id)

# --------------------------------------------------------------------- #
# BACK PLATE  (visual-only, rotated 180 ° about Y, same texture)
# --------------------------------------------------------------------- #
back_body = p.createMultiBody(
    baseMass                = 0,
    baseCollisionShapeIndex = -1,                  # no physics
    baseVisualShapeIndex    = back_vis,
    basePosition            = [-GAP, 0, half],     # 0.5 mm behind front
    baseOrientation         = p.getQuaternionFromEuler([0, math.pi, 0])
)
p.changeVisualShape(back_body, -1, textureUniqueId=tex_id)

# --------------------------------------------------------------------- #
# LOOP  (Esc to quit)
# --------------------------------------------------------------------- #
print("Press Esc in the PyBullet window to exit")
while p.isConnected():
    p.stepSimulation()
    time.sleep(1 / 240)
    keys = p.getKeyboardEvents()
    if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
        break

p.disconnect()
