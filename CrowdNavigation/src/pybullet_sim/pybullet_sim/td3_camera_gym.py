import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import random
import math
import os
import time
from astar import astar
import cv2
import matplotlib.pyplot as plt



# ----- TUNABLE PARAMETERS FOR ENVIORMENT (not hyperparams) -----
ENV_BOUNDARY = 1.5          # Robot operates within -2..2 in x,y
MIN_DISTANCE = ENV_BOUNDARY * 0.8 # Spawn the robot and goal this distance apart
EP_LENGTH = 1000           # Max steps per episode
PHYSICS_DT = 1/240          # integrator step for simulation engine
CTRL_DT    = 1/25           # 25 Hz RL loop
SUB_STEPS  = int(round(CTRL_DT / PHYSICS_DT))   # = 12
WHEEL_BASE = 0.14           # Distance between left & right wheels
MAX_LINEAR_SPEED = 0.3       # m/s
MAX_ANGULAR_SPEED = 1.5     # rad/s
GOAL_REACHED_DIST = 0.2     # Robot is "at" goal if closer than this

NUM_MOVING_OBSTACLES = 10
OBSTACLE_SPEED = 0.1


WP_DISTANCE = 0.8
RADIUS_COLLISION = 0.09
CLUTTER_PROB = 0.15

# LiDAR specs
NUM_READINGS = 60           #use 40 for old model   # 360° / 0.12° = 3000 <- LIDAR measurements ON REAL WORLD, 180° / 3° = 60 <- SIM  <-- demo model uses 64
MAX_LIDAR_RANGE = 18          # 3  # Up to ~12 m for black objects , 18 is max range of LIDAR on slam s2l lidar
MIN_LIDAR_RANGE = 0.03            # Minimum measurable distance
LIDAR_HEIGHT_OFFSET = 0.25         # Slightly above ground/robot’s base
EXP_DECAY_METERS = 1  # Exponential decay for LiDAR readings, to weight by distance to goal

# Reward & penalty weights        #-0.2, 500, 10, 3, 10, 1
REWARD_GOAL_BONUS = 800
REWARD_WP = 200
REWARD_DTG_POSITIVE = 1.2    # Reward for reducing distance to goal, after * dist_improve number becomes very small
REWARD_HTG_POSITIVE = 1    # Reward for facing goal
REWARD_ACTION_HIGH = 1  # Reward for forward & near-zero rotation
REWARD_ACTION_MED = 0.5      # Reward for forward & rotating
ASTAR_GAIN = 1
PENALTY_COLLISION = -800
PENALTY_NEAR_COLLISION = -10
PENALTY_TURN = 0
PENALTY_TIME = -0.5

REWARD_IN_VIEW = 0.5  # Reward for being in view of the goal
REWARD_PROGRESS = 20 / CTRL_DT  # Reward for making progress towards the goal, scaled by control time step
PENALTY_ALIGN = -1  # Penalty for not facing the goal
REWARD_SMOOTH = 0.5 #/ CTRL_DT  # Reward for smooth movement
PENALTY_STEP = -0.01  # Penalty for taking a step

CAM_RES        = (512, 512)          # width, height   (keep square for CNNs)
CAM_FOV        = 70               # degrees
CAM_NEAR, CAM_FAR = 0.05, 5.0
CAM_OFFSET_L   = 0.05             # 10 cm in front of base
CAM_OFFSET_U   = 0.10             # 12 cm above base
DIST_REF_SIM = 4.0  # Reference distance for normalizing ArUco tag distance, how far the sim cam can pick up the tag

spawn_walls = True


class CrowdAvoidanceEnv(gym.Env):
    """Gym environment for a robot navigating a PyBullet scene."""

    metadata = {'render.modes': ['human']}

    def __init__(self, use_gui=False, preset_grid=None, preset_start=None, preset_goal=None):
        super().__init__()
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setTimeStep(PHYSICS_DT)
        p.setPhysicsEngineParameter(numSubSteps=5)

        self.past_observations = np.zeros((5, NUM_READINGS))  # Store last 5 LiDAR frames (framestacking)
        self.raw_lidar_scan = np.zeros(NUM_READINGS)  # Store current LiDAR scan

        w, h = CAM_RES
        fx = fy = (w / 2) / math.tan(math.radians(CAM_FOV) / 2)
        cx, cy = w / 2, h / 2
        self.camera_K = np.array([[fx,   0, cx],
                                [ 0,  fy, cy],
                                [ 0,   0,  1]], dtype=np.float32)

        # ArUco dictionary & detector
        self.aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_param = cv2.aruco.DetectorParameters()
        self.TAG_SIZE_M       = 0.12   # metres (edge length)

        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1,  1]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.concatenate((
                np.array([0, -1, 0, -1, -1], dtype=np.float32), # min goal distance, angle, tag flag, linear speed, angular speed
                np.full(NUM_READINGS, 0, dtype=np.float32) # min lidar readings
            )),
            high=np.concatenate((
                np.array([1, 1, 1, 1, 1], dtype=np.float32), # max goal distance, angle, tag flag, linear speed, angular speed
                np.full(NUM_READINGS, 1, dtype=np.float32) # max lidar readings
            )),
            dtype=np.float32
        )

        self.robot = None
        self.goal_position = [0, 0, 0.05]
        self.max_steps = EP_LENGTH
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.collision_happened = False

        self.current_wp = None
        self.num_steps_since_wp = 0
        self.wp_reached = False
        self.prev_wp_dist = None

        self.grid_size = 10  # Create 20x20 grid
        self.cell_size = (ENV_BOUNDARY * 2) / self.grid_size
        self.grid_map = None
        self.astar_path = None
        self.plane_id = None


        # view expert trajectories
        self.preset_grid = preset_grid
        self.preset_start = preset_start
        self.preset_goal = preset_goal

        self.act_lin_speed = 0.0
        self.act_ang_speed = 0.0
        self.previous_lin_speed = 0.0
        self.previous_ang_speed = 0.0
        self.tau_lin = 0.2
        self.tau_ang = 0.2
        self.distance = 0.0
        self.bearing = 0.0
        self.flag = 0.0

        self.reset()

    def world_to_grid(self, x, y):
        gx = int((x + ENV_BOUNDARY) / self.cell_size)
        gy = int((y + ENV_BOUNDARY) / self.cell_size)
        return (gy, gx)

    def grid_to_world(self, gx, gy):
        x = gx * self.cell_size - ENV_BOUNDARY + self.cell_size / 2
        y = gy * self.cell_size - ENV_BOUNDARY + self.cell_size / 2
        return x, y
    

    def _spawn_aruco(
            self,
            xy,                       # (x, y) centre on the floor
            tag_png="aruco_rgb.png",  # RGB or RGBA file already on disk
            mesh_file="aruco_plane.obj",  # 1×1 quad, UV 0-1, normal +X
            size=0.12,                # edge length in metres
            thickness=0.12,          # collision thickness
            gap=0.0005                # rear plate offset to avoid z-fight
            
    ):
            """
            Creates:
            • front plate: collision + texture  (normal +X)
            • back  plate: visual-only, rotated 180° about Y (normal –X)
            Returns the bodyUniqueId of the *front* (collision) plate so you can
            keep a handle (e.g. self.goal_marker = ...).
            """

            (rx, ry, _), _ = p.getBasePositionAndOrientation(self.robot)
            # tag pose
            (tx, ty) = xy

            # yaw that makes +X axis of plate point at robot
            yaw = math.atan2(ry - ty, rx - tx)
            
            front_ori = p.getQuaternionFromEuler([0,  math.pi / 2, yaw])
            back_ori  = p.getQuaternionFromEuler([0,  math.pi / 2, yaw + math.pi])


            half = size / 2
            base_z = half + 0.05
            tx_id = p.loadTexture(os.path.abspath(tag_png))

            coll_id = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[thickness / 2, half, half]
            )

            vis_id = p.createVisualShape(
                p.GEOM_MESH, fileName=mesh_file, meshScale=[size, size, 1]
            )


            fx, fy = xy
            front_body = p.createMultiBody(
                baseMass                = 0,
                baseCollisionShapeIndex = coll_id,
                baseVisualShapeIndex    = vis_id,
                basePosition            = [fx, fy, base_z],
                baseOrientation         = front_ori
            )
            p.changeVisualShape(front_body, -1, textureUniqueId=tx_id)

     
            p.createMultiBody(
                baseMass                = 0,
                baseCollisionShapeIndex = -1,     # no physics
                baseVisualShapeIndex    = vis_id,
                basePosition            = [fx - gap * math.cos(yaw),
                                        fy - gap * math.sin(yaw),
                                        base_z],
                baseOrientation         = back_ori
            )
            p.changeVisualShape(front_body, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)


            return front_body

        


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Required for Gym compatibility
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        if hasattr(self, "debug_lines"):
            for line_id in self.debug_lines:
                p.removeUserDebugItem(line_id)

        self.previous_linear_speed = 0.0
        self.previous_angular_speed = 0.0
        self.previous_goal_distance = None
        self.previous_goal_angle = None
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_length = 0

        self.current_wp = None
        self.num_steps_since_wp = 0
        self.wp_reached = False
        self.prev_wp_dist = None
        self.collision_happened = False

        self.lin_speed = 0
        self.ang_speed = 0  
        self.act_lin_speed = 0.0
        self.act_ang_speed = 0.0
        self.tau_lin = float(self.np_random.uniform(0.18, 0.32))
        self.tau_ang = float(self.np_random.uniform(0.15, 0.30))

        wall_half_extents = [self.cell_size * 0.5] * 3

        # Use preset grid, start, goal if provided
        if False: #self.preset_grid is not None and self.preset_start is not None and self.preset_goal is not None:
            self.grid_map = self.preset_grid.copy()
            self.astar_path = astar(self.grid_map, self.world_to_grid(*self.preset_start), self.world_to_grid(*self.preset_goal))
            start_x, start_y = self.preset_start
            goal_x, goal_y = self.preset_goal
        else:
            # Try generating a valid grid + A* path before touching the simulation
            while True:
                grid_map = np.zeros((self.grid_size, self.grid_size), dtype=int)
                self.astar_path = None

                # Add random internal obstacles
                for y in range(self.grid_size):
                    for x in range(self.grid_size):
                        if self.np_random.random() < CLUTTER_PROB:
                            grid_map[y, x] = 1

                # Try placing robot and goal
                for _ in range(100):  # Cap attempts
                    start_x = self.np_random.uniform(-ENV_BOUNDARY + 0.5, -0.5)
                    start_y = self.np_random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5)
                    goal_x = self.np_random.uniform(0.5, ENV_BOUNDARY - 0.5)
                    goal_y = self.np_random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5)

                    distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
                    if distance < MIN_DISTANCE:
                        continue

                    start_idx = self.world_to_grid(start_x, start_y)
                    goal_idx = self.world_to_grid(goal_x, goal_y)
                    
                    # Ensure robot and goal are not in obstacles
                    if not self.is_safe_spawn(grid_map, start_idx, safe_radius=1):
                        continue

                    if grid_map[goal_idx[0], goal_idx[1]] == 1:
                        continue

                    path = astar(grid_map, start_idx, goal_idx)
                    if path:
                        # Optional: forcibly clear spawn cells in grid map
                        grid_map[start_idx[0], start_idx[1]] = 0
                        grid_map[goal_idx[0], goal_idx[1]] = 0

                        self.grid_map = grid_map
                        self.astar_path = path
                        break

                if self.astar_path:
                    break  # Exit the world generation loop

        # After valid map found → setup physics world
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("plane.urdf")
        self.obstacle_ids = []

        wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_half_extents)
        wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=wall_half_extents, rgbaColor=[0.4, 0.4, 0.4, 1])

        # Populate the world with internal obstacles
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid_map[y, x] == 1:
                    wx, wy = self.grid_to_world(x, y)
                    body_id = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=wall_collision,
                        baseVisualShapeIndex=wall_visual,
                        basePosition=[wx, wy, wall_half_extents[2]]
                    )
                    self.obstacle_ids.append(body_id)

        # Add border walls around the grid
        if spawn_walls:
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if x == 0 or y == 0 or x == self.grid_size - 1 or y == self.grid_size - 1:
                        self.grid_map[y, x] = 1  # Mark as wall
                        wx, wy = self.grid_to_world(x, y)
                        body_id = p.createMultiBody(
                            baseMass=0,
                            baseCollisionShapeIndex=wall_collision,
                            baseVisualShapeIndex=wall_visual,
                            basePosition=[wx, wy, wall_half_extents[2]]
                        )
                        self.obstacle_ids.append(body_id)

        self.goal_position = [goal_x, goal_y, 0.0]   # z-coordinate will be set in helper


        


        # Draw A* path
        '''self.debug_lines = []
        for i in range(len(self.astar_path) - 1):
            (gy1, gx1) = self.astar_path[i]
            (gy2, gx2) = self.astar_path[i + 1]
            x1, y1 = self.grid_to_world(gx1, gy1)
            x2, y2 = self.grid_to_world(gx2, gy2)
            line_id = p.addUserDebugLine(
                lineFromXYZ=[x1, y1, 0.05],
                lineToXYZ=[x2, y2, 0.05],
                lineColorRGB=[0, 0, 1],
                lineWidth=2.0,
                lifeTime=0
            )
            self.debug_lines.append(line_id)'''

        self.robot = p.loadURDF(
            "./urdf/MicroROS.urdf",
            basePosition=[start_x, start_y, 0.05], 
            useFixedBase=False
        )

        self.goal_marker   = self._spawn_aruco(
            xy=(goal_x, goal_y),
            tag_png="aruco_rgb.png",     # or any file name you saved
            mesh_file="aruco_plane.obj", # your pre-made quad with UVs
            size=0.12
        )

        self.robot_start_pos = (start_x, start_y)


        num_joints = p.getNumJoints(self.robot)
        for joint in range(num_joints):
            p.changeDynamics(self.robot, joint, lateralFriction=0.7)
        p.changeDynamics(self.robot, -1, lateralFriction=0.7)
        p.changeDynamics(self.robot, -1, ccdSweptSphereRadius=0.05)
        p.setPhysicsEngineParameter(enableConeFriction=True)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.0001)
        p.setCollisionFilterPair(self.robot, self.plane_id, -1, -1, enableCollision=False)

        p.resetBaseVelocity(
            self.robot,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0]
        )

        self.previous_goal_distance = 0
        self.previous_goal_angle = 0
        
        return self._get_observation(), {}



    def step(self, action):
        """Advance one sim step with actuator dynamics & noise."""
        self.current_step += 1

        # Desired (commanded) speeds
        cmd_lin = float(action[0]) * MAX_LINEAR_SPEED
        cmd_ang = float(action[1]) * MAX_ANGULAR_SPEED

        # First-order actuator lag  v̇ = (v_cmd − v) / τ
        alpha_lin = min(CTRL_DT  / self.tau_lin, 1.0)   # τ_lin set in reset()
        alpha_ang = min(CTRL_DT  / self.tau_ang, 1.0)

        self.act_lin_speed += alpha_lin * (cmd_lin - self.act_lin_speed)
        self.act_ang_speed += alpha_ang * (cmd_ang - self.act_ang_speed)

        # Simple process / wheel-slip noise
        slip = self.np_random.normal(1.0, 0.03)          # ±3 % speed variation
        self.act_lin_speed = np.clip(
            self.act_lin_speed * slip,
            -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED
        )

        # Send *executed* velocity to Bullet
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        _, _, yaw = p.getEulerFromQuaternion(orn)

        vx = self.act_lin_speed * math.cos(yaw)
        vy = self.act_lin_speed * math.sin(yaw)

        p.resetBaseVelocity(
            self.robot,
            linearVelocity=[vx, vy, 0.0],
            angularVelocity=[0.0, 0.0, self.act_ang_speed]
        )
        for _ in range(SUB_STEPS):
            p.stepSimulation()

        # Observation, reward, bookkeeping 
        obs = self._get_observation()            # make sure this uses act_* speeds
        reward, done = self._compute_reward(action)

        self.episode_reward += reward
        self.episode_length += 1

        if self.current_step >= self.max_steps:
            done = True

        info = {}
        terminated = bool(done)
        truncated = False
        if done:
            info["episode"] = {"r": self.episode_reward, "l": self.episode_length}

        return obs, reward, terminated, truncated, info


    def get_goal_angle(self):
        robot_pos, robot_ori = p.getBasePositionAndOrientation(self.robot)
        rx, ry, _ = robot_pos
        gx, gy, _ = self.goal_position
        abs_angle = math.atan2(gy - ry, gx - rx)
        _, _, yaw = p.getEulerFromQuaternion(robot_ori)

        angle = abs_angle - yaw
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        return angle
    
    def compute_waypoint(self, robot_pos, goal_pos, R=WP_DISTANCE):
        dx = goal_pos[0] - robot_pos[0]
        dy = goal_pos[1] - robot_pos[1]
        dist_rg = np.sqrt(dx*dx + dy*dy)

        if dist_rg <= R:
            # Already near goal, so the “waypoint” is just the goal
            return goal_pos
        else:
            # Lambda = R / dist_rg
            lam = R / dist_rg
            wx = robot_pos[0] + lam * dx
            wy = robot_pos[1] + lam * dy
            return (wx, wy)
    
    def _compute_reward(self, action):

        if self.current_step == 1:
            self.previous_goal_distance = None
            self.previous_goal_angle = None
            self.collision_happened = False
            self.flag = 0.0
        
        
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        gx, gy, _ = self.goal_position
        goal_dist = math.sqrt((gx - robot_pos[0])**2 + (gy - robot_pos[1])**2)
        goal_angle = self.get_goal_angle() 

        lin_vel = self.act_lin_speed
        ang_vel = self.act_ang_speed
        norm_lin_vel = action[0]
        norm_ang_vel = action[1]
        lidar_scan = self.raw_lidar_scan
        min_distance = np.min(lidar_scan)

        delta = self.previous_goal_distance - goal_dist if self.previous_goal_distance is not None else 0.0
        self.previous_goal_distance = goal_dist

     


        

        # Detect collisions
        contacts = p.getContactPoints(self.robot)
        if min_distance < RADIUS_COLLISION:
            self.collision_happened = True
        for contact in contacts:
            if contact[2] not in [self.robot, self.plane_id, self.goal_marker] and contact[2] in self.obstacle_ids:
                self.collision_happened = True
        
        lidar_penalty = lambda x: 0.1*(0.6 - x) if x < 0.12 else 0.0

        if self.distance < GOAL_REACHED_DIST and self.flag > 0.5:
            print("GOAL REACHED! ")
            return REWARD_GOAL_BONUS, True
        elif self.collision_happened:
            print("COLLISION! ")
            return PENALTY_COLLISION, True
        '''elif lin_vel > 0 and abs(goal_angle) <= np.deg2rad(20):
            reward = 0.01*(0.5 - abs(goal_angle)/np.pi) - lidar_penalty(min_distance) 
            self.previous_goal_distance = goal_dist
            self.previous_goal_angle = goal_angle
            return reward, False
        else:
            #reward = 0.5*(lin_vel - 0.5*abs(ang_vel) - 1.5*lidar_penalty(min_distance))
            reward = -0.005 - lidar_penalty(min_distance)
            self.previous_goal_distance = goal_dist
            self.previous_goal_angle = goal_angle
            return reward, False'''

        if self.flag > 0.5:
            in_view = REWARD_IN_VIEW * self.flag
            progress = REWARD_PROGRESS * delta
        else:
            in_view = 0.0
            progress = 0.0
        smooth = REWARD_SMOOTH * (norm_lin_vel - 0.5*abs(norm_ang_vel))


        reward = in_view + progress + smooth - lidar_penalty(min_distance) #+ PENALTY_TIME
        #print("reward: ", reward, " progress: ", progress, " smooth: ", smooth, " lidar: ", lidar_penalty(min_distance), "distance: ", self.distance)

        return reward, False


    def _perform_lidar_scan(self):
        """Perform a simulated 180 ° LiDAR sweep (front half-plane)."""

        # Robot pose 
        robot_pos, robot_ori = p.getBasePositionAndOrientation(self.robot)
        robot_x, robot_y, robot_z = robot_pos

        # Compute yaw straight from the rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(robot_ori)).reshape(3, 3)
        # atan2( R21 , R11 ) is the standard way to extract yaw
        robot_yaw = math.atan2(rot_matrix[1, 0], rot_matrix[0, 0])

        # Ray start and angles
        start_pos = [robot_x, robot_y, robot_z + LIDAR_HEIGHT_OFFSET]

        angles = np.linspace(
            -math.pi / 2 ,    # start at –90°
            math.pi / 2 ,     # end at +90°
            NUM_READINGS,
            endpoint=False,
            dtype=np.float32
        )

        rotated_angles = angles + robot_yaw      # align with robot heading

        # Compute ray end points 
        end_positions = np.column_stack((
            robot_x + np.cos(rotated_angles) * MAX_LIDAR_RANGE,
            robot_y + np.sin(rotated_angles) * MAX_LIDAR_RANGE,
            np.full(NUM_READINGS, robot_z + LIDAR_HEIGHT_OFFSET)
        ))

        # Batch ray cast 
        results = p.rayTestBatch([start_pos] * NUM_READINGS, end_positions.tolist())

        ranges = []
        for hit_id, _, hit_fraction, *_ in results:
            if hit_id == self.robot:
                ranges.append(MAX_LIDAR_RANGE)           # ignore self-hits
            else:
                distance = hit_fraction * MAX_LIDAR_RANGE if hit_id != -1 else MAX_LIDAR_RANGE
                ranges.append(max(distance, MIN_LIDAR_RANGE))

        return np.asarray(ranges, dtype=np.float32)
    
    def _render_robot_camera(self):
        """
        Returns an (H, W, 3) uint8 RGB image from the robot's “nose camera”.
        """
        if self.current_step % 2 == 1 and hasattr(self, "last_rgb"):
            return self.last_rgb          # reuse previous frame
        w, h = CAM_RES
        fx, fy, fz = [CAM_OFFSET_L, 0, CAM_OFFSET_U]

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        # eye = base + R ⋅ offset
        eye = np.array(pos) + R @ np.array([fx, fy, fz])

        # look 1 m straight ahead in local +X
        target = eye + R @ np.array([1, 0, 0])
        up     = R @ np.array([0, 0, 1])           # local +Z

        view   = p.computeViewMatrix(eye, target, up)
        proj   = p.computeProjectionMatrixFOV(
                    fov=CAM_FOV, aspect=float(w)/h,
                    nearVal=CAM_NEAR, farVal=CAM_FAR)

        img = p.getCameraImage(
                width=w, height=h,
                viewMatrix=view, projectionMatrix=proj,
                renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb = np.reshape(img[2], (h, w, 4))[:, :, :3]   # RGBA→RGB
        self.last_rgb = rgb.copy()  # cache for next step
        return rgb.astype(np.uint8)
    
    def _aruco_distance_bearing(self, rgb):
        """
        Returns [d_norm, sin_bearing, tag_visible].
        d_norm      : 0 → close, 1 → far (≥ DIST_REF)
        sin_bearing : -1 left edge, 0 centre, +1 right edge
        tag_visible : 1 when tag detected, 0 otherwise
        """
        corners, ids, _ = cv2.aruco.detectMarkers(
            rgb, self.aruco_dict, parameters=self.aruco_param)

        if ids is None:
            self.flag = 0.0
            self.distance = 1.0
            self.bearing = 0.0
            return np.array([self.distance, self.bearing, 0.0], dtype=np.float32)

        # assume the first detected marker is the goal tag
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.TAG_SIZE_M, self.camera_K, None)
        x, y, z = tvec[0][0]            # camera frame: +Z forward

        self.distance = float(np.linalg.norm([x, y, z]))
        self.bearing  = math.atan2(x, z)     # left = +, right = –
        self.flag = 1.0
        sin_bearing = math.sin(self.bearing)
        d_norm = min(1.0, self.distance / DIST_REF_SIM)  # normalize to [0, 1] range

        return np.array([d_norm, sin_bearing, 1.0], dtype=np.float32)


    def _get_observation(self):

        camera_rgb = self._render_robot_camera()            # (84,84,3)
        tag_state = self._aruco_distance_bearing(camera_rgb)    # (3,)

        v_norm = np.clip(
            self.act_lin_speed / MAX_LINEAR_SPEED, -1.0, 1.0)
        w_norm = np.clip(
            self.act_ang_speed / MAX_ANGULAR_SPEED, -1.0, 1.0)

        # LiDAR data 
        self.raw_lidar_scan = self._perform_lidar_scan()
        weighted_lidar = np.exp(-self.raw_lidar_scan / EXP_DECAY_METERS)  # Weight by tag distance
        weighted_lidar[self.raw_lidar_scan >= MAX_LIDAR_RANGE] = 0.0
        '''self.past_observations = np.roll(self.past_observations, shift=-1, axis=0) # Shift new observation into list
        self.past_observations[-1] = lidar_scan
        flattened_lidar = self.past_observations.flatten()
        norm_lidar = (flattened_lidar - MIN_LIDAR_RANGE) / (MAX_LIDAR_RANGE - MIN_LIDAR_RANGE) # Min Max normalization'''
        
        obs = np.concatenate((
                tag_state,  # [distance, bearing, no_tag_flag]
                np.array([v_norm, w_norm], dtype=np.float32),
                weighted_lidar.astype(np.float32)
            ), axis=0)
        
        #obs = (obs - obs.mean()) / (obs.std() + 1e-8)


        return obs
    
    def is_safe_spawn(self, grid_map, idx, safe_radius=1.2):
        y, x = idx
        cell_radius = int(safe_radius / self.cell_size)  # convert meters to cells

        for dy in range(-cell_radius, cell_radius + 1):
            for dx in range(-cell_radius, cell_radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid_map.shape[0] and 0 <= nx < grid_map.shape[1]:
                    if grid_map[ny, nx] == 1:
                        return False
        return True
    
    

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()

