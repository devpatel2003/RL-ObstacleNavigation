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
MAX_LINEAR_SPEED = 0.2       # m/s
MAX_ANGULAR_SPEED = 3     # rad/s
GOAL_REACHED_DIST = 0.2     # Robot is "at" goal if closer than this

NUM_MOVING_OBSTACLES = 10
OBSTACLE_SPEED = 0.1

WP_DISTANCE = 0.8
RADIUS_COLLISION = 0.05
CLUTTER_PROB = 0.1

# LiDAR specs
NUM_READINGS = 60           #use 40 for old model   # 360° / 0.12° = 3000 <- LIDAR measurements ON REAL WORLD, 180° / 3° = 60 <- SIM  <-- demo model uses 64
MAX_LIDAR_RANGE = 18          # 3  # Up to ~12 m for black objects , 18 is max range of LIDAR on slam s2l lidar
MIN_LIDAR_RANGE = 0.03            # Minimum measurable distance
LIDAR_HEIGHT_OFFSET = 0.25         # Slightly above ground/robot’s base
EXP_DECAY_METERS = 1  # Exponential decay for LiDAR readings, to weight by distance to goal  (unused in new obs)

# Reward & penalty weights        #-0.2, 500, 10, 3, 10, 1
REWARD_GOAL_BONUS = 150
REWARD_WP = 200
REWARD_DTG_POSITIVE = 1.2    # Reward for reducing distance to goal, after * dist_improve number becomes very small
REWARD_HTG_POSITIVE = 1    # Reward for facing goal
REWARD_ACTION_HIGH = 1  # Reward for forward & near-zero rotation
REWARD_ACTION_MED = 0.5      # Reward for forward & rotating
ASTAR_GAIN = 1
PENALTY_COLLISION = -180
PENALTY_NEAR_COLLISION = -0.2
PENALTY_TURN = 0
PENALTY_TIME = -0.05

REWARD_IN_VIEW = 0.07  # Reward for being in view of the goal
REWARD_PROGRESS = 30  # Reward for making progress towards the goal, scaled by control time step
PENALTY_ALIGN = -1  # Penalty for not facing the goal
REWARD_SMOOTH = 0.05 #/ CTRL_DT  # Reward for smooth movement
PENALTY_STEP = -0.001  # Penalty for taking a step

CAM_RES        = (512, 512)          # width, height   (keep square for CNNs)
CAM_FOV        = 70               # degrees
CAM_NEAR, CAM_FAR = 0.05, 5.0
CAM_OFFSET_L   = 0.05             # 10 cm in front of base
CAM_OFFSET_U   = 0.10             # 12 cm above base
DIST_REF_SIM = 4.0  # Reference distance for normalizing ArUco tag distance, how far the sim cam can pick up the tag
USE_ANALYTIC_TAG = True   # toggle at top (or __init__ arg)


spawn_walls = True

# Memory/stacking params
FRAME_STACK = 5                    # number of LiDAR frames to stack
TAG_MEMORY_SEC = 10.0              # cap for time-since-tag-seen normalization (seconds)


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

        # Store last 5 LiDAR frames (framestacking)  (initialized later in reset())
        self.past_observations = np.zeros((FRAME_STACK, NUM_READINGS), dtype=np.float32)
        self.raw_lidar_scan = np.zeros(NUM_READINGS, dtype=np.float32)  # Store current LiDAR scan (meters)

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

        # ----- Observation layout -----
        # [d_norm, sin_bearing, tag_flag, v_norm, w_norm,
        #  sin_yaw, cos_yaw, delta_yaw,
        #  sin_last, cos_last, last_d_norm, time_since_seen_norm,
        #  LiDAR_stack (FRAME_STACK * NUM_READINGS)]
        scalar_low  = np.array([0, -1, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0], dtype=np.float32)
        scalar_high = np.array([1,  1, 1,  1,  1,  1,  1,  1,  1,  1, 1, 1], dtype=np.float32)
        lidar_low   = np.full(FRAME_STACK * NUM_READINGS, 0, dtype=np.float32)
        lidar_high  = np.full(FRAME_STACK * NUM_READINGS, 1, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.concatenate((scalar_low, lidar_low)),
            high=np.concatenate((scalar_high, lidar_high)),
            dtype=np.float32
        )

        self.robot = None
        self.left_wheel_joints = [1]   # zq_Joint, yh_Joint
        self.right_wheel_joints = []  # yq_Joint, zh_Joint

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
        self.tau_lin = 0.15
        self.tau_ang = 0.15
        self.distance = 0.0              # current detected tag distance  (m)
        self.bearing = 0.0               # current detected tag bearing  (rad, left=+)
        self.flag = 0.0                  # tag visible flag (0/1)

        # heading memory
        self.prev_yaw = 0.0

        # last‑seen tag memory (for obs when tag not visible)
        self.last_tag_bearing = 0.0      # last *camera* bearing
        self.last_tag_dist = DIST_REF_SIM
        self.steps_since_tag = 1_000_000 # large so normalized=1 at start

        # IMU noise params (tune)
        self.imu_bias_rate_std = math.radians(0.01)   # rad/s^1.5 random-walk
        self.imu_white_std     = math.radians(0.05)    # rad/s gyro noise
        self.imu_bias          = 0.0                  # will be sampled per reset
        self.imu_yaw_meas      = 0.0                  # integrated noisy yaw
        self.last_true_yaw     = 0.0                  # for delta calc


        self.CTRL_DT = CTRL_DT
        self.termination_reason = None  # "success", "collision", "timeout", or None



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
            yaw = self.np_random.uniform(-math.pi, math.pi)
            
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
        self.tau_lin = float(self.np_random.uniform(0.12, 0.23))
        self.tau_ang = float(self.np_random.uniform(0.15, 0.23))

        self.prev_yaw = 0.0


        # last tag memory reset
        self.last_tag_bearing = 0.0
        self.last_tag_dist = DIST_REF_SIM
        self.steps_since_tag = int(TAG_MEMORY_SEC / self.CTRL_DT)  # start "long ago"

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

        # capture yaw at episode start so we can report RELATIVE yaw
        _, robot_ori0 = p.getBasePositionAndOrientation(self.robot)
        _, _, yaw0 = p.getEulerFromQuaternion(robot_ori0)
        self.reset_yaw = yaw0          # store absolute world yaw at reset
        self.prev_rel_yaw = 0.0        # previous relative yaw (starts at 0)

        # sample a static gyro bias (deg/s → rad/s)
        self.imu_bias = self.np_random.uniform(
            low=math.radians(-0.2),
            high=math.radians(0.2)
        )
        # initialize true + measured yaw to 0 rel
        self.last_true_yaw = 0.0
        self.imu_yaw_meas  = 0.0



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

        for joint in self.left_wheel_joints + self.right_wheel_joints:
            p.setJointMotorControl2(
                self.robot, joint,
                controlMode=p.VELOCITY_CONTROL,
                force=0
            )

        p.resetBaseVelocity(
            self.robot,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0]
        )

        # clear LiDAR stack
        self.past_observations[:] = 0.0

        self.previous_goal_distance = None
        self.previous_goal_angle = None
        self.termination_reason = None

        
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

        # Simple process / wheel-slip noise   (reduced to ±1 %)
        slip = self.np_random.normal(1.0, 0.01)          # ±1 % speed variation
        self.act_lin_speed = np.clip(
            self.act_lin_speed * slip,
            -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED
        )

        # Send *executed* velocity to Bullet
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        _, _, yaw = p.getEulerFromQuaternion(orn)
        
        wheel_radius = 0.033
        half_axle = WHEEL_BASE / 2.0

        v = self.act_lin_speed
        w = self.act_ang_speed

        left_speed = (v - w * half_axle) / wheel_radius
        right_speed = (v + w * half_axle) / wheel_radius

        force_all = 0.5  # N·m, adjust for your robot's wheel motors

        # zq_Joint (0): Left wheel, REVERSED
        p.setJointMotorControl2(
            self.robot, 0,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_speed,
            force=force_all
        )

        # yq_Joint (1): Right wheel, NORMAL
        p.setJointMotorControl2(
            self.robot, 1,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-right_speed,
            force=force_all
        )

        # yh_Joint (2): Left wheel, NORMAL
        p.setJointMotorControl2(
            self.robot, 2,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-right_speed,
            force=force_all
        )

        # zh_Joint (3): Right wheel, REVERSED
        p.setJointMotorControl2(
            self.robot, 3,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_speed,
            force=force_all
        )

 

        for _ in range(SUB_STEPS):
            p.stepSimulation()

        # Observation, reward, bookkeeping 
        obs = self._get_observation()            # make sure this uses act_* speeds
        reward, done = self._compute_reward(action)

        self.episode_reward += reward
        self.episode_length += 1

        terminated = bool(done)            # success or collision
        truncated = False
        if self.current_step >= self.max_steps and not terminated:
            truncated = True
            self.termination_reason = "timeout"

        # Start info dict with defaults so Monitor never KeyErrors
        info = {"is_success": False,
                "is_collision": False,
                "is_timeout": False}

        if terminated:
            success   = (self.termination_reason == "success")
            collision = (self.termination_reason == "collision")
            info["is_success"]   = success
            info["is_collision"] = collision
        elif truncated:  # reached max steps
            info["is_timeout"] = True

        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_length,
            }

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
            # self.flag set in _aruco_distance_bearing()
        
        
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        gx, gy, _ = self.goal_position
        goal_dist = math.sqrt((gx - robot_pos[0])**2 + (gy - robot_pos[1])**2)
        goal_angle = self.get_goal_angle() 

        lin_vel = self.act_lin_speed
        ang_vel = self.act_ang_speed
        norm_lin_vel = action[0]
        norm_ang_vel = action[1]
        min_distance = np.min(self.raw_lidar_scan)  # meters

        
        # Detect collisions
        contacts = p.getContactPoints(self.robot)
        if min_distance < RADIUS_COLLISION:
            self.collision_happened = True
        for contact in contacts:
            if contact[2] not in [self.robot, self.plane_id, self.goal_marker] and contact[2] in self.obstacle_ids:
                self.collision_happened = True
        
        # smooth near-obstacle penalty ramp: 1.0 at contact radius, 0.0 when >= 3× radius
        if min_distance <0.8:
            danger = (0.8 - min_distance)  # 0..1
        else:
            danger = 0.0
        lidar_penalty = danger * (PENALTY_NEAR_COLLISION)  # positive scalar; we subtract below

        if goal_dist < GOAL_REACHED_DIST:
            #print("GOAL REACHED! ")
            self.termination_reason = "success"
            return REWARD_GOAL_BONUS, True

        elif self.collision_happened:
            #print("COLLISION! ")
            self.termination_reason = "collision"
            return PENALTY_COLLISION, True

        # progress weight reduced when tag not visible
        delta = self.previous_goal_distance - goal_dist if self.previous_goal_distance is not None else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        self.previous_goal_distance = goal_dist

        prog_scale = 1.0 if self.flag > 0.5 else 0.3 #<-- use 0.0 to disable progress reward when tag not visible
        progress = REWARD_PROGRESS * delta * prog_scale

        in_view = (1.0 - abs(math.sin(self.bearing))) * REWARD_IN_VIEW if self.flag > 0.5 else 0  # reward for being in view of the goal, used to be reward in view

        smooth = REWARD_SMOOTH * (norm_lin_vel - 0.25*abs(norm_ang_vel))

        reward = in_view + progress + smooth + lidar_penalty  + PENALTY_TIME
        #reward = smooth + lidar_penalty
        #print(f"Reward: {reward:.3f} | Progress: {progress:.3f} | In view: {in_view:.3f} | Smooth: {smooth:.3f} | Lidar penalty: {lidar_penalty:.3f}")

        self.termination_reason = None
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
            # do NOT overwrite distance/bearing here; preserve last detection for memory
            # self.distance holds last detection; when invisible we leave as-is
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # assume the first detected marker is the goal tag
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.TAG_SIZE_M, self.camera_K, None)
        x, y, z = tvec[0][0]            # camera frame: +Z forward

        
        # update last-seen memory
        self.last_tag_bearing = self.bearing
        self.last_tag_dist = self.distance
        self.steps_since_tag = 0

        self.distance = float(np.linalg.norm([x, y, z]))
        self.bearing  = math.atan2(x, z)     # left = +, right = –
        self.flag = 1.0

        sin_bearing = math.sin(self.bearing)
        d_norm = min(1.0, self.distance / DIST_REF_SIM)  # normalize to [0, 1] range

        return np.array([d_norm, sin_bearing, 1.0], dtype=np.float32)
    
    USE_ANALYTIC_TAG = True   # toggle at top (or __init__ arg)

    def _tag_state_gt(self):
        """
        Cheap analytic 'tag' sensor using ground-truth poses.
        Mimics ArUco outputs: [d_norm, sin_bearing, tag_flag].
        Optionally enforces 180° FOV and occlusion test.
        """
        # robot pose
        (rx, ry, _), robot_ori = p.getBasePositionAndOrientation(self.robot)
        _, _, yaw = p.getEulerFromQuaternion(robot_ori)

        gx, gy, _ = self.goal_position
        dx = gx - rx
        dy = gy - ry
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - yaw
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi

        # visible if within ±70° (front hemi); cheap occlusion check via single ray
        visible = abs(bearing) <= (math.radians(CAM_FOV) / 2.0)
        if visible:
            # raycast to goal ~waist height
            start = [rx, ry, 0.2]
            end   = [gx, gy, 0.2]
            hit_id, *_ = p.rayTest(start, end)[0]
            if hit_id not in (-1, self.goal_marker):
                visible = False

        if visible:
            self.flag = 1.0
            self.distance = dist
            self.bearing = bearing
            self.last_tag_bearing = bearing
            self.last_tag_dist = dist
            self.steps_since_tag = 0
            d_norm = min(1.0, dist / DIST_REF_SIM)
            return np.array([d_norm, math.sin(bearing), 1.0], dtype=np.float32)
        else:
            self.flag = 0.0
            # don't overwrite last seen memory
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)



    def _get_observation(self):

        if USE_ANALYTIC_TAG:
            tag_state = self._tag_state_gt()
        else:
            camera_rgb = self._render_robot_camera()
            tag_state = self._aruco_distance_bearing(camera_rgb)

        v_norm = np.clip(
            self.act_lin_speed / MAX_LINEAR_SPEED, -1.0, 1.0)
        w_norm = np.clip(
            self.act_ang_speed / MAX_ANGULAR_SPEED, -1.0, 1.0)

        # Robot heading (relative yaw)
        _, robot_ori = p.getBasePositionAndOrientation(self.robot)
        _, _, yaw_abs = p.getEulerFromQuaternion(robot_ori)
        yaw_rel_true = yaw_abs - self.reset_yaw
        yaw_rel_true = (yaw_rel_true + math.pi) % (2 * math.pi) - math.pi

        # integrate noisy IMU yaw from last step
        # true delta since last step:
        delta_true = yaw_rel_true - self.last_true_yaw
        delta_true = (delta_true + math.pi) % (2 * math.pi) - math.pi

        # add bias + white noise
        gyro_meas = delta_true / self.CTRL_DT \
                    + self.imu_bias \
                    + self.np_random.normal(0.0, self.imu_white_std)

        # random-walk bias drift (optional small)
        self.imu_bias += self.np_random.normal(0.0, self.imu_bias_rate_std) * math.sqrt(self.CTRL_DT)

        # integrate measured yaw
        self.imu_yaw_meas += gyro_meas * self.CTRL_DT
        self.imu_yaw_meas = (self.imu_yaw_meas + math.pi) % (2 * math.pi) - math.pi

        # update last true yaw
        self.last_true_yaw = yaw_rel_true

        # choose what to feed the agent: measured, not true
        yaw_rel = self.imu_yaw_meas

        sin_yaw = math.sin(yaw_rel)
        cos_yaw = math.cos(yaw_rel)

        # delta in relative yaw since last obs (normalized to [-1,1])
        delta_rel_yaw = yaw_rel - self.prev_rel_yaw
        delta_rel_yaw = (delta_rel_yaw + math.pi) % (2 * math.pi) - math.pi
        delta_yaw_norm = np.clip(delta_rel_yaw / math.pi, -1.0, 1.0)
        self.prev_rel_yaw = yaw_rel

        # Update last-seen goal bearing when tag NOT visible
        if self.flag < 0.5:
            # goal stays fixed in world; robot rotated by +delta_rel_yaw, so goal bearing in robot frame shifts by -delta_rel_yaw
            self.last_tag_bearing -= delta_rel_yaw
            # wrap to [-pi, pi]
            self.last_tag_bearing = (self.last_tag_bearing + math.pi) % (2 * math.pi) - math.pi

        # LiDAR data (meters -> normalized inverted: near=1, far=0)
        self.raw_lidar_scan = self._perform_lidar_scan()
        lidar_norm_inv = 1.0 - np.clip(self.raw_lidar_scan / MAX_LIDAR_RANGE, 0.0, 1.0)

        # framestack
        self.past_observations = np.roll(self.past_observations, shift=-1, axis=0) # Shift new observation into list
        self.past_observations[-1] = lidar_norm_inv
        lidar_flat = self.past_observations.flatten()

        # time since tag seen (seconds -> 0..1)
        self.steps_since_tag += 1
        t_seen = min(self.steps_since_tag * self.CTRL_DT, TAG_MEMORY_SEC)
        time_since_seen_norm = t_seen / TAG_MEMORY_SEC

        # last tag memory features
        sin_last = math.sin(self.last_tag_bearing)
        cos_last = math.cos(self.last_tag_bearing)
        last_d_norm = min(1.0, self.last_tag_dist / DIST_REF_SIM)

        mem_feats = np.array([sin_yaw, cos_yaw, delta_yaw_norm,
                              sin_last, cos_last, last_d_norm, time_since_seen_norm],
                             dtype=np.float32)

        obs = np.concatenate((
                tag_state,  # [distance, bearing, tag_flag]
                np.array([v_norm, w_norm], dtype=np.float32),
                mem_feats,
                lidar_flat.astype(np.float32)
            ), axis=0).astype(np.float32, copy=False)
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
