#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag


def load_map(filename):
    im = mpimg.imread("../maps/" + filename)  
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np


def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:    
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only ever be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_settings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_settings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 0.1 #s
        self.num_substeps = 100

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]
        self.workspace = self.bounds      # search space (m)

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self) -> np.ndarray:
        #Return an [x,y] coordinate to drive the robot towards
        # print("TO DO: Sample point to drive towards")

        point = np.random.rand(2, 1) * np.reshape(self.workspace[:, 1] - self.workspace[:, 0], (2, 1)) + np.reshape(self.workspace[:, 0], (2, 1))   # sample point from self.workspace

        return np.reshape(point, (2, 1))
    
    def check_if_duplicate(self, point) -> bool:
        #Check if point is a duplicate of an already existing node
        # print("TO DO: Check that nodes are not duplicates")

        # see closest_node function

        return self.closest_node(point) != -1
    
    def closest_node(self, point) -> int:
        #Returns the index of the closest node
        # print("TO DO: Implement a method to get the closest node to a sampled point")

        min_dist = np.inf
        closest = -1        # index of current closest node

        # calculate euclidean distance to each node; also check for duplicates
        for i, n in enumerate(self.nodes):
            # duplicate?
            if point[0, 0] == n.point[0, 0] and point[1, 0] == n.point[1, 0]:
                return -1

            distance = np.linalg.norm(point - n.point[0:2])
            if distance < min_dist:
                min_dist = distance
                closest = i

        return closest
    
    def simulate_trajectory(self, point_i: np.ndarray, point_s: np.ndarray) -> np.ndarray:
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        # print("TO DO: Implement a method to simulate a trajectory given a sampled point")
        
        vel, rot_vel = self.robot_controller(point_i, point_s)

        robot_traj = self.trajectory_rollout(point_i, vel, rot_vel)
        return robot_traj
    
    def robot_controller(self, point_i: np.ndarray, point_s: np.ndarray) -> tuple:
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        # print("TO DO: Implement a control scheme to drive you towards the sampled point")

        R = np.array([
            [0, 1],
            [-1, 0]
        ])

        theta_s = np.arctan2(point_s[1, 0] - point_i[1, 0], point_s[0, 0] - point_i[0, 0])
        dtheta = theta_s - point_i[2, 0]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

        pi = point_i[0:2]
        ps = np.array(point_s)

        direction = -1 if dtheta < 0 else 1
        Rd = R if dtheta < 0 else R.T 
        Rn = R if dtheta < 0 and dtheta > -np.pi/2 or dtheta > np.pi/2 and dtheta < np.pi else R.T 

        d = Rd @ np.array([[np.cos(point_i[2, 0]), np.sin(point_i[2, 0])]]).T
        l = ps - pi
        m = pi + l/2
        n = Rn @ l

        # solve for circle
        coeffs = np.linalg.inv(np.hstack([d, -n])) @ (m-pi)
        c = pi + coeffs[0] * d
        r = np.linalg.norm(c - pi)

        arc_angle = 2 * np.arccos(np.dot(d[:,0], n[:,0]) / np.linalg.norm(d) / np.linalg.norm(n))
        s = r * arc_angle 

        # speeds
        v = min(self.vel_max, s / self.num_substeps / self.timestep)
        w = direction * arc_angle * v / s 

        max_retry = 10
        i = 0
        while w > self.rot_vel_max and i < max_retry:
            v = v * self.rot_vel_max / w 
            w = arc_angle * v / s 
            i += 1

        return v, w
    
    def trajectory_rollout(self, origin: np.ndarray, vel: float, rot_vel: float) -> np.ndarray:
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        # print("TO DO: Implement a way to rollout the controls chosen")

        positions = np.zeros((3, self.num_substeps))
        positions[:, 0] = np.reshape(origin, (3,))

        for i in range(1, self.num_substeps):
            theta = positions[2, i-1]

            R = np.matrix([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

            q_dot = R @ np.array([[vel, 0, rot_vel]]).T
                
            positions[:, i] = positions[:, i-1] + np.reshape(q_dot * self.timestep, (3,))

        # return np.zeros((3, self.num_substeps))
        return positions
    
    def point_to_cell(self, point):
        # Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        # point is a 2 by N matrix of points of interest

        # point (np.ndarray): An N by 2 matrix of points of interest, where N is the number of points.
        # Returns: np.ndarray: An array of cell indices [row,col] in the occupancy map corresponding to each input point.

        # Calibrate offset: 
        origin = self.map_settings_dict["origin"][:2]
        point_calibrated = point - origin


        cell_idx = (point_calibrated/self.map_settings_dict["resolution"]).astype(int)
        
        # Flip Y-Axis
        cell_idx[:, 1] = (cell_idx[:, 1] - self.map_shape[1])*-1

        x = cell_idx[:, 1]
        y = cell_idx[:, 0]

        cell = np.column_stack((x, y))

        return cell


    def points_to_robot_circle(self, points):
        # Convert a series of [x,y] points to robot map footprints for collision detection
        # Hint: The disk function is included to help you with this function
        # print("TO DO: Implement a method to get the pixel locations of the robot path")

        # Calling point_to_cell
        points_idx = self.point_to_cell(points)  
        robot_radius_in_cells = self.robot_radius / self.map_settings_dict["resolution"]
        footprints = []
        for i in range(points_idx.shape[0]):
            row, col = points_idx[i]

            # Generate the disk footprint (ensures points are within map boundaries)
            rr, cc = disk(center=(row, col), radius=robot_radius_in_cells, shape=self.map_shape)
            footprints.append(np.column_stack((rr, cc)))  # Stack rows and columns into Nx2 format

        return footprints
    
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        return np.zeros((3, self.num_substeps))
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        print("TO DO: Implement a cost to come metric")
        return 0
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    #Planner Functions
    def rrt_planning(self, max_steps=1000, focused_window=5):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        # self.workspace = np.array([
        #     [-1, 1],
        #     [-1, 1]
        # ], dtype='float') * 2
        self.workspace = self.bounds

        print()
        start_time = time.time()

        for step in range(max_steps): #Most likely need more iterations than this to complete the map!
            print(f"\rPlanning... ({step}/{max_steps}) {time.time() - start_time:.2f}s", end='')

            #Sample map space
            point = self.sample_map_space()

            #Get the closest point
            closest_node_id = self.closest_node(point)
            if closest_node_id == -1:
                continue    # duplicate

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for collisions
            # print("TO DO: Check for collisions and add safe points to list of nodes.")

            # occupied cells along trajectory
            traj_cells = self.points_to_robot_circle(trajectory_o[0:2, :].T)

            safe = True
            at_goal = False
            for i, cells in enumerate(traj_cells):
                occupied_cells = self.occupancy_map[cells[:, 0], cells[:, 1]]
                if np.any(occupied_cells < 1):
                    safe = False 
                    break
                if np.linalg.norm(self.goal_point.reshape((2,)) - trajectory_o[:2, i].reshape((2,))) < self.stopping_dist:
                    at_goal = True 
                    trajectory_o = trajectory_o[:, :i+1]
                    break

            #Check if goal has been reached
            # print("TO DO: Check if at goal point.")
            # at_goal = False
            # for i, tp in enumerate(trajectory_o.T):
            #     if np.linalg.norm(self.goal_point.reshape((2,)) - tp[:2]) < self.stopping_dist:
            #         at_goal = True
            #         trajectory_o = trajectory_o[:, :i+1]
            #         break

            # add node to list if safe
            if safe:
                self.nodes.append(Node(
                    # np.vstack([point, trajectory_o[-1, -1]]),
                    trajectory_o[:, -1].reshape((3, 1)),
                    closest_node_id,
                    self.nodes[closest_node_id].cost + 1
                ))

                # expand search space
                if step > 0.9*max_steps:
                    self.workspace[0, 0] = max(self.bounds[0, 0], self.goal_point[0] - focused_window)
                    self.workspace[1, 0] = max(self.bounds[1, 0], self.goal_point[1] - focused_window)
                    self.workspace[0, 1] = min(self.bounds[0, 1], self.goal_point[0] + focused_window)
                    self.workspace[1, 1] = min(self.bounds[1, 1], self.goal_point[1] + focused_window)        
                else:
                    endpoint = trajectory_o[:2, -1]
                    self.workspace[0, 0] = max(self.bounds[0, 0], min(self.workspace[0, 0], endpoint[0] - self.timestep * self.num_substeps * 1.5))
                    self.workspace[1, 0] = max(self.bounds[1, 0], min(self.workspace[1, 0], endpoint[1] - self.timestep * self.num_substeps * 1.5))
                    self.workspace[0, 1] = min(self.bounds[0, 1], max(self.workspace[0, 1], endpoint[0] + self.timestep * self.num_substeps * 1.5))
                    self.workspace[1, 1] = min(self.bounds[1, 1], max(self.workspace[1, 1], endpoint[1] + self.timestep * self.num_substeps * 1.5))

                # visualise
                # self.window.add_point(point.reshape((2,)), radius=3, color=(0, 0, 255))
                # self.window.add_se2_pose(np.hstack([np.reshape(point, (2,)), trajectory_o[-1, -1]]), length=5, color=(0, 0, 255))
                self.window.add_se2_pose(trajectory_o[:, -1].reshape(3,), length=5, color=(0, 0, 255))
                for i, tp in enumerate(trajectory_o[0:2, :].T):
                    self.window.add_point(tp, radius=1, color=(0, 50, 0))
                # for i, (tp1, tp2) in enumerate(zip(trajectory_o[0:2, :-1].T, trajectory_o[0:2, 1:].T)):
                #     self.window.add_line(tp1, tp2)
                # view window
                # self.window.add_line(np.array([self.workspace[0, 0], self.workspace[1, 0]]), np.array([self.workspace[0, 1], self.workspace[1, 0]]), width=3, color=(0, 0, 255))
                # self.window.add_line(np.array([self.workspace[0, 1], self.workspace[1, 0]]), np.array([self.workspace[0, 1], self.workspace[1, 1]]), width=3, color=(0, 0, 255))
                # self.window.add_line(np.array([self.workspace[0, 0], self.workspace[1, 0]]), np.array([self.workspace[0, 0], self.workspace[1, 1]]), width=3, color=(0, 0, 255))
                # self.window.add_line(np.array([self.workspace[0, 0], self.workspace[1, 1]]), np.array([self.workspace[0, 1], self.workspace[1, 1]]), width=3, color=(0, 0, 255))

            if at_goal:
                break
        
        print()

        # view final path
        path = self.recover_path()
        for i, p in enumerate(path[:-1]):
            self.window.add_se2_pose(p[:, 0], length=10, color=(0, 255, 0))
            # traj = self.simulate_trajectory(p, path[i+1][:2])
            # for j, tp in enumerate(traj[0:2, :].T):
            #     self.window.add_point(tp, radius=3, color=(100, 255, 100))
        self.window.add_se2_pose(path[-1][:, 0], length=10, color=(0, 255, 0))

        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for Collision
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    np.random.seed(10000)   # 10000 is fast for simple_map
    #Set map information
    # map_filename = "willowgarageworld_05res.png"
    # map_settings_filename = "willowgarageworld_05res.yaml"
    # map_filename = "simple_map.png"
    # map_settings_filename = "simple_map.yaml"
    map_filename = "myhal.png"
    map_settings_filename = "myhal.yaml"

    #robot information
    # goal_point = np.array([[10], [10]]) #m  # seed 20700
    # goal_point = np.array([[41.5], [-44]])
    # goal_point = np.array([[30], [30]])
    goal_point = np.array([[7], [0]])
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_settings_filename, goal_point, stopping_dist)
    # nodes = path_planner.rrt_star_planning()
    nodes = path_planner.rrt_planning(10000, 5)
    node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)
    np.save("nodes.npy", nodes)


if __name__ == '__main__':
    main()