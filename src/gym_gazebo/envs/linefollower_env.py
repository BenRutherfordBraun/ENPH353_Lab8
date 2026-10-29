import cv2
import rclpy
import time
import gym_gazebo
import gymnasium as gym
import numpy as np
from cv_bridge import CvBridge
from gym_gazebo.core.gazebo_env import GazeboEnv
from rclpy.node import Node as RosNode
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Twist as RosTwist

NUM_OBS_STATES = 3
NUM_ACTIONS = 3

class LineFollowerEnv(GazeboEnv):
    def __init__(self):
        # Intializing ROS2
        rclpy.init(args=None)
        
        super(LineFollowerEnv, self).__init__(launch_pkg='linefollow_ros',
                                    launch_file='my_agent.launch.xml',
                                    world_name='monza_gym')

        # Setting up ROS2 node
        self.ros_node = RosNode('line_follower_RL')

        # Initializing variables
        self.is_resetting = False
        self.latest_obs = None
        self.new_obs_event = False

        # Setting the observation and action spaces
        self.observation_space = gym.spaces.Discrete(NUM_OBS_STATES)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        # ROS2 topics
        observation_topic = "/camera/image_raw"
        velocity_topic = "/cmd_vel"

        # Publishing and subscribing to ROS2 topics
        self.pub_cmd_vel_msg = self.ros_node.create_publisher(
            RosTwist, velocity_topic, 10)
            
        self.sub_obs = self.ros_node.create_subscription(
            RosImage, observation_topic, self.callback_obs_feed_, 10)


    def callback_obs_feed_(self, msg: RosImage):
        '''
        @brief observation callback resets the new_obs_event flag
        @param msg: Image from Gazebo sim
        '''
        # Takes in observation, returns a flag an observation is recieved
        if not self.is_resetting:
            self.latest_obs = msg
            self.new_obs_event = True


    ## MAIN FUNCTIONS:
    def reset(self, seed=None, options=None):
        '''
        @brief resets the simulation
        '''
        self.is_resetting = True
        print("Resetting")

        # Zero out velocities
        vel_cmd = RosTwist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.pub_cmd_vel_msg.publish(vel_cmd)

        # Reset position and joints
        self._reset_agents()

        # Unpause the sim
        self._pause_sim(False)

        # Allow some time to settle
        time.sleep(0.1) # Wait 100 ms
        
        self.is_resetting = False

        # Wait for next observation to come in before kickstarting the data loop
        self.new_obs_event = False
        while not self.new_obs_event:
            # Telling executor (node) to spin
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # # Pause the sim after the first observation comes in
        self._pause_sim(True)
        
        state, _ = self.process_obs(self.latest_obs)

        print("Done resetting")
        return state, {}
    
    
    def step(self, action: int):
        '''
        @brief takes a step in the simulation
        @param action
        @retval state
        @retval reward
        @retval terminated
        @retval truncated
        '''
        # Process the action
        vel_cmd = self.process_action(action)

        # Unpause the sim
        self._pause_sim(False)

        # Moving the agent
        self.pub_cmd_vel_msg.publish(vel_cmd)

        # Leave running until next observation comes in
        self.new_obs_event = False
        while not self.new_obs_event:
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # Pause the simulation after the next observation comes in
        self._pause_sim(True)

        state, terminated = self.process_obs(self.latest_obs)
        
        reward = self.process_reward(action, state, terminated)

        truncated = False # Used if we ran maximum number of steps.
        return state, reward, terminated, truncated, {}

    
    ## HELPER FUNCTIONS
    def process_action(self, action: int):
        '''
        @brief Translate the action the robot is taking into a Twist message
        @param action: an integer where 0 is left, 1 is forward and 2 is right 
        @retval vel_cmd: Twist message equivalent of the action
        '''
        vel_cmd = RosTwist()

        if action == 0:  # Turn left
            vel_cmd.linear.x = 0.3
            vel_cmd.angular.z = 0.6
        elif action == 1:  # Move forward
            vel_cmd.linear.x = 0.8
            vel_cmd.angular.z = 0.0
        elif action == 2:  # Turn right
            vel_cmd.linear.x = 0.3
            vel_cmd.angular.z = -0.6
        
        return vel_cmd


    def process_reward(self, action, state, terminated):
        '''
        @brief depending on the action, state and terminated values assign
               a reward
        @retval reward: the reward for the current step
        '''
        # Severe penalty for losing the line
        if state == -1 or terminated:
            return -200
            
        reward = 0
        
        # State 0: Line is on the left
        if action == 0:
            
            reward = 1
            
                
        # State 1: Line is in the center
        elif action == 1:
            
            reward = 5
            
                
        # State 2: Line is on the right
        elif action == 2:
            
            reward = 1
            
                
        return reward


    def process_obs(self, obs):
        '''
        @brief Identifies the line in the image and populates the state array
               based on the line position (i.e. state=0 line is left, 
               state=1 line is in the center)
        @param obs raw image from Gazebo sim
        @retval state: the line position as a value: 
            -1 (no line), 0 (left), 1 (center), 2 (right)
        @retval terminated: the termination condition occured - the line is no 
                    longer in the field of the camera
        '''
        cv_image = CvBridge().imgmsg_to_cv2(obs, desired_encoding='bgr8')
        height, width, _ = cv_image.shape

        # 1. Isolate Region of Interest (ROI)
        # Focus only on the bottom third of the image to ignore background clutter
        roi_top = int(2 * height / 3)
        roi = cv_image[roi_top:height, 0:width]

        # 2. Convert to grayscale and apply binary thresholding
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # ASSUMPTION: Black line on a light background. 
        # THRESH_BINARY_INV turns the dark line white (255) for contour detection.
        # If using a white line on a dark floor, change to cv2.THRESH_BINARY
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # 3. Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Default to "no line" condition
        state = -1
        terminated = True

        # 4. Extract center point if a line is detected
        if len(contours) > 0:
            # Assume the largest contour is the track line
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate image moments to find the centroid (center of mass)
            M = cv2.moments(largest_contour)
            
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                
                # 5. Map the x-coordinate to the discrete state space
                left_bound = width / 3
                right_bound = 2 * width / 3

                if cx < left_bound:
                    state = 0       # Line is in the left third
                elif cx > right_bound:
                    state = 2       # Line is in the right third
                else:
                    state = 1       # Line is in the center third
                    
                terminated = False  # Line successfully tracked

        return state, terminated


    # Shutting down ROS2
    def user_close(self):
        self.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()