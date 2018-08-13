'''
Coders Strike Back Single mode environment
author: Alessandro Nicolosi
url: https://github.com/alenic/gym-coders-strike-back
Original game: https://www.codingame.com/multiplayer/bot-programming/coders-strike-back
'''
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CodersStrikeBackSingle(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        # Game constants
        self.maxThrust = 200.0
        self.maxSteeringAngle = 0.1*np.pi # Maximum steering angle in radians [+-0.1 rad = +-15 Deg]
        self.friction = 0.85
        self.__M_PI2 = 2.0*np.pi

        self.seed()
        self.viewer = None
        # State = [theta, x, xdot, y, ydot, firstCkptX, firstCkptY, secondCkptX, secondCkptY]
        self.state = None 
        self.steps_beyond_done = None
        
        minPos = -20000.0
        maxPos = 20000.0
        minVel = -600.0
        maxVel = 600.0
        # action_space = [targetX, targetY, thrust]
        self.action_space = spaces.Box(low=np.array([minPos, minPos, 0.0]),
                                       high=np.array([maxPos, maxPos, self.maxThrust]))

        self.observation_space = spaces.Box(np.array([0.0, minPos, minVel, minPos, minVel, 0, 0, 0, 0]),
                                            np.array([self.__M_PI2, maxPos, maxVel, maxPos, maxVel, 16000, 9000, 16000, 9000]))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def __getDeltaAngle(self, a):
      right = a - self.state[0] if self.state[0] <= a else self.__M_PI2 - self.state[0] + a
      left =  self.state[0] - a if self.state[0] >= a else self.state[0] + self.__M_PI2 - a

      if right < left:
        return right
      else:
        return -left

    # Game dynamics
    def movePod(self, targetX, targetY, thrust):
        theta, x, x_dot, y, y_dot = self.state[:5]
        targetAngle = math.atan2(targetY-y, targetX-x)
        da = self.__getDeltaAngle(targetAngle)

        if da > self.maxSteeringAngle:
            da = self.maxSteeringAngle
        elif da < -self.maxSteeringAngle:
            da = -self.maxSteeringAngle
        
        theta += da

        if theta >= self.__M_PI2:
            theta -= self.__M_PI2

        if theta < 0.0:
            theta += self.__M_PI2
        
        # Update dynamics
        x_dot = x_dot + math.cos(theta)*thrust
        y_dot = y_dot + math.sin(theta)*thrust

        x = round(x + x_dot)
        y = round(y + y_dot)

        # Apply the friction
        x_dot = int(0.85*x_dot)
        y_dot = int(0.85*y_dot)
        
        self.state[:5] = theta, x, x_dot, y, y_dot


    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        self.movePod(action[0], action[1], action[2])
        theta, x, x_dot, y, y_dot, firstCkptX, firstCkptY, secondCkptX, secondCkptY = self.state
        
        playerPos = np.array([x,y])
        firstCkptPos = np.array([firstCkptX, firstCkptY])

        done = False
        reward = -1.0
        if np.linalg.norm(playerPos-firstCkptPos) < 600:
            ckptIndex = self.firstCkptIndex
            reward = 0.0
            if ckptIndex== self.nCkpt-1:
                done = True
            elif ckptIndex == self.nCkpt-2:
                ckptIndex += 1
                firstCkptX, firstCkptY = self.checkpoint[ckptIndex]
                secondCkptX, secondCkptY = [-1, -1]
            else:
                ckptIndex += 1
                firstCkptX, firstCkptY = self.checkpoint[ckptIndex]
                secondCkptX, secondCkptY = self.checkpoint[ckptIndex+1]

            self.state[5:] = firstCkptX, firstCkptY, secondCkptX, secondCkptY
            self.firstCkptIndex = ckptIndex

        self.tick += 1
        if self.tick >= 10000:
            done = True

        if done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                if self.steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1
                reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        # Checkpoint coordinates
        self.nCkpt = self.np_random.randint(2, 15)
        checkpointX = self.np_random.randint(500, 15500, (self.nCkpt, 1))
        checkpointY = self.np_random.randint(500, 8500, (self.nCkpt, 1))
        self.checkpoint = np.column_stack([checkpointX,checkpointY])
        self.firstCkptIndex = 0
        # State init
        self.state = np.zeros(9)
        self.state[0] = self.np_random.randint(0,359)*np.pi/180.0
        self.state[1] = self.np_random.randint(500,15500)
        self.state[2] = self.np_random.randint(-50,50)
        self.state[3] = self.np_random.randint(500,8500)
        self.state[4] = self.np_random.randint(-50,50)
        self.state[5:7] = self.checkpoint[0]
        self.state[7:9] = self.checkpoint[1]

        self.tick = 0
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 640
        screen_height = 360
        scale = screen_width/16000
        podRadius = scale*400
        checkpointRadius = scale*600

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.checkpointCircle = []
            self.checkpointCircle2 = []
            for i in range(self.nCkpt):
                self.checkpointCircle.append(rendering.make_circle(checkpointRadius, filled=False))
                self.checkpointCircle2.append(rendering.make_circle(checkpointRadius*0.5, filled=True))
                self.checkpointCircle[-1].set_color(0.7,0.7,1.0)
                self.checkpointCircle2[-1].set_color(1,1,1)
                self.transformeCkpt = rendering.Transform()
                self.checkpointCircle[-1].add_attr(self.transformeCkpt)
                self.checkpointCircle2[-1].add_attr(self.transformeCkpt)
                self.viewer.add_geom(self.checkpointCircle2[-1])
                self.viewer.add_geom(self.checkpointCircle[-1])
                self.transformeCkpt.set_translation(scale*self.checkpoint[i][0], scale*self.checkpoint[i][1])
                
            p1 = (podRadius*math.cos(150.0*np.pi/180.0), podRadius*math.sin(150.0*np.pi/180.0))
            p2 = (podRadius, 0)
            p3 = (podRadius*math.cos(210.0*np.pi/180.0), podRadius*math.sin(210.0*np.pi/180.0))
            pod = rendering.FilledPolygon([p1,p2,p3])
            pod.set_color(1.0,0.5,0.5)
            self.podTransform = rendering.Transform()
            pod.add_attr(self.podTransform)
            self.viewer.add_geom(pod)

        if self.state is None: return None
        
        for i in range(self.firstCkptIndex):
            self.checkpointCircle[i].set_color(1,1,1)
            self.checkpointCircle2[i].set_color(1,1,1)

        self.checkpointCircle[self.firstCkptIndex].set_color(0,0,0)
        self.checkpointCircle2[self.firstCkptIndex].set_color(0,0,0)
        if self.firstCkptIndex <= self.nCkpt-2:
            self.checkpointCircle[self.firstCkptIndex+1].set_color(0,1,0)
        self.podTransform.set_translation(scale*self.state[1], scale*self.state[3])
        self.podTransform.set_rotation(self.state[0])
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()