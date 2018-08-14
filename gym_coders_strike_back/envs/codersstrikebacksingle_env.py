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
import os


class CodersStrikeBackSingle(gym.Env):
    metadata = {
        'render.modes': ['human'],
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
            reward = 50.0
            if ckptIndex== self.nCkpt-1:
                done = True
            elif ckptIndex == self.nCkpt-2:
                ckptIndex += 1
                firstCkptX, firstCkptY = self.checkpoint[ckptIndex]
                secondCkptX, secondCkptY = [-1, -1]
                reward = 100.0 # Last checkpoint
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
        mGrid = 4
        nGrid = 8
        self.gamePixelWidth = 16000
        self.gamePixelHeight = 9000
        self.nCkpt = mGrid*nGrid
        self.checkpoint = np.zeros((self.nCkpt,2))
        # Get
        ckptList = np.arange(self.nCkpt)
        np.random.shuffle(ckptList)
        ckptGrid = ckptList.reshape(mGrid,nGrid)

        for i in range(mGrid):
            for j in range(nGrid):
                index = ckptGrid[i][j]
                self.checkpoint[index][0] = int((j+0.5)*self.gamePixelWidth//nGrid)
                self.checkpoint[index][1] = int((i+0.5)*self.gamePixelHeight//mGrid)

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
        scale = screen_width/self.gamePixelWidth
        podRadius = scale*400*2.0
        checkpointRadius = scale*600*2.0

        if self.viewer is None:
            from gym_coders_strike_back.envs import pygame_rendering
            self.viewer = pygame_rendering.Viewer(screen_width, screen_height)
            
            dirname = os.path.dirname(__file__)
            backImgPath = os.path.join(dirname, 'imgs','back.png')
            self.viewer.setBackground(backImgPath)
            
            ckptImgPath = backImgPath = os.path.join(dirname, 'imgs','ckpt.png')
            
            self.checkpointCircle = []
            for i in range(self.nCkpt):
                xCkpt = scale*self.checkpoint[i][0]
                yCkpt = scale*self.checkpoint[i][1]
                ckptBoject = pygame_rendering.Checkpoint(ckptImgPath, pos=(xCkpt, yCkpt), number=i, width=checkpointRadius, height=checkpointRadius)
                ckptBoject.setVisible(False)
                self.viewer.addCheckpoint(ckptBoject)
            
            podImgPath = backImgPath = os.path.join(dirname, 'imgs','pod.png')
            xPod = scale*self.state[1]
            yPod = scale*self.state[3]
            podObject = pygame_rendering.Pod(podImgPath, pos=(xPod, yPod), theta=self.state[0], width=podRadius, height=podRadius)
            self.viewer.addPod(podObject)

        if self.state is None: return None
        

        for i in range(self.firstCkptIndex):
            self.viewer.checkpoints[i].setVisible(False)
        
        self.viewer.checkpoints[self.firstCkptIndex].setVisible(True)
        if self.firstCkptIndex < self.nCkpt-1:
            self.viewer.checkpoints[self.firstCkptIndex+1].setVisible(True)

        theta = self.state[0]
        xPod = scale*self.state[1]
        yPod = scale*self.state[3]

        self.viewer.pods[0].setPos((xPod, yPod))
        self.viewer.pods[0].rotate(theta)

        return self.viewer.render()

    def close(self):
        if self.viewer: self.viewer.close()
