"""
Coders Strike Back Single mode environment
author: Alessandro Nicolosi
url: https://github.com/alenic/gym-coders-strike-back
Original game: https://www.codingame.com/multiplayer/bot-programming/coders-strike-back
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import os
from .csbsingleenv import CodersStrikeBackSingle


class CodersStrikeBackSingleFull(CodersStrikeBackSingle):
    metadata = {"render.modes": ["human"], "video.frames_per_second": 30}

    def __init__(self):
        super(CodersStrikeBackSingleFull, self).__init__()
        # Game constants
        self.maxThrust = 200.0
        self.maxSteeringAngle = (
            0.1 * np.pi
        )  # Maximum steering angle in radians [+-0.1 rad = +-15 Deg]
        self.friction = 0.85
        self.__M_PI2 = 2.0 * np.pi

        self.seed()
        self.viewer = None
        # State = [theta, x, xdot, y, ydot, firstCkptX, firstCkptY, secondCkptX, secondCkptY]
        self.state = None
        self.steps_beyond_done = None

        minPos = -200000.0
        maxPos = 200000.0
        minVel = -600.0
        maxVel = 600.0
        # action_space = [targetX, targetY, thrust]
        self.action_space = spaces.Box(
            low=np.array([minPos, minPos, 0.0]),
            high=np.array([maxPos, maxPos, self.maxThrust]),
        )

        self.observation_space = spaces.Box(
            np.array([0.0, minPos, minVel, minPos, minVel, 0, 0, 0, 0]),
            np.array(
                [self.__M_PI2, maxPos, maxVel, maxPos, maxVel, 16000, 9000, 16000, 9000]
            ),
        )

        self.totalReward = 0

    def sample(self):
        self.sampled = True
        # State = [theta, x, xdot, y, ydot, firstCkptX, firstCkptY, secondCkptX, secondCkptY]
        self.theta0 = self.np_random.randint(0, 359) * np.pi / 180.0
        self.x0 = self.np_random.randint(0, self.gamePixelWidth)
        self.vx0 = self.np_random.randint(-100, 100)
        self.y0 = self.np_random.randint(0, self.gamePixelHeight)
        self.vy0 = self.np_random.randint(-100, 100)

        # Checkpoint coordinates
        mGrid = 4
        nGrid = 8
        self.gamePixelWidth = 16000
        self.gamePixelHeight = 9000
        self.nCkpt = mGrid * nGrid
        self.checkpoint = np.zeros((self.nCkpt, 2))
        # Get
        ckptList = np.arange(self.nCkpt)
        self.np_random.shuffle(ckptList)
        ckptGrid = ckptList.reshape(mGrid, nGrid)

        for i in range(mGrid):
            for j in range(nGrid):
                index = ckptGrid[i][j]
                self.checkpoint[index][0] = int(
                    (j + 0.5) * self.gamePixelWidth // nGrid
                )
                self.checkpoint[index][1] = int(
                    (i + 0.5) * self.gamePixelHeight // mGrid
                )

        self.ckpt1_x0 = self.checkpoint[0][0]
        self.ckpt1_y0 = self.checkpoint[0][1]
        self.ckpt2_x0 = self.checkpoint[1][0]
        self.ckpt2_y0 = self.checkpoint[1][1]
