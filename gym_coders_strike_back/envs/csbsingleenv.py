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


class CodersStrikeBackSingle(gym.Env):
    metadata = {"render.modes": ["human"], "video.frames_per_second": 30}

    def __init__(self):
        # Game constants
        self.maxThrust = 200.0
        self.maxSteeringAngle = (
            0.1 * np.pi
        )  # Maximum steering angle in radians [+-0.1 rad = +-15 Deg]
        self.friction = 0.85
        self.__M_PI2 = 2.0 * np.pi

        self.seed()
        self.viewer = None

        self.gamePixelWidth = 16000
        self.gamePixelHeight = 9000
        self.nCkpt = 2
        self.sampled = False
        self._checkpoint_radius = 600
        self._checkpoint_radius2 = self._checkpoint_radius**2
        # State = [theta, x, vx, y, vy, target1_x, target1_y, target2_x, target2_y]
        self.state = None
        self._theta = None
        self._x = None
        self._y = None
        self._vx = None
        self._vy = None
        self._target1_x = None
        self._terget1_y = None
        self._target2_x = None
        self._terget2_y = None

        self._x_prev = None
        self._y_prev = None
        self.steps_beyond_done = None

        minPos = -200000.0
        maxPos = 200000.0
        minVel = -600.0
        maxVel = 600.0
        # action_space = [targetX, targetY, thrust]
        self.action_space = spaces.Box(
            low=np.array([minPos, minPos, 0.0]),
            high=np.array([maxPos, maxPos, self.maxThrust]),
            dtype=np.float64
        )

        self.observation_space = spaces.Box(
            np.array([0.0, minPos, minVel, minPos, minVel, 0, 0, 0, 0]),
            np.array(
                [self.__M_PI2, maxPos, maxVel, maxPos, maxVel, 16000, 9000, 16000, 9000]
            ),
            dtype=np.float64
        )

        self.totalReward = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getAngle(self, target):
        # Get the angle [0,2*pi] of the vector going from pod's position to a target
        d = np.array([target[0] - self._x, target[1] - self._y])
        norm_d = np.linalg.norm(d)

        angle = math.acos(d[0] / (norm_d+1e-16))

        if d[1] < 0:
            angle = self.__M_PI2 - angle

        return angle

    def getDeltaAngle(self, target):
        # Get the minimum delta angle needed by the pod to reach the target
        angle = self.getAngle(target)

        # Positive amount of angle to reach the target turning to the right of the pod
        right = (
            angle - self._theta
            if self._theta <= angle
            else self.__M_PI2 - self._theta + angle
        )
        # Positive amount of angle to reach the target turning to the left of the pod
        left = (
            self._theta - angle
            if self._theta >= angle
            else self._theta + self.__M_PI2 - angle
        )

        # Get the minimum delta angle (positive in right, negative in left)
        if right < left:
            return right
        else:
            return -left

    def checkpointCollision(self, ckpt_pos):
        pos2 = np.array([self._x, self._y])

        cp2 = ckpt_pos - pos2
        if cp2[0]*cp2[0] + cp2[1]*cp2[1] < self._checkpoint_radius2:
            return True
        
        pos1 = np.array([self._x_prev, self._y_prev])
        p21 = pos2 - pos1
        cp1 = ckpt_pos - pos1
        
        norm2_p21 = p21[0]*p21[0] + p21[1]*p21[1]
        norm2_cp1 = cp1[0]*cp1[0] + cp1[1]*cp1[1]

        if norm2_cp1 >= norm2_p21:
            return False
        
        dot_p = np.dot(p21, cp1)
        if dot_p < 0:
            return False
        
        # compute minimum distance point from checkpoint
        norm_p21 = np.sqrt(norm2_p21)
        min_dist_p = pos1 + dot_p/norm_p21
        
        return min_dist_p[0]*min_dist_p[0] + min_dist_p[1]*min_dist_p[1] < self._checkpoint_radius2



    def setState(self):
        self.state = np.array([ self._theta,
                                self._x,
                                self._vx,
                                self._y,
                                self._vy,
                                self._target1_x,
                                self._target1_y,
                                self._target2_x,
                                self._target2_y,
                               ])

    def sample(self):
        self.sampled = True
        # State = [theta, x, xdot, y, ydot, firstCkptX, firstCkptY, secondCkptX, secondCkptY]
        self.theta0 = self.np_random.randint(0, 359) * np.pi / 180.0
        self.x0 = self.np_random.randint(0, self.gamePixelWidth)
        self.vx0 = self.np_random.randint(-100, 100)
        self.y0 = self.np_random.randint(0, self.gamePixelHeight)
        self.vy0 = self.np_random.randint(-100, 100)

        # sample two checkpoints with a minimum distance between them
        while True:
            ckpt1_x = self.np_random.randint(0, self.gamePixelWidth)
            ckpt1_y = self.np_random.randint(0, self.gamePixelHeight)
            ckpt2_x = self.np_random.randint(0, self.gamePixelWidth)
            ckpt2_y = self.np_random.randint(0, self.gamePixelHeight)
            diff = np.array([ckpt1_x - ckpt2_x, ckpt1_y - ckpt2_y])
            if np.linalg.norm(diff) > 1000:
                break

        self.ckpt1_x0 = ckpt1_x
        self.ckpt1_y0 = ckpt1_y
        self.ckpt2_x0 = ckpt2_x
        self.ckpt2_y0 = ckpt2_y

        # Checkpoint coordinates
        self.checkpoint = np.array(
            [[self.ckpt1_x0, self.ckpt1_y0], [self.ckpt2_x0, self.ckpt2_y0]]
        )

    def reset(self):
        if not self.sampled:
            self.sample()

        # Get
        self.firstCkptIndex = 0
        # State init
        self._theta = self.theta0
        self._x = self.x0
        self._vx = self.vx0
        self._y = self.y0
        self._vy = self.vy0
        self._target1_x = self.ckpt1_x0
        self._target1_y = self.ckpt1_y0
        self._target2_x = self.ckpt2_x0
        self._target2_y = self.ckpt2_y0
        self._x_prev = None
        self._y_prev = None

        self.tick_from_last_ckpt = 100
        self.steps_beyond_done = None
        self.totalReward = 0
        self.viewer = None

        self.setState()
        return self.state

    # Game dynamics
    def movePod(self, targetX, targetY, thrust):
        da = self.getDeltaAngle(np.array([targetX, targetY]))

        # Saturate delta angle
        da = max(-self.maxSteeringAngle, min(self.maxSteeringAngle, da))

        self._theta += da

        if self._theta >= self.__M_PI2:
            self._theta -= self.__M_PI2

        if self._theta < 0.0:
            self._theta += self.__M_PI2

        # Update dynamics
        self._vx += math.cos(self._theta) * thrust
        self._vy += math.sin(self._theta) * thrust

        self._x = round(self._x + self._vx)
        self._y = round(self._y + self._vy)

        # Apply the friction
        self._vx = int(0.85 * self._vx)
        self._vy = int(0.85 * self._vy)


    def step(self, action):
        self._x_prev = self._x
        self._y_prev = self._y
        # Saturate thrust
        action[2] = np.clip(action[2], 0, self.maxThrust)

        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        self.movePod(action[0], action[1], action[2])

        done = False
        reward = 0.0

        if self.checkpointCollision(np.array([self._target1_x, self._target1_y])):
        #if np.linalg.norm(playerPos - firstCkptPos) < self._checkpoint_radius:
            self.tick_from_last_ckpt = 100
            ckptIndex = self.firstCkptIndex
            if ckptIndex == self.nCkpt - 1:
                # Last checkpoint reached
                reward = 1.0
                done = True
            elif ckptIndex == self.nCkpt - 2:
                # There is only the last checkpoint
                ckptIndex += 1
                self._target1_x, self._target1_y = self.checkpoint[ckptIndex]
                self._target2_x, self._target2_y = self._target1_x, self._target1_y
                reward = 1.0
            else:
                ckptIndex += 1
                self._target1_x, self._target1_y = self.checkpoint[ckptIndex]
                self._target2_x, self._target2_y = self.checkpoint[ckptIndex + 1]
                reward = 1.0
            

            self.firstCkptIndex = ckptIndex

        self.tick_from_last_ckpt -= 1
        if self.tick_from_last_ckpt <= 0:
            done = True

        self.totalReward += reward

        if done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                if self.steps_beyond_done == 0:
                    logger.warn(
                        "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_done += 1
                reward = 0.0

        
        self.setState()
        return self.state, reward, done, {}

    def render(self, mode="human"):
        screen_width = 640
        screen_height = 360
        scale = screen_width / self.gamePixelWidth
        podRadius = scale * 400 * 2.0
        checkpointRadius = scale * 600 * 2.0

        if self.viewer is None:
            from gym_coders_strike_back.envs import pygame_rendering

            self.viewer = pygame_rendering.Viewer(screen_width, screen_height)

            dirname = os.path.dirname(__file__)
            backImgPath = os.path.join(dirname, "imgs", "back.png")
            self.viewer.setBackground(backImgPath)

            ckptImgPath = backImgPath = os.path.join(dirname, "imgs", "ckpt.png")

            self.checkpointCircle = []
            for i in range(self.nCkpt):
                xCkpt = scale * self.checkpoint[i][0]
                yCkpt = scale * self.checkpoint[i][1]
                ckptBoject = pygame_rendering.Checkpoint(
                    ckptImgPath,
                    pos=(xCkpt, yCkpt),
                    number=i,
                    width=checkpointRadius,
                    height=checkpointRadius,
                )
                ckptBoject.setVisible(False)
                self.viewer.addCheckpoint(ckptBoject)

            podImgPath = backImgPath = os.path.join(dirname, "imgs", "pod.png")
            xPod = scale * self._x
            yPod = scale * self._y
            podObject = pygame_rendering.Pod(
                podImgPath,
                pos=(xPod, yPod),
                theta=self._theta,
                width=podRadius,
                height=podRadius,
            )
            self.viewer.addPod(podObject)

            text = pygame_rendering.Text(
                "Reward", backgroundColor=(0, 0, 0), pos=(0, 0)
            )
            self.viewer.addText(text)

        if self.state is None:
            return None

        for i in range(self.nCkpt):
            self.viewer.checkpoints[i].setVisible(False)

        self.viewer.checkpoints[self.firstCkptIndex].setVisible(True)
        if self.firstCkptIndex < self.nCkpt - 1:
            self.viewer.checkpoints[self.firstCkptIndex + 1].setVisible(True)

        self.viewer.pods[0].setPos((scale*self._x, scale*self._y))
        self.viewer.pods[0].rotate(self._theta)

        self.viewer.text.setText("Tot Reward: %.2f" % self.totalReward)

        return self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
