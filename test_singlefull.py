import gym
from gym_coders_strike_back import *
import time
import numpy as np


# TODO
'''
env = gym.make('CodersStrikeBackFull-v0')
fps =  env.metadata.get('video.frames_per_second')

totalReward = 0.0
# Set pseudorandom seed for the getting the same game
env.seed(1234)
state = env.reset()
for i in range(1,10000):
    # display the game
    render = env.render()
    if not render:
        break
    # Take action (simple policy)
    targetX, targetY = state[5:7]
    thrust = 60
    action = np.array([targetX, targetY, thrust], dtype=np.float32)
    # Do a game step
    state, reward, done, _ = env.step(action)

    # Print the state
    
    print('---------- Tick %d' % i)
    print('Pod angle ', state[0])
    print('Pod (x,y): (%d, %d)' % (state[1], state[3]))
    print('Pod velocity (v_x,v_y): (%d, %d)' % (state[2], state[4]))
    print('First Checkpoint (x,y): (%d, %d)' % (state[5],state[6]))
    print('Second Checkpoint (x,y): (%d, %d)' % (state[7],state[8]))
    print('Reward ', reward)
    print('done ', done)

    totalReward += reward
    # Slow down the cycle for a realistic simulation
    time.sleep(1.0/fps)

    # The game end if the flag done is True
    if done:
        break

print("Total reward: ", totalReward)
env.close()
'''