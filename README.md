# gym-coders-strike-back

This environment is an implementation of the [Coders Strike Back](https://www.codingame.com/multiplayer/bot-programming/coders-strike-back) contest, hosted by the [Codin Game](https://www.codingame.com) website.

Actually, only a single game mode is implemented,  i.e. you can control only one pod through a random set of checkpoints, and the game finish when you reach the final checkpoint.

## Installation

```bash
git clone https://github.com/alenic/gym-coders-strike-back
cd gym-coders-strike-back
python setup.py install
```

## Usage

This is an example of usage, it will simulate a single episode with the simple policy: target the checkpoint with constant thrust

```python
import gym
from gym_coders_strike_back import *
import time
import numpy as np

env = gym.make('CodersStrikeBack-v0')
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
```
