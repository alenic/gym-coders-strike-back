# gym-coders-strike-back

This environment is an implementation of the [Coders Strike Back](https://www.codingame.com/multiplayer/bot-programming/coders-strike-back) contest, hosted by the [Codin Game](https://www.codingame.com) website.

Actually, only a single game mode is implemented,  i.e. you can control only one pod through a random set of checkpoints, and the game finish when you reach the final checkpoint.

## Installation

```bash
git clone https://github.com/alenic/gym-coders-strike-back
cd gym-coders-strike-back
pip setup.py install
```

## Usage

This is an example of usage, it will simulate a single episode with the simple policy: target the checkpoint with constant thrust

```python
import gym
from gym_coders_strike_back import *
import time
import numpy as np

targetX = 0
targetY = 0

env = gym.make('CodersStrikeBack-v0')
fps =  env.metadata.get('video.frames_per_second')
env.reset()
for i in range(1,10000):
    env.render()
    # Take action
    thrust = 50
    action = np.array([targetX, targetY, thrust], dtype=np.float32)
    # Step
    state, reward, done, _ = env.step(action)

    # Print
    targetX, targetY = state[5:7]
    print('---------- Tick %d' % i)
    print('angle ', state[0])
    print('X ', state[1])
    print('Y ', state[3])
    print('vX ', state[2])
    print('vY ', state[4])
    print('reward ', reward)
    print('done ', done)

    # Slow down the cycle for a realistic simulation
    time.sleep(1.0/fps)

    # The game end if the flag done is True
    if done:
        break

env.close()
```
