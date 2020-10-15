import gym
import gym_pysc2
import sys
from absl import flags

FLAGS = flags.FLAGS
FLAGS(['hello_world.py'])
env = gym.make("SC2MoveToBeacon-v0")
env.reset()