import sys
import os
from collector import Collector
import torch
from torch.autograd import Variable
from queue import Queue
import numpy as np
import imageio

# Change your policy file here!
import dense_model as model
print("Using dense_model as policy file.")

file_name = str(sys.argv[1])
env_type = 'snake-plural-v0'
vid_name = file_name.split(".")[0]+"vid.mp4"
quota = 10 # Minimum reward to produce video

# Environment Choices
grid_size = [15,15]
n_foods = 2
n_snakes = 2
snake_size = 3
unit_size = 4
n_obs_stack = 2 # number of observations to stack for a single environment state
view_net_input = False


if len(sys.argv) > 2:
    for arg in sys.argv[2:]:
        str_arg = str(arg)
        if "grid_size=" in str_arg: grid_size= [int(str_arg[len('grid_size='):]),int(str_arg[len('grid_size='):])]
        if "n_foods=" in str_arg: n_foods= int(str_arg[len('n_foods='):])
        if "n_snakes=" in str_arg: n_snakes= int(str_arg[len('n_snakes='):])
        if "snake_size=" in str_arg: snake_size= int(str_arg[len('snake_size='):])
        if "unit_size=" in str_arg: unit_size= int(str_arg[len('unit_size='):])
        if "n_obs_stack=" in str_arg: n_obs_stack= int(str_arg[len('n_obs_stack='):])
        if "quota=" in str_arg: quota= int(str_arg[len('quota='):])
        if "env_type=" in str_arg: env_type = str_arg[len('env_type='):]
        if "vid_name=" in str_arg: vid_name = str_arg[len('vid_name='):]

print("file_name:", file_name)
print("n_obs_stack:", n_obs_stack)
print("grid_size:", grid_size)
print("n_foods:", n_foods)
print("n_snakes:", n_snakes)
print("snake_size:", snake_size)
print("unit_size:", unit_size)
print("env_type:", env_type)
print("quota:", quota)

dummy_q = Queue(5)
dummy_q.put(1)
collector = Collector(dummy_q, grid_size=grid_size, n_snakes=n_snakes, snake_size=snake_size, n_foods=n_foods, unit_size=unit_size, n_obs_stack=n_obs_stack, net=None, n_tsteps=30, gamma=0, env_type=env_type, preprocessor=model.Model.preprocess)
net = model.Model(collector.state_shape, collector.action_space, env_type=env_type, view_net_input=view_net_input)
collector.net = net
dummy = Variable(torch.ones(3,*collector.state_shape))
collector.net.forward(dummy)
collector.net.load_state_dict(torch.load(file_name))
collector.net.train(mode=False)
collector.net.req_grads(False)

vid_data = collector.get_vid_data(quota)

os.system('rm '+vid_name)
writer = imageio.get_writer(vid_name, mode='I', fps=12)
for i,img in enumerate(vid_data):
    temp = img.repeat(4, axis=0).repeat(4, axis=1)
    writer.append_data(temp)
writer.close()
