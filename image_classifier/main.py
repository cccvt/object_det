import os
from train_nn import train_network
from subprocess import Popen

Popen(['tensorboard --logdir={}/log'.format(os.getcwd())], shell=True)
train_network(show_results=True)
