import numpy as np
from copy import deepcopy as dpc
import matplotlib.pyplot as plt
import random as rnd

import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib import eager as tfe

'''
def update_policy( policy, optimizer ):
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    # if len(rewards) > 1 : rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []
'''

class Policy :

    def __init__(self,dim_state,dim_action,gamma=0.9):
        tf.enable_eager_execution()
        tf.logging.set_verbosity(tf.logging.ERROR)

        self.state_space = dim_state
        self.action_space = dim_action

        self.gamma = gamma

        self.global_step = tf.Variable(0)
        self.loss_avg = tfe.metrics.Mean()

        self.model = keras.Sequential( [ 
            keras.layers.Dense( 128, activation=tf.nn.relu, use_bias=False, input_shape=( self.state_space, ) ),
            keras.layers.Dropout( rate=0.6 ),
            keras.layers.Dense( self.action_space, activation=tf.nn.softmax )
        ] )

        self.optimizer = tf.train.AdamOptimizer( )

        # Episode policy and reward history
        # self.policy_history = Variable(torch.Tensor())
        # self.reward_episode = []

        # Overall reward and loss history
        # self.reward_history = []
        self.loss_history = []

    def update_policy_supervised( self, states, actions ) :

        epochs = 5
        for e in range(epochs) :
            # Calculate Loss
            with tf.GradientTape() as tape:
                actions_ = self.model( states )
                loss = tf.losses.sparse_softmax_cross_entropy( labels=actions, logits=actions_ )
                grads = tape.gradient( loss, self.model.trainable_variables )

            self.optimizer.apply_gradients( zip( grads, self.model.trainable_variables ), self.global_step )
            
            self.loss_history.append( loss.numpy() )

            print( f'\tEpoch {e+1:d}/{epochs}... | Loss: {loss:.3f}' )        


