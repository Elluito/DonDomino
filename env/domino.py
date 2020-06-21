from itertools import combinations
import random as rnd
from copy import deepcopy as dpc
from trueskill import Rating, rate
from enum import Enum
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from supervised_model.Main_supervisado import build_model
from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward'))
DEBUG = False
gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# capacity=3000
if gpus :
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            print("\n\nENTRE EN LAS GPUS IN PUSE SET MEMORY GROWTH TRUE")
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=capacity*0.8)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class Bone :
    IDX = [[0, 1, 2, 3, 4, 5, 6],
           [1, 7, 8, 9, 10, 11, 12],
           [2, 8, 13, 14, 15, 16, 17],
           [3, 9, 14, 18, 19, 20, 21],
           [4, 10, 15, 19, 22, 23, 24],
           [5, 11, 16, 20, 23, 25, 26],
           [6, 12, 17, 21, 24, 26, 27]]

    def __init__(self, n1:int, n2:int):
        self.n1 = n1
        self.n2 = n2

    def __str__(self):
        return f'[{self.n1:d}|{self.n2:d}]'

    __repr__ = __str__

    def __eq__(self, value):
        if not isinstance(value, Bone): return False
        return (self.n1 == value.n1 and self.n2 == value.n2) or (self.n1 == value.n2 and self.n2 == value.n1)

    def __contains__(self, key):
        return self.n1 == key or self.n2 == key

    def inv(self):
        self.n1, self.n2 = self.n2, self.n1
        return self

    def sum(self):
        return self.n1 + self.n2

class PlayerType(Enum) :
    RANDOM = 0
    IMITATION = 1
    POLICY = 2

    def __str__(self):
        if self == PlayerType.RANDOM : return 'Random Player'
        if self == PlayerType.IMITATION: return 'Imitation Bot Player'
        if self == PlayerType.POLICY: return 'Policy with imitation Bot Player initialization'

class Player:
    def __init__(self, id: int, nMax: int, nTotalBones: int, typeAgent):
        self.id = id
        self.bones = []
        self.initialBones = []

        self.MMR = Rating()

        self.nMax = nMax
        self.nTotal = nTotalBones

        self.typeAgent = PlayerType(typeAgent)

        self.board = []
        ZEROS = lambda : [0]*self.nTotal
        self.passBones = [ ZEROS(), ZEROS(), ZEROS() ]
        plays = [1]*(self.nMax+1)

        self.state = []
        self.state.extend( ZEROS() )
        self.state.extend( ZEROS() )
        self.state.extend( ZEROS() )
        self.state.extend( plays )
        self.model = build_model((91, 1,),type="fc")
        self.model.load_weights("../models/DOMINATOR_e31-val_loss_2.2780.hdf5")


    def __str__(self):
        s = f'Player {self.id:d}:\n\t'
        for bone in self.bones: s += str(bone) + "  "
        s += '\n\t' + str( self.state )
        return s

    def printMMR(self):
        return f'Player {self.id:d} ({str(self.typeAgent)}): ' + str( self.MMR )

    def playerId(self, otherID:int ):
        tempID = otherID - self.id
        if tempID < 0 : tempID += 4
        return tempID-1

    def update(self, board, idPlayer, move ) :
        self.board = board

        a, b = board[0].n1, board[-1].n2
        plays = [0] * (self.nMax+1)
        plays[a] = 1
        plays[b] = 1

        if move == 'x' :
            idTemp = self.playerId(idPlayer)
            passTemp = dpc( self.passBones[ idTemp ] )

            for i in Bone.IDX[a] : passTemp[ i ] = 1
            for i in Bone.IDX[b] : passTemp[ i ] = 1

            self.passBones[ idTemp ] = dpc( passTemp )

        boardTemp = [0]*self.nTotal
        for b in self.board : boardTemp[ Bone.IDX[b.n1][b.n2] ] = 1

        myBones = [0]*self.nTotal
        for b in self.bones: myBones[Bone.IDX[b.n1][b.n2]] = 1

        bones1 = [int(not (m or b or p)) for m, b, p in zip(myBones, boardTemp, self.passBones[0])]
        bones2 = [int(not (m or b or p)) for m, b, p in zip(myBones, boardTemp, self.passBones[1])]
        bones3 = [int(not (m or b or p)) for m, b, p in zip(myBones, boardTemp, self.passBones[2])]

        self.state = []
        self.state.extend( bones1 )
        self.state.extend( bones2 )
        self.state.extend( bones3 )
        self.state.extend( plays )

    def addBone(self, bone) :
        self.bones.append( bone )
        self.initialBones.append(bone)

    def play(self, board,policy):
        if self.typeAgent == PlayerType.RANDOM :  return self.playRandom(board)
        if self.typeAgent == PlayerType.IMITATION:  return self.playImitation(board)
        if self.typeAgent == PlayerType.POLICY: return self.playPolicy(board,policy)

    def playRandom(self, board):
        if not board :
            bone = Bone( self.nMax, self.nMax )
            self.bones.remove(bone)
            board.append(bone)
            action = np.zeros((7, 7))
            action[6, 6] = 1
            action = action.ravel()
            return board, bone, len(self.bones) == 0, bone is None, action

        bone = None
        nJug1, nJug2 = board[0].n1, board[-1].n2
        idx1, idx2 = False, False

        for f in self.bones:
            if nJug1 in f: bone, idx1 = f, True
            if nJug2 in f: bone, idx2 = f, True
            if idx1 or idx2: break

        if bone is not None:
            self.bones.remove( bone )
            if idx1:
                if bone.n2 == nJug1:
                    board = [bone] + board
                else:
                    board = [bone.inv()] + board
            else:
                if bone.n1 == nJug2:
                    board = board + [bone]
                else:
                    board = board + [bone.inv()]
        if bone:
            action = np.zeros((7,7))
            action[bone.n1,bone.n2] = 1
            action = action.ravel()
            return board, bone, len(self.bones) == 0, bone is None,action
        else:
            action = np.zeros((7, 7))
            action = action.ravel()
            return board, bone, len(self.bones) == 0, bone is None, action



    def playImitation(self,board):
        if not board :
            bone = Bone( self.nMax, self.nMax )
            self.bones.remove(bone)
            board.append(bone)
            action = np.zeros((7,7))
            action[6,6] = 1
            action = action.ravel()
            return board, bone, len(self.bones) == 0, bone is None,action

        n1, n2 = board[0].n1, board[-1].n2

        observation = np.array( (self.state) )
        observation = observation.reshape(1,91,1)
        output = self.model.predict(observation)
        output = output.reshape(7, 7)

        # Mask
        mask1 = np.zeros((7,7), dtype = int)
        for b in self.bones :
            mask1[b.n1, b.n2] = 1
            mask1[b.n2, b.n1] = 1

        mask2 = np.zeros((7,7), dtype = int)
        for i in range(self.nMax+1) :
            mask2[n1, i] = 1
            mask2[n2, i] = 1

        output = output * mask1 * mask2
        maxV = np.amax( output )
        idxMax = np.where( output == maxV )
        action = np.zeros((7,7))
        action[idxMax]= 1
        action = action.ravel()
        idxMax = list(zip(idxMax[0],idxMax[1]))[0]

        if maxV == 0 :
            bone = None
        else :
            bone = Bone( idxMax[0], idxMax[1] )
            self.bones.remove(bone)
            if idxMax[0] == n1 : board = [bone.inv()] + board
            else : board = board + [bone]

        return board, bone, len(self.bones) == 0, bone is None,action


    def playPolicy(self,board,policy):
        if not board :
            bone = Bone( self.nMax, self.nMax )
            self.bones.remove(bone)
            board.append(bone)
            action = np.zeros((7, 7))
            action[6, 6] = 1
            action = action.ravel()
            return board, bone, len(self.bones) == 0, bone is None, action


        n1, n2 = board[0].n1, board[-1].n2

        observation = np.array( (self.state) )
        observation = observation.reshape(1,91,1)
        output = policy.model.predict(observation)
        output = output.reshape(7, 7)

        # Mask
        mask1 = np.zeros((7,7), dtype = int)
        for b in self.bones :
            mask1[b.n1, b.n2] = 1
            mask1[b.n2, b.n1] = 1

        mask2 = np.zeros((7,7), dtype = int)
        for i in range(self.nMax+1) :
            mask2[n1, i] = 1
            mask2[n2, i] = 1

        output = output * mask1 * mask2
        a=0
        while a == 0 and np.any(output)!=0:
            a = np.random.choice(output.ravel())
            temp_idx = np.nonzero(output.ravel())
            temp_x = output.ravel()[temp_idx]
            a = np.random.choice(temp_x, p=tf.nn.softmax(temp_x))

        idxMax = np.where(output == a)
        action = np.zeros((7,7))
        action[idxMax] = 1
        action = action.ravel()
        idxMax = list(zip(idxMax[0],idxMax[1]))[0]

        if a == 0 :
            bone = None
        else :
            bone = Bone( idxMax[0], idxMax[1] )
            self.bones.remove(bone)
            if idxMax[0] == n1 : board = [bone.inv()] + board
            else : board = board + [bone]

        return board, bone, len(self.bones) == 0, bone is None,action

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return np.random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
class Policy:
    # __slots__ = ( 'width', 'height', 'dim_action', 'gamma','load_name','use_prior','use_image','model','memory','epsilon','escala','mapeo','state_space','priority','priority_memory','action_space')

    def __init__(self, gamma=0.99, load_name=None,use_prior =False,use_image =False):
        # tf.enable_eager_execution()


        # tf.logging.set_verbosity(tf.logging.ERROR)
        self.priority = use_prior

        self.use_image = use_image
        self.gamma = gamma


        self.model = build_model((91,1,),mode="sup")
        self.model.summary()
        # self.model.load_weights("../models/DOMINATOR_e31-val_loss_2.2780.hdf5")
        self.optimizer = tf.keras.optimizers.Adam(lr=0.0001)
            # self.model.compile(loss=tf.compat.v1.losses.huber_loss, optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0002,momentum=0.01))


        if load_name is not None: self.model = keras.models.load_model(load_name)






        # Episode policy and reward history

    # @tf.function
    def func(self,y_true, y_pred):
        errors = tf.pow(tf.reduce_sum(y_true- y_pred, axis=1), 2)
        print(self.pesos)

        loss = tf.reduce_mean(tf.multiply(self.pesos, errors))
        return loss

    def load_Model(self, load_name=None):
        self.model.load_weights(load_name)


    def saveModel(self, name):

        self.model.save_weights('../models/' + name + '.hdf5')



    # @tf.function
    def update_policy(self,states,rewards,actions):
        if not self.priority:
                network = self.model
                reward_sum = 0
                discounted_rewards = []
                for reward in rewards[::-1]:  # reverse buffer r
                    reward_sum = reward + self.gamma * reward_sum
                    discounted_rewards.append(reward_sum)
                discounted_rewards.reverse()
                discounted_rewards = np.array(discounted_rewards)
                # standardise the rewards
                discounted_rewards -= np.mean(discounted_rewards)
                discounted_rewards /= np.std(discounted_rewards)+1e-9
                states = np.vstack(states)
                actions = np.vstack(actions)
                with  tf.GradientTape() as tape:
                    y_pred = network(states,training=True)
                    out = keras.backend.clip(y_pred,1e-8,1-1e-8)
                    log_like = actions*keras.backend.log(out)
                    advantages = tf.constant(discounted_rewards.reshape(-1,1),dtype=tf.float32)
                    loss = tf.keras.backend.sum(-log_like*advantages)

                gradients = tape.gradient([loss], network.trainable_variables)
                del tape
                self.optimizer.apply_gradients(zip(gradients, network.trainable_variables))
                # loss = network.train_on_batch(states, discounted_rewards)
                print("Loss: {}".format(loss.numpy()))





        else:
            if len(self.priority_memory) < BATCH_SIZE:
                return
            obs_batch, act_batch, rew_batch, next_obs_batch, not_done_mask, weights, indxes = self.priority_memory.sample(BATCH_SIZE,0.5)
            self.pesos = np.array(weights,dtype=np.float32)
            non_final_mask = np.where(not_done_mask==0)[0]
            act_batch = np.array([list(range(len(act_batch))), act_batch]).transpose()
            next_state_values = np.zeros([BATCH_SIZE], dtype=float)
            next_state_values[non_final_mask] = np.max(self.model.predict(next_obs_batch[non_final_mask]), axis=1)

            rew_batch = (rew_batch - np.mean(rew_batch)) / (np.std(rew_batch) + 0.001)
            # rew_batch = rew_batch/max(np.abs(rew_batch))

            q_update = (rew_batch + self.gamma * next_state_values)
            q_values = np.array(self.model.predict([obs_batch]))
            q_values[act_batch[:, 0], act_batch[:, 1]] = q_update

            with tf.GradientTape() as tape:
                # tape.watch(self.model.trainable_variables)
                y_pred = self.model([obs_batch],training=True)


                errors = tf.pow(tf.reduce_sum(q_values-y_pred,axis=1),2)

                loss = tf.reduce_mean(tf.multiply(weights,errors))
                loss = tf.reduce_mean(errors)

            grads = tape.gradient(loss, self.model.trainable_variables)
            del tape

            # grads = self.optimizer.compute_gradients(f,self.model.trainable_variables)


            # for i,elem in enumerate(grads):
            #     grads[i] =elem[1].numpy()

            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))



            # salidas = self.model.fit(obs_batch, q_values, batch_size=len(q_values), epochs=20, verbose=0)
            # print(salidas.history["loss"])
            td_error = self.model.predict([obs_batch])[act_batch[:, 0], act_batch[:, 1]]-q_update
            self.priority_memory.update_priorities(indxes, abs(td_error))
    def add_transition(self,*args):
        self.episodic_memory.append(Transition(*args))


class Game:
    def __init__(self, nMax: int, nJug: int):
        self.nMax = nMax
        self.totalBones = self.nTotalBones()
        assert self.nTotalBones() % nJug == 0, "Bones cannot be deal!!!"
        self.nJug = nJug

        nBones = int(self.nTotalBones() / nJug)
        types = [0,2,0,0]
        self.players = [ Player(i, nMax, self.totalBones, types[i]) for i in range(nJug) ]

        self.board = []
        self.bones = []
        self.policy = Policy()
        self.transitions = [[] for _ in range(nJug)]


    def nTotalBones(self) -> int:
        n = self.nMax
        return int(0.5 * (n + 1) * (n + 2))

    def deal(self):
        numbers = range(self.nMax + 1)

        self.bones = list( combinations( numbers, 2 ) )
        for i in numbers: self.bones.append((i, i))
        rnd.shuffle(self.bones)

        jugIdx = [ i % self.nJug for i in range( len( self.bones ) ) ]
        rnd.shuffle( jugIdx )

        for bone, id in zip( self.bones, jugIdx ):
            b = Bone( bone[0], bone[1] )
            self.players[id].addBone( b )

    def reset(self):
        self.board = []
        self.bones = []
        for player in self.players: player.bones = []

    def printPlayers(self):
        for p in self.players: print( p )

    def printBoard(self):
        s = f'Board:\n\t'
        for b in self.board: s += str( b ) + "  "
        print(s + "\n")

    def play(self):
        self.reset()

        self.deal()
        ended = False

        idx = -1
        for i, p in enumerate( self.players ):
            if Bone( self.nMax, self.nMax ) in p.bones:
                idx = i
                break

        nPass, k = 0, 1

        while not ended :
            # board, bone, len(self.bones) == 0, bone is None
            self.board, bone, ended, playerPass,action = self.players[idx].play( self.board,self.policy)

            move = 'x' if playerPass else bone




            for p in self.players : p.update( self.board, idx, move )

            if playerPass: nPass += 1
            else: nPass = 0

            if nPass == self.nJug:
                ended = True
                #Esto es para que el update de cuando estan
                s0 = np.sum([b.sum() for b in self.players[0].bones])
                s1 = np.sum([b.sum() for b in self.players[1].bones])
                s2 = np.sum([b.sum() for b in self.players[2].bones])
                s3 = np.sum([b.sum() for b in self.players[3].bones])
                idx = np.argmin([s0, s1, s2, s3])
                observation = np.array((self.players[idx].state))
                observation = observation.reshape(1, 91, 1)
                self.transitions[idx].append(Transition(observation, action, 1))
                for i in range(self.nJug):

                    if i != idx and len(self.transitions[i]) != 0:
                        state = self.transitions[i][-1].state
                        action = self.transitions[i][-1].action
                        reward = -1
                        self.transitions[i][-1] = Transition(state, action, reward)


            if move != 'x' and not ended:
                observation = np.array((self.players[idx].state))
                observation = observation.reshape(1, 91, 1)
                self.transitions[idx].append(Transition(observation, action, 0))

            if move!='x' and ended:
                observation = np.array((self.players[idx].state))
                observation = observation.reshape(1, 91, 1)
                self.transitions[idx].append(Transition(observation,action,1))
                for i in range(self.nJug):

                    if i!= idx and len(self.transitions[i])!=0 :
                        state = self.transitions[i][-1].state
                        action = self.transitions[i][-1].action
                        reward=-1
                        self.transitions[i][-1] = Transition(state, action, reward)

            if DEBUG:
                if playerPass : print(f'Turn {k:d}: The Player {idx:d} pass')
                else : print(f'Turn {k:d}: The Player {idx:d} plays the bone {bone}')

                self.printPlayers()
                self.printBoard()

            k += 1
            idx += 1
            idx %= self.nJug
        for trans in self.transitions:
            if trans:
                batch = Transition(*zip(*trans))
                states = batch.state
                actions = batch.action
                rewards = batch.reward
                self.policy.update_policy(states,rewards,actions)
        self.transitions=[[] for _ in range(self.nJug)]
        idx = (idx - 1) % self.nJug
        locked = True



        if nPass < self.nJug :
            print(f'\tPlayer {idx:d} wins!!!!')

            locked = False
        else :
            s0 = np.sum([b.sum() for b in self.players[0].bones])
            s1 = np.sum([b.sum() for b in self.players[1].bones])
            s2 = np.sum([b.sum() for b in self.players[2].bones])
            s3 = np.sum([b.sum() for b in self.players[3].bones])
            idx = np.argmin( [s0,s1,s2,s3] )

            locked = True

            print( f'\tGame Locked :(. Player {idx:d} wins!!!!' )

        rates = [[self.players[0].MMR], [self.players[1].MMR], [self.players[2].MMR], [self.players[3].MMR]]
        ranks = [1] * 4
        ranks[idx] = 0

        (r1,), (r2,), (r3,), (r4,) = rate(rates, ranks=ranks)

        self.players[0].MMR = dpc(r1)
        self.players[1].MMR = dpc(r2)
        self.players[2].MMR = dpc(r3)
        self.players[3].MMR = dpc(r4)

        return idx, locked


game = Game(6,4)
nGames = 100

wins = [0,0,0,0]
winsL = [0,0,0,0]
game.policy.load_Model("../models/modelo.h5")
for i in range(nGames) :
    print(f'Game {(i+1):d}')
    idxWin, locked = game.play()

    if locked : winsL[idxWin] += 1
    else: wins[idxWin] += 1
    for p in game.players : print( '\t' + p.printMMR() )

    if i%100==0:
        game.policy.saveModel("modelo")

for i,(w,l) in enumerate( zip(wins,winsL) ) : print(f'Player {i:d} wins {w:d} games and wins {l:d} locked games. Total wins: {w+l:d}')