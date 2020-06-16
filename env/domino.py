from itertools import combinations
import random as rnd
from copy import deepcopy as dpc
from trueskill import Rating, rate

DEBUG = False

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

class Player:
    def __init__(self, id: int, nMax: int, nTotalBones: int, typeAgent):
        self.id = id
        self.bones = []
        self.initialBones = []

        self.MMR = Rating()

        self.nMax = nMax
        self.nTotal = nTotalBones

        self.typeAgent = typeAgent

        self.board = []
        ZEROS = lambda : [0]*self.nTotal
        self.passBones = [ ZEROS(), ZEROS(), ZEROS() ]
        plays = [1]*(self.nMax+1)

        self.state = []
        self.state.extend( ZEROS() )
        self.state.extend( ZEROS() )
        self.state.extend( ZEROS() )
        self.state.extend( plays )


    def __str__(self):
        s = f'Player {self.id:d}:\n\t'
        for bone in self.bones: s += str(bone) + "  "
        s += '\n\t' + str( self.state )
        return s

    def printMMR(self):
        return f'Player {self.id:d}: ' + str( self.MMR )

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

    def playRandom(self, board):
        if not board :
            bone = Bone( self.nMax, self.nMax )
            self.bones.remove(bone)
            board.append(bone)
            return board, bone, len(self.bones) == 0, bone is None

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

        return board, bone, len(self.bones) == 0, bone is None

class Game:
    def __init__(self, nMax: int, nJug: int):
        self.nMax = nMax
        self.totalBones = self.nTotalBones()
        assert self.nTotalBones() % nJug == 0, "Bones cannot be deal!!!"
        self.nJug = nJug

        nBones = int(self.nTotalBones() / nJug)
        types = ['random', 'random', 'random', 'random']
        self.players = [ Player(i, nMax, self.totalBones, types[i]) for i in range(nJug) ]

        self.board = []
        self.bones = []


    def nTotalBones(self) -> int:
        n = self.nMax
        return int(0.5 * (n + 1) * (n + 2))

    def deal(self):
        numbers = range(self.nMax + 1)

        self.bones = list( combinations( numbers, 2 ) )
        for i in numbers: self.bones.append((i, i))

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
            self.board, bone, ended, playerPass = self.players[idx].playRandom( self.board )

            move = 'x' if playerPass else bone
            for p in self.players : p.update( self.board, idx, move )

            if playerPass: nPas += 1
            else: nPas = 0

            if nPas == self.nJug: ended = True

            if DEBUG:
                if playerPass : print(f'Turn {k:d}: The Player {idx:d} pass')
                else : print(f'Turn {k:d}: The Player {idx:d} plays the bone {bone}')

                self.printPlayers()
                self.printBoard()

            k += 1
            idx += 1
            idx %= self.nJug

        idx = (idx - 1) % self.nJug

        rates = [ [self.players[0].MMR], [self.players[1].MMR], [self.players[2].MMR], [self.players[3].MMR] ]
        ranks = [1]*4
        ranks[idx] =0

        (r1,), (r2,), (r3,), (r4,) = rate( rates, ranks=ranks )

        self.players[0].MMR = dpc( r1 )
        self.players[1].MMR = dpc( r2 )
        self.players[2].MMR = dpc( r3 )
        self.players[3].MMR = dpc( r4 )


game = Game(6,4)
nGames = 500
for i in range(nGames) :
    print(f'Game {(i+1):d}')
    game.play()
    for p in game.players : print( '\t' + p.printMMR() )