from collections import deque
from copy import deepcopy
import random
from statistics import mean
from sys import setrecursionlimit
import numpy as np
import scipy.linalg as la


class Position:
    def __init__(self, position, turn):
        self.position = list(position)
        self.turn = turn
        self.won = None
        self.parents = []
        self.children = []
        self.initialized = False
        self.distance = 100  # This variable denotes the distance from an absolutely won position (where one player is dead) to prevent looping
        if sum(self.position[0:2]) == 0:
            self.won = False
            self.distance = 0
        if sum(self.position[2:4]) == 0:
            self.won = True
            self.distance = 0

    def getValidPositions(self, cache):
        # Moves using the first pair of hands if the first player, or the 2nd pair of hands if otherwise
        offset = 0 if self.turn else 2
        # Using a set automatically handles duplicates
        possibilities = set()
        # Simulates each hand hitting each other possible hand
        for i in range(2):
            for j in range(2):
                newpos = self.position.copy()
                if newpos[2 - offset + j] == 0:
                    continue
                newpos[2 - offset + j] = (
                    newpos[2 - offset + j] + newpos[offset + i]
                ) % 5
                # Ensure ordering
                if newpos[2 - offset + 1] < newpos[2 - offset]:
                    newpos[2 - offset + 1], newpos[2 - offset] = (
                        newpos[2 - offset],
                        newpos[2 - offset + 1],
                    )
                if newpos != self.position:
                    possibilities.add(tuple(newpos))

        # merges
        for i in range(self.position[offset + 1] + 1):
            newpos = self.position.copy()
            newpos[1 + offset] -= i
            newpos[0 + offset] += i
            if newpos[1 + offset] < newpos[0 + offset]:
                newpos[0 + offset], newpos[1 + offset] = (
                    newpos[1 + offset],
                    newpos[0 + offset],
                )
            if newpos != self.position and max(newpos) < 5 and min(newpos) >= 0:
                possibilities.add(tuple(newpos))

        # Gets every child object (for recursion)
        possibility_objects = []
        for i in possibilities:
            pobject = Position(i, not self.turn)
            pobject_string = str(pobject)
            if pobject_string not in cache:
                cache[pobject_string] = pobject
            else:
                pobject = cache[pobject_string]
            possibility_objects.append(pobject)
        return possibility_objects

    def recursivelyGenerateChildren(self, cache):
        if self.won:
            return
        self.initialized = True
        # Get kids
        # Update parent with kid info
        self.children = self.getValidPositions(cache)
        # Update kids with parent info
        for child in self.children:
            child.parents.append(self)

        for child in self.children:
            if child.initialized:
                pass
            else:
                child.recursivelyGenerateChildren(cache)

    def __str__(self):
        string = ""
        for i in self.position:
            string += str(i)

        string += "1" if self.turn else "2"
        return string



setrecursionlimit(1000000)
def gengraph(damping, skip=True):
    cache = {}
    initial = Position([1, 1, 1, 1], True)
    cache[str(initial)] = initial
    initial.recursivelyGenerateChildren(cache)
    wonqueue = deque()
    for i in cache.values():
        if i.won == True or i.won == False:
            wonqueue.append(i)

    # Calculate which positions are guranteed to be won by a side, and how many moves it takes to win from such a position
    while len(wonqueue) > 0:
        item = wonqueue.popleft()
        for parent in item.parents:
            allarelost = True6.7710394
            veteranparent = parent.won is not None
            for child in parent.children:
                if child.won == parent.turn:
                    parent.won = parent.turn
                    allarelost = False
                if child.won == None:
                    allarelost = False
            if allarelost:
                parent.won = not parent.turn
            if not veteranparent and parent.won is not None:
                wonqueue.append(parent)
                parent.distance = item.distance + 1
            else:
                if item.distance < parent.distance:
                    parent.distance = item.distance + 1

    positionlist = list(cache.values())

    # Remove any position where True (the human player) is winning. The AI plays optimally, which means the human will never be winning. Thus, considering that as a possibility in the PageRank calculation will throw off the results.
    for position in positionlist:
        position.children = [x for x in position.children if not x.won == True]
    positionlist = [x for x in positionlist if not x.won == True]

    # Build a sympy matrix that is then used to rref the pagerank
    matrixA = []
    matrixC = []
    for position in positionlist:
        matrixrow = [0] * (len(positionlist))
        matrixrow[positionlist.index(position)] = 1
        matrixCval = [0]
        if position.won == False:
            matrixCval[0] = 1
        else:
            children = position.children
            childind = []
            for child in children:
                childind.append(positionlist.index(child))

            for ind in childind:
                matrixrow[ind] = -1 / len(position.children) * damping
            matrixCval = [0]
        matrixC.append(matrixCval)
        matrixA.append(matrixrow)

    A = np.array(matrixA, dtype="float")
    C = np.array(matrixC, dtype="float")
    sol = np.linalg.solve(A, C)
    oldsol = sol
    for (i, position) in enumerate(positionlist):
        position.blunderscore = sol[i][0]
    if skip:
        newpositionlist = deepcopy(positionlist)
        for position in newpositionlist:
            if position.turn == False:
                maxblunder = 0
                for child in position.children:
                    maxblunder = max(maxblunder, child.blunderscore)
                position.children = [
                    x for x in position.children if x.blunderscore == maxblunder
                ]
                if len(position.children) != 1 and position.won != False:
                    print("ALARM")
        matrixA = []
        matrixC = []
        for position in newpositionlist:
            matrixrow = [0] * (len(newpositionlist))
            matrixrow[newpositionlist.index(position)] = 1
            matrixCval = [0]
            if position.won == False:
                matrixCval[0] = 1
            else:
                children = position.children
                childind = []
                for child in children:
                    childind.append(newpositionlist.index(child))

                for ind in childind:
                    matrixrow[ind] = -1 / len(position.children) * damping
                matrixCval = [0]
            matrixC.append(matrixCval)
            matrixA.append(matrixrow)

        A = np.array(matrixA, dtype="float")
        C = np.array(matrixC, dtype="float")
        sol = np.linalg.solve(A, C)
        for (i, position) in enumerate(newpositionlist):
            cache[str(position)].blunderscore = sol[i][0]
    return cache


# player class - given a position, makes a move
class ComputerPlayer:
    def move(self, position):
        wonmoves = []
        mindepth = 1000
        for child in position.children:
            if child.won == False:
                mindepth = min(mindepth, child.distance)
                wonmoves.append(child)

        for move in wonmoves:
            if move.distance == mindepth:
                return move
        return random.choice(position.children)


# player class - given a position, makes a move
class EfficientComputerPlayer:
    def move(self, position):
        wonmoves = []
        mindepth = 1000
        maxblunder = 0
        for child in position.children:
            maxblunder = max(maxblunder, child.blunderscore)
            if child.won == False:
                mindepth = min(mindepth, child.distance)
                wonmoves.append(child)

        for move in wonmoves:
            if move.distance == mindepth:
                return move
        for child in position.children:
            if child.blunderscore == maxblunder:
                return child


class RandomPlayer:
    def move(self, position):
        return random.choice(position.children)


class HumanPlayer:
    def move(self, position):
        return random.choice(
            list(x for x in position.children if (x.distance > 6 or x.won != False))
        )


def runtests(turn, iters, cache):
    resarr = []
    initial = cache["11112"] if not turn else cache["11111"]
    initialcounter = 1 if not turn else 0
    iters = iters
    counters = []
    for i in range(iters):
        pos = initial
        counter = initialcounter
        while pos.won != False:
            if counter % 2 == 0:
                pos = RandomPlayer().move(pos)
            else:
                pos = ComputerPlayer().move(pos)
            counter += 1
        counters.append(counter)
    print(mean(counters))
    resarr.append(mean(counters))

    counters = []
    for i in range(iters):
        pos = initial
        counter = initialcounter
        while pos.won != False:
            if counter % 2 == 0:
                pos = RandomPlayer().move(pos)
            else:
                pos = EfficientComputerPlayer().move(pos)
            counter += 1
        counters.append(counter)
    print(mean(counters))
    resarr.append(mean(counters))

    counters = []
    for i in range(iters):
        pos = initial
        counter = initialcounter
        while pos.won != False:
            if counter % 2 == 0:
                pos = HumanPlayer().move(pos)
            else:
                pos = ComputerPlayer().move(pos)
            counter += 1
        counters.append(counter)
    print(mean(counters))
    resarr.append(mean(counters))

    counters = []
    for i in range(iters):
        pos = initial
        counter = initialcounter
        while pos.won != False:
            if counter % 2 == 0:
                pos = HumanPlayer().move(pos)
            else:
                pos = EfficientComputerPlayer().move(pos)
            counter += 1
        counters.append(counter)
    print(mean(counters))
    resarr.append(mean(counters))
    return resarr


resarff = []
for test in [0.1, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99, 0.999]:
    print(test)
    cache = gengraph(test)
    resarr1 = runtests(0, 10000000, cache)
    cache = gengraph(test)
    resarr2 = runtests(1, 10000000, cache)
    resarrf = []
    for i in range(4):
        resarrf.append(resarr1[i] / 2 + resarr2[i] / 2)
    resarff.append(resarrf)
    print(test)

for (i, r) in enumerate(resarff):
    print([0.1, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99, 0.999][i], ": ", r)

pos = cache["11112"]

# currentpos = initial
while False:
    print("Available moves are:")
    for ind, child in enumerate(currentpos.children):
        print(
            ind,
            " ",
            child,
            " ",
            child.won,
            " ",
            child.blunderscore,
            " ",
            child.distance,
        )
    while True:
        move = input("Enter your move (number)")
        if move == "explore":
            for i in cache[input("Enter cache compatible position string: ")].children:
                print(i, i.won, i.blunderscore, sep=" ")
        elif move == "help":
            print("Set - sets position using cache compatible position string")
            print("Explore - outputs children of a move")
            for ind, child in enumerate(currentpos.children):
                print(ind, " ", child, " ", child.won, " ", child.blunderscore)

        elif move == "set":
            newpos = cache[input("Enter cache compatible position string: ")]
            break
        else:
            try:
                newpos = currentpos.children[int(move)]
                break
            except:
                pass
    bestwinpos = None
    bestwindistance = 1000
    bestblunderscore = 0
    for child in newpos.children:
        bestblunderscore = max(bestblunderscore, child.blunderscore)
        print(child, " ", child.won, " ", child.blunderscore, " ", child.distance)
        if child.won == False:
            if bestwindistance > child.distance:
                bestwindistance = child.distance
                bestwinpos = child

    if bestwinpos is not None:
        currentpos = bestwinpos
    else:
        for child in newpos.children:
            if child.blunderscore == bestblunderscore:
                currentpos = child

    print("AI PLAYED ", currentpos, " ", currentpos.won)
    if sum(currentpos.position[0:2]) == 0:
        print("YOU LOST")
        break
