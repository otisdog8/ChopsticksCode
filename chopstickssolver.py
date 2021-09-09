from collections import deque
from random import randint
import numpy as np
import scipy.linalg as la


class Position:
    def __init__(self, position, turn, cache):
        self.position = list(position)
        self.turn = turn
        self.cache = cache
        self.won = None
        self.parents = []
        self.children = []
        self.initialized = False
        self.distance = 100  # added this later to prevent looping
        if sum(self.position[0:2]) == 0:
            self.won = False
            self.distance = 0
        if sum(self.position[2:4]) == 0:
            self.won = True
            self.distance = 0
        self.rationalmoves = []
        self.blunderscore = 0

    def getValidPositions(self):
        # Make all possible things
        offset = 0 if self.turn else 2
        # hits
        possibilities = set()
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

        possibility_objects = []
        for i in possibilities:
            pobject = Position(i, not self.turn, self.cache)
            pobject_string = str(pobject)
            if pobject_string not in self.cache:
                self.cache[pobject_string] = pobject
            else:
                pobject = self.cache[pobject_string]
            possibility_objects.append(pobject)
        return possibility_objects

    def recursivelyGenerateChildren(self):
        if self.won:
            return
        self.initialized = True
        # Get kids
        # Update parent with kid info
        self.children = self.getValidPositions()
        # Update kids with parent info
        for child in self.children:
            child.parents.append(self)

        for child in self.children:
            if child.initialized:
                pass
            else:
                child.recursivelyGenerateChildren()

    def calculateRationalMoves(self):
        for child in self.children:
            if child.won == (not self.turn):
                pass
            else:
                self.rationalmoves.append(child)

    def __str__(self):
        string = ""
        for i in self.position:
            string += str(i)

        string += "1" if self.turn else "2"
        return string


cache = {}
initial = Position([1, 1, 1, 1], True, cache)
cache[str(initial)] = initial
initial.recursivelyGenerateChildren()
wonqueue = deque()
for i in cache.values():
    if i.won == True or i.won == False:
        wonqueue.append(i)

# Calculate won
# Make a queue with all won
# Go up one parent
# Run win calculation
while len(wonqueue) > 0:
    item = wonqueue.popleft()
    for parent in item.parents:
        allarelost = True
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

# Calculate wonscore
for i in cache.values():
    i.calculateRationalMoves()

positionlist = list(cache.values())

# Build a sympy matrix that is then used to rref the pagerank
matrix = []
for position in positionlist:
    children = position.children
    childind = []
    for child in children:
        childind.append(positionlist.index(child))
    matrixrow = [0] * (len(positionlist) + 1)
    matrixrow[positionlist.index(position)] = 1
    for ind in childind:
        matrixrow[ind] = -1
    matrixrow[-1] = (
        (0.15 if position.won else -0.15)
        if position.won == True or position.won == False
        else 0
    )
    matrix.append(matrixrow)


A = np.array(matrix, dtype="float")
(_, rref) = la.qr(A)

minscore = 100
maxscore = -100
for i, r in enumerate(rref):
    maxscore = max(maxscore, r[-1])
    minscore = min(minscore, r[-1])
    positionlist[i].blunderscore = round(r[-1] * 100000) / 100000

currentpos = initial
while True:
    print("Available moves are:")
    for ind, child in enumerate(currentpos.children):
        print(ind, " ", child, " ", child.won, " ", child.blunderscore)
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
    for child in newpos.children:
        print(child, " ", child.won, " ", child.blunderscore)
        if child.won == False:
            if bestwindistance > child.distance:
                bestwindistance = child.distance
                bestwinpos = child
    if bestwinpos is not None:
        currentpos = bestwinpos
    else:
        weightscore = -1 * minscore + (maxscore + minscore) * 0.01
        weights = []
        iweights = []
        for child in newpos.children:
            if child.won != True:
                weights.append((child, 100 / (weightscore + child.blunderscore)))
                iweights.append(100 / (weightscore + child.blunderscore))
        num = randint(0, round(sum(iweights)))
        total = 0
        for (child, weight) in weights:
            if total <= num and num <= total + weight:
                currentpos = child
                break
            else:
                total += weight
    print("AI PLAYED ", currentpos, " ", currentpos.won)
    if sum(currentpos.position[0:2]) == 0:
        print("YOU LOST")
        break
