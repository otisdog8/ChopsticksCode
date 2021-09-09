# Chopsticks
Cool little project I made for the finger game chopsticks

You play first

If you want to play against the AI with the AI playing first, type set `enter` then 11112 `enter`

First two numbers are your hands, second two are the computer's. Last number is the turn (1 for player 2 for computer)

The numbers describing the hands are always sorted (for example, 10111 is impossible, and would be represented by 01111)

The True/False being displayed determines if there is a forced win for either side (True means player, False means computer)

The number is essentially a "blunderscore", it determines the optimal move assuming people are moving randomly. This is used by the computer to choose moves during "drawn" positions