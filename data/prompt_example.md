
### Prompt 1

You will be given a grid world environment and you need to find a path from the agent position to the goal position. The grid world is represented as a 2D array, where . represents an empty cell, # represents an obstacle, A represents the agent and G represents the goal. You can move up, down, left, or right. Your task is to provide a sequence of comma-separated actions (up, down, left, right) that lead to the goal. Don't overthink it. 

```
# . . G .
. . . . #
# # . . .
. . . . .
A . . # .
```


### Prompt 2 (single step)

You are navigating a grid world maze. In this grid:
- 'A' is your current position
- 'G' is the goal you need to reach
- '#' represents obstacles (you cannot move through these)
- '.' represents open spaces (you can move through these)

The grid uses (row, column) coordinates where (0,0) is the top-left corner.
Rows increase going down, columns increase going right.

Your task:
1. First, identify your current position (row, column) and the goal position (row, column)
2. Consider which direction (UP, DOWN, LEFT, RIGHT) brings you closer to the goal
3. Choose ONE valid move (you cannot move through obstacles or off the grid)
4. Show the updated grid with your new position

Example:
Input grid:
```
. . A .
. # . .
. . G .
```

Reasoning:
- My position is at (0,2)
- The goal is at (2,2)
- The goal is below me, so I should move DOWN
- I can move DOWN as there's no obstacle

Move: DOWN

Updated grid:
```
. . . .
. # A .
. . G .
```

Now, navigate this maze:

```
# . . G .
. . . . #
# # . . .
. . . . .
A . . # .
```