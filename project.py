import numpy as np
import heapq
from queue import PriorityQueue
from collections import deque 
import sys

frontier = PriorityQueue()
visited = np.array([])
count = 0

class Problem:
    initial_state = []
    goal_state = np.array([[1,2,3],[4,5,6],[7,8,0]])
    def getGoal(self):
        return self.goal_state
    def getInitial(self):
        return self.initial_state
    def setInitial(self):
        startInput=input("Type \"1\" to use a default puzzle, or \"2\" to enter your own puzzle.")
        if startInput=="1":
            self.initial_state = np.array([[1,2,3],[4,8,0],[7,6,5]])
        else :
            print("Enter your puzzle, use a zero to represent the blank")
            row=input("Enter the first row, use space or tabs between numbers : ")
            self.initial_state.append(row.split(' '));
            row=input("Enter the second row, use space or tabs between numbers : ")
            self.initial_state.append(row.split(' '));
            row=input("Enter the third row, use space or tabs between numbers : ")
            self.initial_state.append(row.split(' ')); 

class Node:  
    matrix = []
    def __init__(self):
        self.matrix = []
        self.matrix_goal = []
        self.child = None
        self.parent = None
        self.gval = 0
        self.hval = 0
        self.fval = 0
    def __init__(self, mat):
        self.matrix = mat
        self.matrix_goal = []
        self.child = None
        self.parent = None
        self.gval = 0
        self.hval = 0    
        self.fval = 0
    def __lt__(self, other):
        return (self.fval < other.fval)
    # def __leq__(self, other):
    #     return ((self.fval) <= (other.fval))
    def __repr__(self):
        return self.matrix 
    def firstChild(self, mat):
        first_child = Node(mat)
    def addChild(self, child_mat, cur_node):
        child_node = Node(child_mat)
        cur_node.child = child_node
        child_node.parent = cur_node
        child_node.matrix_goal = self.matrix_goal
    def getChild(self):
        if (self.child!=None):
            return self.child;
    def setGoal(self, matrix):
        self.matrix_goal = matrix
    def updateHval(self, algorithm):
        if (algorithm == "1"):
            self.hval = 0
        elif (algorithm == "2"):
            self.hval = missedTiles(self.matrix, self.matrix_goal)
        elif (algorithm == "3"):
            self.hval = euclidean(self.matrix, self.matrix_goal)
    def updateFval(self):
        self.fval = self.gval + self.hval
#possible moves        
def left(cur_node, algorithm):
    row, col = findBlank(cur_node)[0]
    if (col > 0):
        newMat = cur_node.matrix.copy()
        newMat[row, col] = newMat[row, col-1]
        newMat[row, col-1] = 0
        cur_node.addChild(newMat, cur_node)
        child = cur_node.child
        child.gval = cur_node.gval + 1
        child.updateHval(algorithm)
        child.updateFval()
        frontier.put(child, child.fval)
        return child
        
        #siblings.append(cur_node.getChild())

def right(cur_node, algorithm):
    row, col = findBlank(cur_node)[0]
    if (col < 2):
        newMat = cur_node.matrix.copy()
        newMat[row, col] = newMat[row, col+1]
        newMat[row, col+1] = 0
        cur_node.addChild(newMat, cur_node)
        child = cur_node.child
        child.gval = cur_node.gval + 1
        child.updateHval(algorithm)
        child.updateFval()
        frontier.put(child, child.fval)
        return child

def down(cur_node, algorithm):
    row, col = findBlank(cur_node)[0]
    if (row < 2):
        newMat = cur_node.matrix.copy()
        newMat[row, col] = newMat[row+1, col]
        newMat[row+1, col] = 0
        cur_node.addChild(newMat, cur_node)
        child = cur_node.child
        child.gval = cur_node.gval + 1
        child.updateHval(algorithm)
        child.updateFval()
        frontier.put(child, child.fval)
        return child
        
def up(cur_node, algorithm):
    row, col = findBlank(cur_node)[0]
    if (row > 0):
        newMat = cur_node.matrix.copy()
        newMat[row, col] = newMat[row-1, col]
        newMat[row-1, col] = 0
        cur_node.addChild(newMat, cur_node)
        child = cur_node.child
        child.gval = cur_node.gval + 1
        child.updateHval(algorithm)
        child.updateFval()
        frontier.put(child, child.fval)
        return child
        

def findBlank(cur_node):
    result = np.where(cur_node.matrix == 0)
    pair = list(zip(result[0], result[1]))
    return pair

def uniformCost(problem):
    print("hello")

def missedTiles(matrix, matrix_goal):
    missed = np.sum(matrix != matrix_goal)
    return missed

def euclidean(matrix, matrix_goal):
    result = np.where(matrix != matrix_goal)
    pair = list(zip(result[0], result[1]))
    dist = 0
    for element in pair:
        row = element[0]
        col = element[1]
        correct = np.where(matrix[row][col] == matrix_goal)
        correct_pair = list(zip(correct[0], correct[1]))
        correct_row = correct_pair[0][0]
        correct_col = correct_pair[0][1]        
        dist += abs(row - correct_row) + abs(col - correct_col)
    return dist

def buildTree(problem, algorithm):
    global visited
    global count 
    equal = 0
    maxFrontier = 0
    maxFval = 9999999999999
    found = False
    answer = []
    stack = deque()
    
    while(1):  
        print("back to start")
        # break if nothing in frontier
        if (frontier.qsize() == 0): break;

        # track max frontier size
        if (not found and (frontier.qsize() > maxFrontier)):
            maxFrontier = frontier.qsize()

        #pop node from the frontier
        visited = np.append(visited, frontier.get())
        cur_node = visited[-1]

        #if it passes the maxFval we found already, we skip to next on frontier
        print(cur_node.fval)
        print(maxFval)
        if(maxFval < cur_node.fval):
            print("goback")
            continue

        #check if it's the goal state
        equal = np.array_equal(cur_node.matrix, problem.goal_state)
       
        #heuristics
        if (algorithm != "1"):
            if (equal and maxFval > cur_node.fval):
                answer = cur_node
                found = True
                maxFval = cur_node.fval
                continue
            # only expand when no answer found
            elif ((not equal) and (maxFval > cur_node.fval)):
                left(cur_node, algorithm)
                right(cur_node, algorithm)
                up(cur_node, algorithm)
                down(cur_node, algorithm) 
            else: continue
            
        else:
            if not(equal):
                print("The best state to expand with g(n) =", cur_node.gval, " and h(n) =", cur_node.hval, "is ...")
                printMatrix(visited[-1].matrix)
                print(" ")
            else:
                break
            l = left(cur_node, algorithm)
            r = right(cur_node, algorithm)
            u = up(cur_node, algorithm)
            d = down(cur_node, algorithm)

            if (l != None and np.array_equal(l.matrix, problem.goal_state)):
                break
            if (r != None and np.array_equal(r.matrix, problem.goal_state)):
                break
            if (u != None and np.array_equal(u.matrix, problem.goal_state)):
                break
            if (d != None and np.array_equal(d.matrix, problem.goal_state)):
                break
    
    cur = answer
    while cur is not None:
        stack.append(cur)
        cur = cur.parent

    while (len(stack) > 1):
        a = stack.pop()
        print("The best state to expand with g(n) =", a.gval, " and h(n) =", a.hval, "is ...")
        printMatrix(a.matrix)
        print(" ")
    print("stack size is", len(stack))
    print("YOU REACHED TO THE GOAL!!!")
    print("To solve this problem the search algorithm expanded a total of {} nodes".format(visited.size) )
    print("The maximum number of nodes in the queue at any one time:", maxFrontier)

def printMatrix(matrix):
    for i in range(3): 
        for j in range(3):
            print(matrix[i][j], end=" ")
        print()
        

def main():
    print("Welcome to 862093078 8 puzzle solver.\n")
    puzzle = Problem()
    puzzle.setInitial()
    first_node = Node(puzzle.getInitial())
    first_node.setGoal(puzzle.goal_state)    
    algorithm = input("\nEnter your choice of algorithm\nUniform Cost Search\nA* with the Misplaced Tile heuristic.\nA* with the Eucledian distance heuristic.\n")    
    #push the first node to the frontier
    #heapq.heappush(frontier, first_node)
    frontier.put(first_node, first_node.fval)
    buildTree(puzzle, algorithm)
    
if __name__=="__main__":
    main()
