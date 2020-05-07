import numpy as np
import heapq
from queue import PriorityQueue
from collections import deque 
import sys
import math
import time

frontier = PriorityQueue()
visited = np.array([])
count = 0
check = set()
# Trivial = np.array([[1,2,3],[4,5,6],[7,8,0]])
# VeryEasy = np.array([[1,2,3],[4,5,6],[7,0,8]])
# Easy = np.array([[1,2,0],[4,5,3],[7,8,6]])
# Doable = np.array([[0,1,2],[4,5,3],[7,8,6]])
# OhBoy =  np.array([[8,7,1],[6,0,2],[5,4,3]])
# Impossible = np.array([[1,2,3],[4,5,6],[8,7,0]])
# THE PROBLEM = np.array([[1,0,3],[4,2,6],[7,5,8]])

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
            self.initial_state = np.array([[1,0,3],[4,2,6],[7,5,8]])
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
def left(cur_node, algorithm, row, col):
    row, col = findBlank(cur_node)[0]
    if (col > 0):
        newMat = cur_node.matrix.copy()
        newMat[row, col] = newMat[row, col-1]
        newMat[row, col-1] = 0
        exists = False
        strMat = np.array2string(newMat)
        if strMat in check:
            exists = True
        else:
            cur_node.addChild(newMat, cur_node)
            child = cur_node.child
            child.gval = cur_node.gval + 1
            child.updateHval(algorithm)
            child.updateFval()
            frontier.put(child, child.fval)
            return child
        
def right(cur_node, algorithm, row, col):
    row, col = findBlank(cur_node)[0]
    if (col < 2):
        newMat = cur_node.matrix.copy()
        newMat[row, col] = newMat[row, col+1]
        newMat[row, col+1] = 0
        exists = False
        strMat = np.array2string(newMat)
        if strMat in check:
            exists = True
        else:
            cur_node.addChild(newMat, cur_node)
            child = cur_node.child
            child.gval = cur_node.gval + 1
            child.updateHval(algorithm)
            child.updateFval()
            frontier.put(child, child.fval)
            return child

def down(cur_node, algorithm, row, col):
    row, col = findBlank(cur_node)[0]
    if (row < 2):
        newMat = cur_node.matrix.copy()
        newMat[row, col] = newMat[row+1, col]
        newMat[row+1, col] = 0
        exists = False        
        strMat = np.array2string(newMat)
        if strMat in check:
            exists = True
        else:
            cur_node.addChild(newMat, cur_node)
            child = cur_node.child
            child.gval = cur_node.gval + 1
            child.updateHval(algorithm)
            child.updateFval()
            frontier.put(child, child.fval)
            return child
        
def up(cur_node, algorithm, row, col):
    row, col = findBlank(cur_node)[0]
    if (row > 0):
        newMat = cur_node.matrix.copy()
        newMat[row, col] = newMat[row-1, col]
        newMat[row-1, col] = 0
        exists = False
        strMat = np.array2string(newMat)
        if strMat in check:
            exists = True
        else:
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
        dist += math.sqrt((row - correct_row)**2 + (col - correct_col)**2)
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
    numNodes = []
    while(1):  
        # break if nothing in frontier
        if (frontier.qsize() == 0): break;
        # track max frontier size
 
        if (not found and (frontier.qsize() > maxFrontier)):
            maxFrontier = frontier.qsize()

        #pop node from the frontier
        temp = frontier.get()
        strMat = np.array2string(temp.matrix)
        if strMat in check:
            continue
        check.add(strMat)
        visited = np.append(visited, temp)
        cur_node = visited[-1]

        #if it passes the maxFval we found already, we skip to next on frontier
        if (maxFval < cur_node.fval):
            continue
       
        #check if it's the goal state
        equal = np.array_equal(cur_node.matrix, problem.goal_state)
        #heuristics
        if (algorithm != "1"):
            if (equal and maxFval > cur_node.fval):
                print("YOU REACHED TO THE GOAL!!!")
                printMatrix(visited[-1].matrix)
                answer = cur_node
                numNodes.append(visited.size)
                found = True
                maxFval = cur_node.fval
                continue
            # only expand when no answer found
            elif ((not equal) and (maxFval > cur_node.fval)):
                print("expanding...")
                print("The best state to expand with g(n) =", cur_node.gval, " and h(n) =", cur_node.hval, "is ...")
                printMatrix(visited[-1].matrix)
                print(" ")
                row, col = findBlank(cur_node)[0]
                left(cur_node, algorithm, row, col)
                right(cur_node, algorithm, row, col)
                up(cur_node, algorithm, row, col)
                down(cur_node, algorithm, row, col) 
            else: continue
            
        else:
            if not(equal):
                print("expanding.... g(n) =", cur_node.gval, " and h(n) =", cur_node.hval)
                printMatrix(visited[-1].matrix)
                row, col = findBlank(cur_node)[0]
                left(cur_node, algorithm, row, col)
                right(cur_node, algorithm, row, col)
                up(cur_node, algorithm, row, col)
                down(cur_node, algorithm, row, col) 
                print(" ")
            else:
                print("YOU REACHED TO THE GOAL!!!")
                printMatrix(visited[-1].matrix)
                numNodes.append(visited.size)
                answer = cur_node
                maxFval = cur_node.fval
                break   
   
    cur = answer
    if (visited.size > 1):
        while cur is not None:
            stack.append(cur)
            if (cur.parent != None): 
                cur = cur.parent
            else:
                break
    
    print("-----------------------------------------------------------------------------")
    print("TRACK OF THE BEST POSSIBLE MOVES")
    while (len(stack) > 1):
        a = stack.pop()
        print("The best state to expand with g(n) =", a.gval, " and h(n) =", a.hval, "is ...")
        printMatrix(a.matrix)
    print("THE PUZZLE IS SOLVED WITH BEST POSSIBLE MOVES")
    if (len(stack)>0):
        a = stack.pop()
        printMatrix(a.matrix)
    print("-----------------------------------------------------------------------------")
    i = 0
    num = 0
    for item in numNodes:
        i += 1
        print(i,"th goal is found with expanding {} nodes".format(item - num))
        num = item
    print("To solve this problem the search algorithm expanded a total of {} nodes".format(visited.size) )
    print("The maximum number of nodes in the queue at any one time:", maxFrontier)
    print("-----------------------------------------------------------------------------")

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
    strMat = np.array2string(first_node.matrix)
    algorithm = input("\nEnter your choice of algorithm\nUniform Cost Search\nA* with the Misplaced Tile heuristic.\nA* with the Eucledian distance heuristic.\n")    
    #push the first node to the frontier
    #heapq.heappush(frontier, first_node)
    frontier.put(first_node, first_node.fval)
    start_time = time.time()
    buildTree(puzzle, algorithm)
    print("Execution of solving the puzzle took %s seconds" % (time.time() - start_time))
    
if __name__=="__main__":
    main()
