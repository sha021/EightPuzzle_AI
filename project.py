import numpy as np
frontier = []
visited = []

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
        self.child = None
        self.parent = None
        
    def __init__(self, mat):
        self.matrix = mat
        self.child = None
        self.parent = None
    def __repr__(self):
        return self.matrix 
    def firstChild(self, mat):
        first_child = Node(mat)
        frontier.append(first_child)
    def addChild(self, child_mat, cur_node):
        child_node = Node(child_mat)
        cur_node.child = child_node
        child_node.parent = cur_node
    def getChild(self, cur_node):
        if (cur_node.child!=None):
            return cur_node.child;
        
def findBlank(cur_node):
    result = np.where(cur_node.matrix == 0)
    pair = list(zip(result[0], result[1]))
    return pair
def left(cur_node):
    row, col = findBlank(cur_node)[0]
    if (col > 0):
        newMat = cur_node.matrix.copy()
        newMat[row, col] = newMat[row, col-1]
        newMat[row, col-1] = 0
        cur_node.addChild(newMat, cur_node)
        print(cur_node.matrix)
def right(cur_node):
    row, col = findBlank(cur_node)[0]
    if (col < 2):
        newMat = cur_node.matrix.copy()
        newMat[row, col] = newMat[row, col+1]
        newMat[row, col+1] = 0
        cur_node.addChild(newMat, cur_node)
        print(cur_node.matrix)
def down(cur_node):
    row, col = findBlank(cur_node)[0]
    if (row < 2):
        newMat = cur_node.matrix.copy()
        newMat[row, col] = newMat[row+1, col]
        newMat[row+1, col] = 0
        cur_node.addChild(newMat, cur_node)
        print(cur_node.matrix)
def up(cur_node):
    row, col = findBlank(cur_node)[0]
    print(row)
    if (row > 0):
        newMat = cur_node.matrix.copy()
        newMat[row, col] = newMat[row-1, col]
        newMat[row-1, col] = 0
        cur_node.addChild(newMat, cur_node)
        print(newMat)


def uniformCost(problem):
    print("hello")


def printMatrix(matrix):
    for i in range(3): 
        for j in range(3):
            print(matrix[i][j], end=" ")
        print("\n")

def missedTile(cur_node, problem):
    missed = np.sum(cur_node.matrix != problem.goal_stat

def main():
    print("Welcome to 862093078 8 puzzle solver.\n")
    puzzle = Problem()
    puzzle.setInitial()
    mat = puzzle.getInitial()
    first_node = Node(mat)
    algorithm=input("\nEnter your choice of algorithm\nUniform Cost Search\nA* with the Misplaced Tile heuristic.\nA* with the Eucledian distance heuristic.\n")    
    # node = Node(initial)
    printMatrix(puzzle.getInitial())
    printMatrix(puzzle.goal_state)
    missedTile(first_node, puzzle)
    #up(first_node)
if __name__=="__main__":
    main()
