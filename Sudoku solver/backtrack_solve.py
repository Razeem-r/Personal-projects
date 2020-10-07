from array import *
import numpy as np

# Sample sudoku form 
# board =[[0, 0, 6, 0, 0, 5, 0, 0, 0],
#         [0, 0, 8, 0, 9, 0, 0, 0, 0],
#         [0, 0, 2, 0, 0, 0, 8, 1, 7],
#         [4, 0, 0, 3, 0, 8, 0, 0, 0],
#         [0, 3, 0, 0, 5, 0, 0, 4, 0],
#         [0, 0, 0, 2, 0, 6, 0, 0, 9],
#         [8, 1, 5, 0, 0, 0, 4, 0, 0],
#         [0, 0, 0, 0, 7, 0, 9, 0, 0],
#         [0, 0, 0, 1, 0, 0, 6, 0, 0]]

#COnvert all numbers to string type and all empty spaces(zeroes) to '.'
for i in range(9):
    for j in range(9):
        board[i][j]= str(board[i][j])
        if board[i][j]=='0':
            board[i][j]='.'


def check(test,i,j,board,f):
    
    for x in range(9):
        if(board[i][x]!='.'):
                if int(board[i][x])==test and x!=j:
                    f = 0
    
    for x in range(9):
        if(board[x][j]!='.'):
            if int(board[x][j])==test and x!=i:
                f = 0
   
    if i>=0 and i<3:
        if j>=0 and j<3:
            for x in range(3):
                for y in range(3):
                    #print(board[x][y],test)
                    if(board[x][y]!='.'):
                        if int(board[x][y])==test and x!=i and y!=j:
                            f = 0
                            #print("      ",i,j,x,y)
        elif j>=3 and j<6:
            for x in range(3):
                for y in range(3,6):
                    if(board[x][y]!='.'):
                        if int(board[x][y])==test and x!=i and y!=j:
                            f = 0

        elif j>=6 and j<9:
            for x in range(3):
                for y in range(6,9):
                    if(board[x][y]!='.'):
                        if int(board[x][y])==test and x!=i and y!=j:
                            f = 0
    if i>=3 and i<6:
        if j>=0 and j<3:
            for x in range(3,6):
                for y in range(3):
                    if(board[x][y]!='.'):
                        if int(board[x][y])==test and x!=i and y!=j:
                            f = 0
        elif j>=3 and j<6:
            for x in range(3,6):
                for y in range(3,6):
                    if(board[x][y]!='.'):
                        if int(board[x][y])==test and x!=i and y!=j:
                            f = 0

        elif j>=6 and j<9:
            for x in range(3,6):
                for y in range(6,9):
                    if(board[x][y]!='.'):
                        if int(board[x][y])==test and x!=i and y!=j:
                            f = 0

    if i>=6 and i<9:
        if j>=0 and j<3:
            for x in range(6,9):
                for y in range(3):
                    if(board[x][y]!='.'):
                        if int(board[x][y])==test and x!=i and y!=j:
                            f = 0
        elif j>=3 and j<6:
            for x in range(6,9):
                for y in range(3,6):
                    if(board[x][y]!='.'):
                        if int(board[x][y])==test and x!=i and y!=j:
                            f = 0
        elif j>=6 and j<9:
            for x in range(6,9):
                for y in range(6,9):
                    if(board[x][y]!='.'):
                        if int(board[x][y])==test and x!=i and y!=j:
                            f = 0

  
    return f
def checkcriteria(test,i,j,board,flag):
    
    global c
    
    if flag!=-1:
        
        c=flag    
    
    while True:
        f = 1
        f = check(test,i,j,board,f)
        if f==1 and test<10:
            board[i][j]=int(test)
            break
        if f==0 and test<10:            
            test=test+1
        if test>9:
            break
        
#     print("board show ",i,j,f,type(board[i][j]))
#     print(c)
#     print(np.asarray(board))
   
    if i<9:
        if board[i][j]==0 or board[i][j]==c:
            board[i][j]=0;
            c=0

            if not (i<=0 and j<=0):
                return -1

        else:
            c=0
            return 1
    return 0
    
def checkdot(test,i,j,board,nav):
#     print("in checkdot")
    if i<9:
        if nav==1 :
            i,j = front(i,j,board)
            if i<9:
                if type(board[i][j])==int:
                    board[i][j]=0

                if board[i][j]=='.' or board[i][j]==0 :
                    board[i][j]=int('0')
                    test=1
                    nav = checkcriteria(test,i,j,board,-1)
        if nav==-1 :
            i,j=back(i,j,board)
            g=board[i][j]
            #print('   g ',g)
            if board[i][j]!=9:
                k=board[i][j]+1
            else:
                k=9
            nav = checkcriteria(k,i,j,board,g) 
        if nav==1 or nav ==-1:
            checkdot(test,i,j,board,nav)
        


def back(i,j,board):
    if i<9:
        while True:
            j-=1
            if j<0:
                i-=1
                j=8
            if type(board[i][j])==int:
                break        
    return i,j


def front(i,j,board): 
    if i<9:
        while True:
            j+=1
            if j>=9:
                i+=1
                j=0
            if i<9:    
                if type(board[i][j])!=str or board[i][j]=='.':
                    break
            if i==9:
                break
    return i,j
c=0    
n=1
f=1
i=0
j=-1
test=1

checkdot(test,i,j,board,n)
print(np.asarray(board))
