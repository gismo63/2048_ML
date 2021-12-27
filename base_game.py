import pygame, sys, random
from pygame.locals import *

# Create the constants (go ahead and experiment with different values)
BOARDWIDTH = 4  # number of columns in the board
BOARDHEIGHT = 4 # number of rows in the board
TILESIZE = 80
WINDOWWIDTH = 640
WINDOWHEIGHT = 480
FPS = 30
BLANK = None

#                 R    G    B
BLACK =         (  0,   0,   0)
WHITE =         (255, 255, 255)
BRIGHTBLUE =    (  0,  50, 255)
DARKTURQUOISE = (  3,  54,  73)
GREEN =         (  0, 204,   0)

BGCOLOR = DARKTURQUOISE
TILECOLOR = GREEN
TEXTCOLOR = WHITE
BORDERCOLOR = BRIGHTBLUE
BASICFONTSIZE = 20

BUTTONCOLOR = WHITE
BUTTONTEXTCOLOR = BLACK
MESSAGECOLOR = WHITE

XMARGIN = int((WINDOWWIDTH - (TILESIZE * BOARDWIDTH + (BOARDWIDTH - 1))) / 2)
YMARGIN = int((WINDOWHEIGHT - (TILESIZE * BOARDHEIGHT + (BOARDHEIGHT - 1))) / 2)

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

def main():
    global FPSCLOCK, DISPLAYSURF, BASICFONT, RESET_SURF, RESET_RECT, NEW_SURF, NEW_RECT, SOLVE_SURF, SOLVE_RECT

    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    pygame.display.set_caption('Slide Puzzle')
    BASICFONT = pygame.font.Font('freesansbold.ttf', BASICFONTSIZE)

    # Store the option buttons and their rectangles in OPTIONS.
    RESET_SURF, RESET_RECT = makeText('Reset',    TEXTCOLOR, TILECOLOR, WINDOWWIDTH - 120, WINDOWHEIGHT - 90)
    NEW_SURF,   NEW_RECT   = makeText('New Game', TEXTCOLOR, TILECOLOR, WINDOWWIDTH - 120, WINDOWHEIGHT - 60)
    SOLVE_SURF, SOLVE_RECT = makeText('Solve',    TEXTCOLOR, TILECOLOR, WINDOWWIDTH - 120, WINDOWHEIGHT - 30)

    startingBoard = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    startingBoard[random.randint(0,BOARDHEIGHT-1)][random.randint(0,BOARDWIDTH-1)] = 2
    mainBoard = startingBoard
    wasChanged = True
    score = 0
    scoreAdd = 0

    while True: # main game loop
        slideTo = None # the direction, if any, a tile should slide
        startingMsg = 'Press arrow keys to slide.'
        msg = 'Press arrow keys to slide.' + str(score) # contains the message to show in the upper left corner.
        
        
        if wasChanged:
            drawBoard(mainBoard, msg)
        wasChanged = False

        checkForQuit()
        for event in pygame.event.get(): # event handling loop
            if event.type == MOUSEBUTTONUP:
                spotx, spoty = getSpotClicked(mainBoard, event.pos[0], event.pos[1])

                if (spotx, spoty) == (None, None):
                    # check if the user clicked on an option button  
                    if NEW_RECT.collidepoint(event.pos):
                        drawBoard(startingBoard, startingMsg)

            elif event.type == KEYUP:
                # check if the user pressed a key to slide a tile
                if event.key in (K_LEFT, K_a):
                    mainBoard, wasChanged, scoreAdd = newBoard(mainBoard, LEFT)
                elif event.key in (K_RIGHT, K_d):
                    mainBoard, wasChanged, scoreAdd = newBoard(mainBoard, RIGHT)
                elif event.key in (K_UP, K_w):
                    mainBoard, wasChanged, scoreAdd = newBoard(mainBoard, UP)
                elif event.key in (K_DOWN, K_s):
                    mainBoard, wasChanged, scoreAdd = newBoard(mainBoard, DOWN)
        score+=scoreAdd
        scoreAdd=0
        pygame.display.update()
        FPSCLOCK.tick(FPS)


def terminate():
    pygame.quit()
    sys.exit()


def checkForQuit():
    for event in pygame.event.get(QUIT): # get all the QUIT events
        terminate() # terminate if any QUIT events are present
    for event in pygame.event.get(KEYUP): # get all the KEYUP events
        if event.key == K_ESCAPE:
            terminate() # terminate if the KEYUP event was for the Esc key
        pygame.event.post(event) # put the other KEYUP event objects back


def newBoard(board, move):#################################################HERE
    isChanged = False
    newScore = 0
    if move == LEFT:
        for i in range(BOARDHEIGHT):
            edge = 0 #if combined then edge should be moved past
            for j in range(1,BOARDWIDTH):
                if board[i][j] != 0:
                    pos = j
                    while pos>edge:
                        if board[i][pos-1] == 0:
                            pos -= 1
                        elif board[i][pos-1] == board[i][j]:
                            newScore+=board[i][j]*2
                            board[i][pos-1] *= 2
                            edge = pos
                            board[i][j] = 0
                            isChanged = True
                            newScore=board[i][pos-1]
                        else:
                            if pos != j:
                                board[i][pos] = board[i][j]
                                board[i][j] = 0
                                isChanged = True
                            edge = pos

                    if board[i][edge] == 0:
                        board[i][edge] = board[i][j]
                        board[i][j] = 0
                        isChanged = True
    if move == UP:
        for j in range(BOARDWIDTH):
            edge = 0 #if combined then edge should be moved past
            for i in range(1,BOARDHEIGHT):
                if board[i][j] != 0:
                    pos = i
                    while pos>edge:
                        if board[pos-1][j] == 0:
                            pos -= 1
                        elif board[pos-1][j] == board[i][j]:
                            newScore+=board[i][j]*2
                            board[pos-1][j] *= 2
                            edge = pos
                            board[i][j] = 0
                            isChanged = True
                        else:
                            if pos != i:
                                board[pos][j] = board[i][j]
                                board[i][j] = 0
                                isChanged = True
                            edge = pos

                    if board[edge][j] == 0:
                        board[edge][j] = board[i][j]
                        board[i][j] = 0
                        isChanged = True
    if move == RIGHT:
        for i in range(BOARDHEIGHT):
            edge = 0 #if combined then edge should be moved past
            for j in range(1,BOARDWIDTH):
                if board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] != 0:
                    pos = j
                    while pos>edge:
                        if board[BOARDHEIGHT-i-1][BOARDHEIGHT-pos] == 0:
                            pos -= 1
                        elif board[BOARDHEIGHT-i-1][BOARDHEIGHT-pos] == board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1]:
                            newScore+=board[i][j]*2
                            board[BOARDHEIGHT-i-1][BOARDHEIGHT-pos] *= 2
                            edge = pos
                            board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] = 0
                            isChanged = True
                        else:
                            if pos != j:
                                board[BOARDHEIGHT-i-1][BOARDHEIGHT-pos-1] = board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1]
                                board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] = 0
                                isChanged = True
                            edge = pos

                    if board[BOARDHEIGHT-i-1][BOARDHEIGHT-edge-1] == 0:
                        board[BOARDHEIGHT-i-1][BOARDHEIGHT-edge-1] = board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1]
                        board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] = 0
                        isChanged = True
    if move == DOWN:
        for j in range(BOARDWIDTH):
            edge = 0 #if combined then edge should be moved past
            for i in range(1,BOARDHEIGHT):
                if board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] != 0:
                    pos = i
                    while pos>edge:
                        if board[BOARDHEIGHT-pos][BOARDHEIGHT-j-1] == 0:
                            pos -= 1
                        elif board[BOARDHEIGHT-pos][BOARDHEIGHT-j-1] == board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1]:
                            newScore+=board[i][j]*2
                            board[BOARDHEIGHT-pos][BOARDHEIGHT-j-1] *= 2
                            edge = pos
                            board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] = 0
                            isChanged = True
                        else:
                            if pos != i:
                                board[BOARDHEIGHT-pos-1][BOARDHEIGHT-j-1] = board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1]
                                board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] = 0
                                isChanged = True
                            edge = pos

                    if board[BOARDHEIGHT-edge-1][BOARDHEIGHT-j-1] == 0:
                        board[BOARDHEIGHT-edge-1][BOARDHEIGHT-j-1] = board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1]
                        board[BOARDHEIGHT-i-1][BOARDHEIGHT-j-1] = 0
                        isChanged = True
    if isChanged:
        zeroInd = []
        numZeros = 0
        for i in range(BOARDHEIGHT):
            for j in range(BOARDWIDTH):
                if not board[i][j]:
                    zeroInd.append([i,j])
                    numZeros+=1
        zeroLoc = random.randint(0,numZeros-1)
        if random.randint(0,9):
            board[zeroInd[zeroLoc][0]][zeroInd[zeroLoc][1]] = 2
        else:
            board[zeroInd[zeroLoc][0]][zeroInd[zeroLoc][1]] = 4
    print(newScore)
    return board, isChanged, newScore

                            


            



def drawBoard(board, message):
    DISPLAYSURF.fill(BGCOLOR)
    if message:
        textSurf, textRect = makeText(message, MESSAGECOLOR, BGCOLOR, 5, 5)
        DISPLAYSURF.blit(textSurf, textRect)

    for tilex in range(len(board)):
        for tiley in range(len(board[0])):
            if board[tilex][tiley]:
                drawTile(tiley, tilex, board[tilex][tiley])

    left, top = getLeftTopOfTile(0, 0)
    width = BOARDWIDTH * TILESIZE
    height = BOARDHEIGHT * TILESIZE
    pygame.draw.rect(DISPLAYSURF, BORDERCOLOR, (left - 5, top - 5, width + 11, height + 11), 4)

    DISPLAYSURF.blit(NEW_SURF, NEW_RECT)

def getLeftTopOfTile(tileX, tileY):
    left = XMARGIN + (tileX * TILESIZE) + (tileX - 1)
    top = YMARGIN + (tileY * TILESIZE) + (tileY - 1)
    return (left, top)


def getSpotClicked(board, x, y):
    # from the x & y pixel coordinates, get the x & y board coordinates
    for tileX in range(len(board)):
        for tileY in range(len(board[0])):
            left, top = getLeftTopOfTile(tileX, tileY)
            tileRect = pygame.Rect(left, top, TILESIZE, TILESIZE)
            if tileRect.collidepoint(x, y):
                return (tileX, tileY)
    return (None, None)


def drawTile(tilex, tiley, number, adjx=0, adjy=0):
    # draw a tile at board coordinates tilex and tiley, optionally a few
    # pixels over (determined by adjx and adjy)
    left, top = getLeftTopOfTile(tilex, tiley)
    pygame.draw.rect(DISPLAYSURF, TILECOLOR, (left + adjx, top + adjy, TILESIZE, TILESIZE))
    textSurf = BASICFONT.render(str(number), True, TEXTCOLOR)
    textRect = textSurf.get_rect()
    textRect.center = left + int(TILESIZE / 2) + adjx, top + int(TILESIZE / 2) + adjy
    DISPLAYSURF.blit(textSurf, textRect)


def makeText(text, color, bgcolor, top, left):
    # create the Surface and Rect objects for some text.
    textSurf = BASICFONT.render(text, True, color, bgcolor)
    textRect = textSurf.get_rect()
    textRect.topleft = (top, left)
    return (textSurf, textRect)

if __name__ == '__main__':
    main()