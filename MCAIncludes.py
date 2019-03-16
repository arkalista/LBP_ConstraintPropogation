######################### MCA - Class ###############
import queue as queue
import numpy as np

############# Procedure: HeaviSide : Threshold Function ##########################
# Input:
#   h is the difference of the pixel with it's neighbour
# Output:
#   Returns the flag for the constraint

def HeaviSide(h):
    if (h == 0):
        return 0
    elif (h > 0):
        return 1
    else:
        return -1


############# Procedure: LBP for a given pixel value ##########################
# Input:
#   h is the difference of the pixel with it's neighbour
# Output:
#   Returns the flag for the constraint


def lbp(h):
    if (h < 0):
        return 0
    else:
        return 1

############# Procedure: LBP Boiler  ##########################


# Input:
#   pixel index p q
#   NEIGHBORS an array of off set => ([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, +1]])
#   cImage is the 2d matrix, the image for which we want to calculate the lbp
# Output:
#   return a list of lbpinfo
#   lbpinfo : p,q,pixel value, lbppattern_list, constraint_list

# Todo's : Create a datastructure for lbpinfo and encode the output in it for further use

def calculateLBP(p, q, NEIGHBORS, cImage):
    h1 = NEIGHBORS[:, 0] + p
    h2 = NEIGHBORS[:, 1] + q
    pixel = cImage[p, q]
    pixelDiff = [HeaviSide(int(cImage[h1[i], h2[i]]) - int(pixel)) for i in range(8)]
    pixelLbp = [lbp(int(cImage[h1[i], h2[i]]) - int(pixel)) for i in range(8)]
    pixelLbp = reversed(pixelLbp)
    pixelLbp = int(''.join(str(e) for e in pixelLbp), 2)
    return [p, q, pixel, pixelLbp, pixelDiff]


############# Procedure: Check Padded Pixels in the Image ##########################
# Input:
#   NEIGHBORS an array of off set => ([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, +1]])
#   NEIGHBORS_OFFSET is an index to NEIGHBORS array, values can be in range 0 to 7

# Output:

def validateNPI(NEIGHBORS_OFFSET, m, n, NEIGHBORS, p, q):
    try:
        x = m + NEIGHBORS[NEIGHBORS_OFFSET][0]
        y = n + NEIGHBORS[NEIGHBORS_OFFSET][1]
        if x > 0 and x < p - 1 and y > 0 and y < q - 1:
            return (x, y)
        else:
            return (-1, -1)
    except:
        h = 1
    return (-1, -1)


############# Procedure: Creating Dictionary for Plato Pixels ##########################
# Input:
#   PixelLBP is a dictionary
#   NEIGHBORS an array of off set => ([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, +1]])
#   (p,q) is the pixel index
#   constraint => {0:Equal, 1:Superior, -1: Inferior}
# Output:
#   returns a dictionary {key:(p,q) value:[(xn,yn)]} xn,yn => constraint neighbour of p,q

def getConstNeigh(PixelLBP, NEIGHBORS, p, q, constraint):
    k = 0
    RawNodes = list(PixelLBP)
    platoDict = {}
    for h in RawNodes:
        x = h[4]
        y = [validateNPI(i, h[0], h[1], NEIGHBORS.tolist(), p, q) for i, val in enumerate(x) if val == constraint]
        z = [(s - 1, t - 1) for (s, t) in y if (s != -1) and (t != -1)]
        platoDict[(h[0] - 1, h[1] - 1)] = z
        k = k + 1
    return platoDict


############# Procedure: Check if the plato node is minima ##########################
# Input:
# Output:
# Description:

def isMinimaTree(plat, conMatplateaus):
    for tPoint in plat:
        tConMat = conMatplateaus[tPoint]
        checkConMat = [val for val in tConMat if val < 0]
        if (checkConMat):
            return 0
    return 1


############# Procedure: Check if the plato node is maxima ##########################
# Input:
# Output:
# Description:

def isMaximaTree(plat, conMatplateaus):
    for tPoint in plat:
        tConMat = conMatplateaus[tPoint]
        checkConMat = [val for val in tConMat if val > 0]
        if (checkConMat):
            return 0
    return 1




############# Procedure: load And PreProcess Image ##########################
# Input:
# Output:
# Description:

def LoadImage(image):
    topRow = image[0, :]
    bottomRow = image[-1, :]
    rImage = np.row_stack([topRow, image, bottomRow])
    firstCol = rImage[:, 0]
    lastCol = rImage[:, -1]
    cImage = np.column_stack([firstCol, rImage, lastCol])
    row, col = cImage.shape
    return [row, col, cImage]


############# Procedure: Extract Plateuas ##########################
# Input:
# Output:
# Description:

def ExtractPlateau(image, platoDict, conMatplateaus):
    minima = []
    maxima = []
    plateutree = {}
    plateutreeVal = {}
    PointIndex = {}

    row, col = image.shape
    platopixel = np.zeros((row, col))

    TreeID = 1
    for x, y in np.ndindex(platopixel.shape):
        plat = []
        if (platopixel[(x, y)] == 0):
            platopixel[(x, y)] = TreeID
            PointIndex[(x, y)] = TreeID
            plat.append((x, y))

            activeQueue = queue.Queue()
            activeQueue.put((x, y))

            while not activeQueue.empty():
                nodeIndex = activeQueue.get()
                listPlatNodes = platoDict[nodeIndex]
                for point in listPlatNodes:
                    if platopixel[point] != TreeID:
                        platopixel[point] = TreeID
                        PointIndex[point] = TreeID
                        plat.append(point)
                        activeQueue.put(point)

            plateutree[TreeID] = plat
            plateutreeVal[TreeID] = image[plat[0]]
            if (isMinimaTree(plat, conMatplateaus)):
                minima.append(TreeID)

            if (isMaximaTree(plat, conMatplateaus)):
                maxima.append(TreeID)

            TreeID = TreeID + 1
    return [platopixel, plateutree, PointIndex, minima, maxima, plateutreeVal]



############ Procedure: Generate LBP Image ##########################
# Input:
# Output:
# Description:

def GenerateLBPImage(image,NEIGHBORS):
    iData = LoadImage(image)
    row = iData[0]
    col = iData[1]
    cImage = iData[2]
    lbpImage = np.zeros((row - 2, col - 2))
    PixelLBP = [calculateLBP(i, j, NEIGHBORS, cImage) for i, j in np.ndindex(cImage.shape) if
                i > 0 and i < row - 1 and j > 0 and j < col - 1]
    for pixel in PixelLBP:
        lbpImage[(pixel[0] - 1, pixel[1] - 1)] = pixel[3]
    return lbpImage

############# Procedure: Return Minima Nodes i.e. Neigboring Minima Plato ##########################
# Input:
# Output:
# Description:

def getMinimaNodeID(MinimaRefereces, plateutree, PointIndex, activeNodeID):
    minNodeListID = []
    NodeList = plateutree[activeNodeID]
    for node in NodeList:
        minNodeList = MinimaRefereces[node]
        for minNode in minNodeList:
            minNodeListID.append(PointIndex[minNode])
    return list(set(minNodeListID))

############# Procedure: Return Maxima Nodes i.e. Neigboring Maxima Plato ##########################
# Input:
# Output:
# Description:

def getMaximaNodeID(MaximaRefereces, plateutree, PointIndex, activeNodeID):
    maxNodeListID = []
    NodeList = plateutree[activeNodeID]
    for node in NodeList:
        maxNodeList = MaximaRefereces[node]
        for maxNode in maxNodeList:
            maxNodeListID.append(PointIndex[maxNode])
    return list(set(maxNodeListID))

############# Procedure: Update SubTree Depth Minima ##########################
# Input:
# Output:
# Description:
# Take one more argument for step size in the function
def UpdateTreeDepth_Minima(nodeID, MinimaTree, stepSize =1):

    activeQueue = queue.Queue()
    passiveQueue = queue.Queue()

    passiveQueue.put(nodeID)

    while True:
        if passiveQueue.empty() and activeQueue.empty():
            break
        else:
            Qlist = []
            while not passiveQueue.empty():
                pasiveNodeID = passiveQueue.get()
                Qlist.append(pasiveNodeID)
            _Qlist = set(Qlist)
            for ele in _Qlist:
                activeQueue.put(ele)

            while not activeQueue.empty():
                activeNodeID = activeQueue.get()
                activeLevel = MinimaTree[activeNodeID][0][1]

                for nodeID in MinimaTree.keys():
                    insertCheck = 0
                    # Check if the activeNode ID exist as the parent for any of the child
                    # then  update the level and key for the NodeID into Passive Queue
                    if (MinimaTree[nodeID][0][0] == activeNodeID):
                        insertCheck = 1
                        for i in range(0, len(MinimaTree[nodeID])):
                            #MinimaTree[nodeID][i][1] = activeLevel + 1
                            MinimaTree[nodeID][i][1] = activeLevel + stepSize
                    if (insertCheck == 1):
                        passiveQueue.put(nodeID)

    return MinimaTree


############# Procedure: Code for Extrema Extents  ##########################
# Input:
# Output:
# Description:

def expandTree_Minima(minimaIndex, MinimaRefereces, plateutree, PointIndex, stepSize =1):
    activeQueue = queue.Queue()
    passiveQueue = queue.Queue()

    minTreeIndex = minimaIndex
    MinimaTree = {}
    # Extracting root node pixels for expansion, and initializing tree Root
    MinimaTree[minTreeIndex] = []
    # Add Nodes to root tree and populating queue for tree expansion
    MinimaTree[minTreeIndex].append([0, 0])
    passiveQueue.put(minTreeIndex)

    while True:
        if passiveQueue.empty() and activeQueue.empty():
            break
        else:
            Qlist = []
            while not passiveQueue.empty():
                pasiveNodeID = passiveQueue.get()
                Qlist.append(pasiveNodeID)
#                activeQueue.put(pasiveNodeID)
            _Qlist = set(Qlist)
            for ele in _Qlist:
                activeQueue.put(ele)

            while not activeQueue.empty():

                activeNodeID = activeQueue.get()
                activeLevel = MinimaTree[activeNodeID][0][1]

                minNodeIDList = getMinimaNodeID(MinimaRefereces, plateutree, PointIndex, activeNodeID)

                for minNodeID in minNodeIDList:
                    passiveQueue.put(minNodeID)
                    # Check the level and then extract a subtree , update level and update parents
                    if minNodeID in MinimaTree.keys():
                        minNodeLevel = MinimaTree[minNodeID][0][1]
                        if minNodeLevel < activeLevel + stepSize:
                            # Updating Tree Level for all the branches and ParentID
                            for i in range(0, len(MinimaTree[minNodeID])):
                                MinimaTree[minNodeID][i][0] = activeNodeID
                                MinimaTree[minNodeID][i][1] = activeLevel + stepSize
                            # Updating SubTree Levels
                            MinimaTree = UpdateTreeDepth_Minima(minNodeID, MinimaTree)
                    else:
                        MinimaTree[minNodeID] = []
                        MinimaTree[minNodeID].append([activeNodeID, activeLevel + stepSize])

    return MinimaTree


############# Procedure: Update SubTree Depth Maxima ##########################
# Input:
# Output:
# Description:

def UpdateTreeDepth_Maxima(nodeID, MaximaTree, stepSize =1):
    activeQueue = queue.Queue()
    passiveQueue = queue.Queue()

    passiveQueue.put(nodeID)

    while True:
        if passiveQueue.empty() and activeQueue.empty():
            break
        else:
            Qlist = []
            while not passiveQueue.empty():
                pasiveNodeID = passiveQueue.get()
                Qlist.append(pasiveNodeID)
            _Qlist = set(Qlist)
            for ele in _Qlist:
                activeQueue.put(ele)

            while not activeQueue.empty():
                activeNodeID = activeQueue.get()
                activeLevel = MaximaTree[activeNodeID][0][1]

                for nodeID in MaximaTree.keys():
                    insertCheck = 0
                    # Check if the activeNode ID exist as the parent for any of the child
                    # then  update the level and key for the NodeID into Passive Queue
                    if (MaximaTree[nodeID][0][0] == activeNodeID):
                        insertCheck = 1
                        for i in range(0, len(MaximaTree[nodeID])):
                            MaximaTree[nodeID][i][1] = activeLevel - stepSize
                    if (insertCheck == 1):
                        passiveQueue.put(nodeID)

    return MaximaTree

############# Procedure: Code for Extrema Extents  ##########################
# Input:
# Output:
# Description:

def expandTree_Maxima(maximaIndex, MaximaRefereces, plateutree, PointIndex):
    activeQueue = queue.Queue()
    passiveQueue = queue.Queue()

    maxTreeIndex = maximaIndex
    MaximaTree = {}
    # Extracting root node pixels for expansion, and initializing tree Root
    MaximaTree[maxTreeIndex] = []
    # Add Nodes to root tree and populating queue for tree expansion
    MaximaTree[maxTreeIndex].append([0, 255])
    passiveQueue.put(maxTreeIndex)

    while True:
        if passiveQueue.empty() and activeQueue.empty():
            break
        else:
            Qlist = []
            while not passiveQueue.empty():
                pasiveNodeID = passiveQueue.get()
                Qlist.append(pasiveNodeID)
#                activeQueue.put(pasiveNodeID)
            _Qlist = set(Qlist)
            for ele in _Qlist:
                activeQueue.put(ele)

            while not activeQueue.empty():

                activeNodeID = activeQueue.get()
                activeLevel = MaximaTree[activeNodeID][0][1]

                maxNodeIDList = getMaximaNodeID(MaximaRefereces, plateutree, PointIndex, activeNodeID)

                for maxNodeID in maxNodeIDList:
                    passiveQueue.put(maxNodeID)
                    # Check the level and then extract a subtree , update level and update parents
                    if maxNodeID in MaximaTree.keys():
                        maxNodeLevel = MaximaTree[maxNodeID][0][1]
                        if maxNodeLevel > activeLevel - 1:
                            # Updating Tree Level for all the branches and ParentID
                            for i in range(0, len(MaximaTree[maxNodeID])):
                                MaximaTree[maxNodeID][i][0] = activeNodeID
                                MaximaTree[maxNodeID][i][1] = activeLevel - 1
                            # Updating SubTree Levels
                            MaximaTree = UpdateTreeDepth_Maxima(maxNodeID, MaximaTree)
                    else:
                        MaximaTree[maxNodeID] = []
                        MaximaTree[maxNodeID].append([activeNodeID, activeLevel - 1])

    return MaximaTree



############# Procedure: Code for Creating tree from Images  ##########################
# Input:
# Output:
# Description:

def CreateImageFromTree_Minima(minForest_Parallel,row,col,plateutree):
        minCellArray = {}
        minLevelArray = {}
        for index,minTree in enumerate(minForest_Parallel):
            for key in minTree.keys():
                if key not in minCellArray.keys():
                    minCellArray[key] = [index]
                    minLevelArray[key] = [minTree[key][0][1]]
                else:
                    minCellArray[key].append(index)
                    minLevelArray[key].append(minTree[key][0][1])

        pass_Image = np.zeros((row - 2, col - 2))
        pass_LevelImage = np.zeros((row - 2, col - 2))

        for treeIndex in plateutree.keys():
            IndexList = plateutree[treeIndex]
            (PixelVal, PixelTreeIndex) = max((v, i) for i, v in enumerate(minLevelArray[treeIndex]))
            # With pixel Level also store the value with which the tree is filled.
            for index in IndexList:
                pass_Image[index] = PixelVal
                pass_LevelImage[index] = PixelTreeIndex

        return [pass_Image,pass_LevelImage]


def CreateImageFromTree_Maxima(maxForest_Parallel, row, col, plateutree):
    maxCellArray = {}
    maxLevelArray = {}
    for index, maxTree in enumerate(maxForest_Parallel):
        for key in maxTree.keys():
            if key not in maxCellArray.keys():
                maxCellArray[key] = [index]
                maxLevelArray[key] = [maxTree[key][0][1]]
            else:
                maxCellArray[key].append(index)
                maxLevelArray[key].append(maxTree[key][0][1])

    pass_Image = np.zeros((row - 2, col - 2))
    pass_LevelImage = np.zeros((row - 2, col - 2))

    for treeIndex in plateutree.keys():
        IndexList = plateutree[treeIndex]
        (PixelVal, PixelTreeIndex) = min((v, i) for i, v in enumerate(maxLevelArray[treeIndex]))
        # With pixel Level also store the value with which the tree is filled.
        for index in IndexList:
            pass_Image[index] = PixelVal
            pass_LevelImage[index] = PixelTreeIndex

    return [pass_Image, pass_LevelImage]
