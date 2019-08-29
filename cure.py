#!/usr/bin/env python
#om

import sys
import os
import pandas as pd
import numpy as np
import math
import heapq
from itertools import combinations

def findCentroid(sampleNP,k):
    k = int(k)
    #print("in find centroid")
    sampleSize = np.size(sampleNP,0)
    distances = {}
    points = {}
    sampleNPcopy = sampleNP
    for i in range(sampleSize):
        first = sampleNP[i]
        for j in range(i+1,sampleSize):
            second = sampleNP[j]
            d1 = math.pow((second[2]-first[2]),2)
            d2 = math.pow((float(second[3])-float(first[3])),2)
            d3 = math.pow((float(second[4])-float(first[4])),2)
            d4 = math.pow((float(second[5])-float(first[5])),2)
            diff = (d1 + d2 + d3 + d4)**(0.5)
            tup = (float(second[0]),float(first[0]))
            if diff not in distances:
                res = [tup]
                distances[diff] = res
            else:
                res = distances[diff]
                res.append(tup)
                distances[diff] = res
    distanceList = list(distances.keys())
    clusterCount = sampleSize
    distanceLoop = {}
    minVal = min(distanceList)
    clusterList = {}
    clusterListCount = 0
    while (clusterCount != k):
        minClusters = distances[minVal]
        if (len(minClusters)>1):
            minClusters = sorted(minClusters, key=lambda x: x[0])
            minClusters = sorted(minClusters, key=lambda x: x[1])
            minClusters = minClusters[0]
        else:
            minClusters = minClusters[0]
        #print("minClusters:",minClusters)
        firstLoc = int(np.where(sampleNP[:,0] == int(minClusters[0]))[0])
        secondLoc = int(np.where(sampleNP[:,0] == int(minClusters[1]))[0])
        firstCoord = sampleNP[firstLoc]
        secondCoord = sampleNP[secondLoc]
        isClusterFirst = firstCoord[6]
        isClusterSecond = secondCoord[6]
        sampleSize += 1
        weights = isClusterFirst + isClusterSecond
        #newCoord,clusterList = getNewPoint(firstCoord,secondCoord,clusterList,sampleNPcopy)
        x1 = (firstCoord[2]*isClusterFirst+secondCoord[2]*isClusterSecond)/weights
        x2 = (firstCoord[3]*isClusterFirst+secondCoord[3]*isClusterSecond)/weights
        x3 = (firstCoord[4]*isClusterFirst+secondCoord[4]*isClusterSecond)/weights
        x4 = (firstCoord[5]*isClusterFirst+secondCoord[5]*isClusterSecond)/weights
        newPoint = [sampleSize,sampleSize,x1,x2,x3,x4,weights]
        if ((isClusterFirst==1.0) and (isClusterSecond==1.0)):
            ll = [firstCoord[0],secondCoord[0]]
            clusterList[sampleSize] = ll
        elif ((isClusterFirst>1.0) and (isClusterSecond>1.0)):
            val1 = clusterList[firstCoord[0]]
            val2 = clusterList[secondCoord[0]]
            val = val1 + val2
            clusterList[sampleSize] = val
            clusterList.pop(firstCoord[0], None)
            clusterList.pop(secondCoord[0],None)
            #if (val1>val2):
            #    val2 = val2 + val1
            #else:
            #    val1 = val1 + val2
        elif (isClusterFirst>1.0):
            val = clusterList[firstCoord[0]]
            val.append(secondCoord[0])
            clusterList[sampleSize] =val
            clusterList.pop(firstCoord[0], None)
        else:
            val = clusterList[secondCoord[0]]
            val.append(firstCoord[0])
            clusterList[sampleSize] =val
            clusterList.pop(secondCoord[0], None)
        #print("newPoint:",newPoint)
        #print("clusterList so far:",clusterList)
        sampleNP = np.delete(sampleNP,(firstLoc,secondLoc), axis = 0)
        sampleNP = np.vstack([sampleNP,[newPoint]])
        #print("sampleNP now:",sampleNP)
        clusterCount = np.size(sampleNP,0)
        distances = {}
        for i in range(clusterCount):
            first = sampleNP[i]
            for j in range(i+1,clusterCount):
                second = sampleNP[j]
                d1 = math.pow((float(second[2])-float(first[2])),2)
                d2 = math.pow((float(second[3])-float(first[3])),2)
                d3 = math.pow((float(second[4])-float(first[4])),2)
                d4 = math.pow((float(second[5])-float(first[5])),2)
                diff = (d1 + d2 + d3 + d4)**(0.5)
                tup = (float(second[0]),float(first[0]))
                if diff not in distances:
                    res = [tup]
                    distances[diff] = res
                else:
                    res = distances[diff]
                    res.append(tup)
                    distances[diff] = res
        distanceList = list(distances.keys())
        minVal = min(distanceList)
        clusterCount = np.size(sampleNP,0)
        if (clusterCount == k):
            return sampleNP,sampleNPcopy,clusterList

    return sampleNP,sampleNPcopy,clusterList


def repPoints(n,cluster,sample,sampleNP):
    #print("in rep points: ")
    n = int(n)
    repPoints = {}
    #print("cluster:",cluster)
    #print("sample:",sample)
    keyLength = np.size(cluster,0)
    #print("keylength:",keyLength)

    for key in sample.keys():
        val = sample[key]
        valLength = len(val)
        #print("n:",n)
        if (valLength<=n):
            #print("all points")
            #repPoints[key] = val
            newVal = val
        else:
            #print("--------------------------------")
            #print("its more")
            #print("key:",key)
            #print("val:",val)
            loc = int(np.where(cluster[:,0] == float(key))[0])
            #print("loc:",loc)
            #print("cluster[loc]:",cluster[loc])
            newVal = findRep(n,val,sampleNP,cluster[loc])
            #print("--------------------------------")
        if key not in repPoints:
            res = []
            res.append(newVal)
            repPoints[key] = newVal
        else:
            res = repPoints[key]
            res.append(newVal)
            repPoints[key] = res
    return repPoints

def findRep(n,val,sampleNP,center):
    #print("******************************")
    #print("finding rep")
    #print("val:",val)
    #print("center:",center)
    valLength = len(val)
    #print("valLength:",valLength)
    #print("sampleNP:",sampleNP)
    distList = []
    repList = []
    #centerN = center[key]
    for i in range(valLength):
        point = val[i]
        #print("point:",point)
        firstCoord = sampleNP[int(point)]
        #print("firstCoord:",firstCoord)
        distance = math.sqrt((pow((center[2]-firstCoord[2]),2) + pow((center[3]-firstCoord[2]),2) + pow((center[4]-firstCoord[4]),2) + pow((center[5]-firstCoord[5]),2)))
        distList.append((distance,point))
    distList = sorted(distList, key = lambda x: x[0])
    distListPoints = np.array(distList)
    distListPoints = list(distListPoints[:,1])
    distListPoints = distListPoints[:-1]
    #print("distList:",distList)
    #print("distListPoints:",distListPoints)
    maxDist = distList[-1]
    distList = distList[:-1]
    #print("maxDist:",maxDist)
    repList.append(maxDist[1])
    #print("repListSoFar:",repList)
    #print("distListPoints:",distListPoints)
    #print("distList:",distList)
    n -= 1
    r = len(distListPoints)
    if (n==0):
        return repList
    for i in range(n):
        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        r = len(distListPoints)
        #print("distListHere:",distListPoints)
        dList = []
        for j in range(r):
            c = len(repList)
            d=100
            coordNum = distListPoints[j]
            coord = sampleNP[int(coordNum)]
            #print("coord:",coord)
            #print("c:",c)
            for k in range(c):
                p = repList[k]
                #print("p:",p)
                point = sampleNP[int(p)]
                #print("point:",point)
                distance = math.sqrt((pow((point[2]-coord[2]),2) + pow((point[3]-coord[3]),2) + pow((point[4]-coord[4]),2) + pow((point[5]-coord[5]),2)))
                if (distance<d):
                    d = distance
                    distanceCoord = coord[0]
            dList.append([distanceCoord,distance])
            #print("dList:",dList)
        dList = sorted(dList, key = lambda x: x[1])
        maxV = dList[-1]
        #print("maxV:",maxV)
        maxCoord = maxV[0]
        #print("maxCoord:",maxCoord)
        repList.append(maxCoord)
        distListPoints=np.array(dList)
        distListPoints = list(distListPoints[:,0])
        distListPoints = distListPoints[:-1]
        #distListPoints = list(filter(lambda x:x!=float(maxCoord),distListPoints))
        ##print("distListPoints:",distListPoints)
        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    #print("repList:",repList)
    return repList

def movePoints(repPoints,cluster,sampleNP,alpha):
    #print("in Moved points")
    #print("repPoints:",repPoints)
    #print("cluster:",cluster)
    alpha = float(alpha)
    movedPoints = {}
    for key in repPoints.keys():
        val = repPoints[key]
        #print("val:",val)
        valLength = len(val)
        loc = int(np.where(cluster[:,0] == float(key))[0])
        centroid = cluster[loc]
        #print("centroid:",centroid)
        x1 = centroid[2]
        x2 = centroid[3]
        x3 = centroid[4]
        x4 = centroid[5]
        for j in range(valLength):
            p = val[j]
            row = sampleNP[int(p)]
            y1 = row[2] + ((x1 - row[2])*alpha)
            y2 = row[3] + ((x2 - row[3])*alpha)
            y3 = row[4] + ((x3 - row[4])*alpha)
            y4 = row[5] + ((x4 - row[5])*alpha)
            if key not in movedPoints:
                res = []
                center = [j,y1,y2,y3,y4]
                res.append(center)
                movedPoints[key] = res
            else:
                #print("its here!")
                #print("move Points so far: ", movePoints)
                res = movedPoints[key]
                #print("res: ", res)
                center = [j,y1,y2,y3,y4]
                res.append(center)
                movedPoints[key] = res

    return movedPoints

def findFinalClusters(movedPoints,k,data):
    #print("findFinalClusters")
    #print("movedPoints:",movedPoints)
    #print("data:",data)
    finalCluster = {}
    k = int(k)
    dataSize = np.size(data,0)
    #print("dataSize:",dataSize)
    for i in range(dataSize):
        point = data[i]
        distance = []
        #print("point:",point)
        for key in movedPoints.keys():
            val = movedPoints[key]
            #print("key:",key)
            minDist = 1000
            #print("val:",val)
            valLength = len(val)
            x1 = float(point[1])
            x2 = float(point[2])
            x3 = float(point[3])
            x4 = float(point[4])
            for j in range(valLength):
                second = val[j]
                #print("second:",second)
                y1 = float(second[1])
                y2 = float(second[2])
                y3 = float(second[3])
                y4 = float(second[4])
                dist = math.sqrt((pow((x1-y1),2) + pow((x2-y2),2) + pow((x3-y3),2) + pow((x4-y4),2)))
                if (dist<minDist):
                    minDist = dist
                    minPoint = key
            distance.append([minPoint,minDist])
        #print("distance:",distance)
        distance = sorted(distance,key = lambda x: x[1])
        minDist = distance[0]
        minCluster = int(minDist[0])
        #print("point:",point)
        p = float(point[0])
        p = int(p)
        #print("p:",p)
        if minCluster not in finalCluster:
            res = [p]
            finalCluster[minCluster] = res
        else:
            res = finalCluster[minCluster]
            res.append(p)
            finalCluster[minCluster] = res
                #dist = math.pow()
    #for i in
    return finalCluster

def getStats(finalCluster,data):
    precision = 0
    recall = 0
    #print("finalCluster:",finalCluster)
    correctPairs = []
    predictedPairs = []
    dataSize = np.size(data,0)
    for i in range(dataSize):
        point = data[i]
        #print("point:",point)
        label = point[5]
        #print("label:",label)
        for j in range(i+1,dataSize):
            point2 = data[j]
            label2 = point2[5]
            if (label2 == label):
                #print("they're same")
                tup = (int(float(point[0])),int(float(point2[0])))
                correctPairs.append(tup)
    #print("correctPairs:",correctPairs)
    for key in finalCluster.keys():
        val = finalCluster[key]
        pairs = list(combinations(val,2))
        predictedPairs.append(pairs)
    total = []
    for i in predictedPairs:
        total += i
    totalLength = len(total)
    correctLength =len(correctPairs)
    #print("predictedPairs:",total)
    intersection = len(list(set(total).intersection(correctPairs)))
    #print("intersection:",intersection)
    precision = intersection/totalLength
    recall = intersection/correctLength
    return precision,recall


num_args = len(sys.argv)
if (num_args!=6):
    print("Insufficient arguments")
    print("Usage: cure.py <k> <sample> <data-set> <n> <alpha>")
    exit(-1)

k = sys.argv[1]
sample = sys.argv[2]
data = sys.argv[3]
n = sys.argv[4]
alpha = sys.argv[5]

fOpen = open(sample,"r")
fOpen2 = open(data,"r")
sampleArr = []
dataArr = []
lineCount = 0
for i in fOpen2:
    line = i.rstrip()
    line = line.split(",")
    line = [lineCount] + line
    label = line[-1]
    lineMap = list(map(float,line[:-1]))
    lineMap.append(label)
    dataArr.append(lineMap)
    lineCount +=1

#print("dataArr: ",dataArr)
dataNP = np.array(dataArr)
#print("dataNP: ",dataNP)
dataNP_train = dataNP[:,:-1]
#print("dataNP_train:",dataNP_train)

lineCount = 0
for i in fOpen:
    line = int(i)
    lineArr = [lineCount] + dataArr[line]
    sampleArr.append(lineArr)
    lineCount += 1


sampleNP = np.array(sampleArr)
sampleNPtrain = sampleNP[:,:-1]
#print("sampleNP: ", sampleNPtrain)
sampleNPtrain = np.append(sampleNPtrain,np.ones([len(sampleNPtrain),1]),1)
samplePD = pd.DataFrame(sampleNPtrain)
samplePD[0] = pd.to_numeric(samplePD[0])
samplePD[1] = pd.to_numeric(samplePD[1])
samplePD[2] = pd.to_numeric(samplePD[2])
samplePD[3] = pd.to_numeric(samplePD[3])
samplePD[4] = pd.to_numeric(samplePD[4])
samplePD[5] = pd.to_numeric(samplePD[5])
samplePD[6] = pd.to_numeric(samplePD[6])
sampleNPtrain = samplePD.as_matrix()
#print("sampleNPtrain:",sampleNPtrain)
res,res2,res5 = findCentroid(sampleNPtrain,k)
#print(res)
#res3,center = hierarchicalCluster(res,sampleNPtrain,int(k))
#print("res5:",res5)
#print("----------------------")
resLength = np.size(res,0)
for i in range(resLength):
    val = res[i]
    if (int(val[6]) == 1):
        res5[int(val[0])] = [val[0]]
#print("res5:",res5)
for key in res5.keys():
    val = res5[key]
    #print("val:",val)
    val = list(map(int,val))
    #print(val)
    #print("-------------------------")

#print("res:",res)

#print("res4:",res4)
#print("res2:",res2)
res6 = repPoints(n,res,res5,res2)
#print("res6:")
for key in res6.keys():
    val = res6[key]
    #print(val)

#print("-----------------------")
for key in res5.keys():
    val = res5[key]
#    print("val:",val)
    val = list(map(int,val))
    #print(val)
    #print("-------------------------")

res7 = movePoints(res6,res,sampleNPtrain,alpha)
#print("res7")
#print("\n")
for key in res7.keys():
    val = res7[key]
    val = sorted(val, key = lambda x: x[1])
    res7[key] = val
    #print(val)
    #print("\n")
    #print("----------------------")

res8 = findFinalClusters(res7,k,dataNP_train)
count = 0

finalCluster = {}

print("\n")
for key in res8.keys():
    strconcat = "Cluster " + str(count) + ":"
    print(strconcat,end=" ")
    print(res8[key])
    print("\n")
    finalCluster[count] = res8[key]
    count +=1

    #print("-----------------------------------------")

precision,recall = getStats(finalCluster,dataNP)
print("Precision = ",precision, end = " ")
print(", recall =",recall)
