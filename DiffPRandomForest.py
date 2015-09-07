import mdrf1 as mf
import random
from operator import*
from math import*
import operator

def randomforest(dataSet):
     sub=[]
     cT=[]
     B=15
     d=5
     eps=B/20
##     print(eps)
     ep=eps/12

     bStrap=bootstrap(dataSet)
     for s in bStrap:
          l0=s[0]
          A=l0[:-1]
          c=[ex[-1]for ex in s[1:]]
          c1=set(c)
          C=list(c1)
          crT=mf.Build_DiffPID3(s[1:],A,C,d,ep)
          cT.append(crT)
     return cT
def bootstrap(dat):
     l=len(dat)
     bs=[]
     bbs=[]
     p=dat[0]
     for x in range(20):
          bs=sampl(dat)
          bbs.append(bs)
     for i in range(len(bbs)):
          bbs[i].insert(0,p)
     return bbs

def sampl(dat):
     w=[]
     l=len(dat)
     for i in range(l-1):
          oo=random.choice(dat[1:])
          w.append(oo)
          #ww=mat(w)
     return w

def createDataSet(fileName):
     dst=[]
     fr=open(fileName)
     for line in fr.readlines():
          curLine=line.strip().split('\t')
          rr=[]
          for word in curLine:
               rr=word.strip().split(',')
          dst.append(rr)
     return dst
##function to classify new sample
def classify(inputTree,featLabels,testVec):
     firstStr = list(inputTree.keys())
     fs=firstStr[0]
     classLabel=0
     secondDict = inputTree[fs]
     featIndex = featLabels.index(fs)
     for key in secondDict.keys():
          if testVec[featIndex] == key:
               if type(secondDict[key]).__name__=='dict':
                    classLabel = classify(secondDict[key],featLabels,testVec)
               else:
                    classLabel = secondDict[key]
     return classLabel

def vote(featValue,trees,flabel):
     cl=[]
     dic={}
     for tree in trees:
          d=classify(tree,flabel,featValue)
          cl.append(d)
     for c in cl:
        if c in dic.keys():
            dic[c]+=1
        else:dic[c]=1

     sss=sorted(dic.items(),key=operator.itemgetter(1), reverse=True)
     return sss[0][0]
