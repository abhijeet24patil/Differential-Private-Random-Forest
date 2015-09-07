import numpy
import math
import random
def Build_DiffPID3(dataset,A,C,d,ep):

    t=max_At(A,dataset)
    Nt=NoisyCount(dataset,ep)
    classList=[ex[-1] for ex in dataset]
    if classList.count(classList[0])==len(classList):
           return classList[0]
    if len(A)==1 or d==0:
##    if len(A)==1 or d==0 or Nt/t*len(C) <= math.sqrt(2)/ep:
##        return majorityCnt(C,ep)
        return majorityCnt1(classList)

    m=int(math.sqrt(len(A)))
    A1=random.sample(A,m)
    ind=[]
    for i in A1:
        ind.append(A.index(i))
##    Atr=InfomGain(ind,ep,dataset,C)
##    Atr=maxoperator(dataset,ind,ep)
    Atr=gini_index(dataset,ind,ep)
    bestFeatLabel = A[Atr]
    myTree = {bestFeatLabel:{}}
    del(A[Atr])
    featValues = [example[Atr] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = A[:]
        myTree[bestFeatLabel][value] = Build_DiffPID3(splitDataSet(dataset, Atr, value),subLabels,C,d-1,ep)
    return myTree



def max_At(A,dat):
    ind=[]
    for i in A:
        a=A.index(i)
        ind.append(a)
    cc=[]
    for i in ind:
        cl=[ex[i]for ex in dat]
        xc=len(set(cl))
        cc.append(xc)
    return max(cc)

def NoisyCount(T,ep):
    Nt=len(T)+numpy.random.laplace(1,1/ep)
    return Nt

def majorityCnt(classList,ep):
     classCount={}
     for vote in classList:
          if vote not in classCount.keys():
            classCount[vote]=1
          else:
            classCount[vote]+=1
     for k in classCount.keys():
         classCount[k]=classCount[k]+numpy.random.laplace(ep)
     for k in classCount.keys():
         if classCount[k]==max(classCount.values()):
             return k

def InformGain(A,ep,dat,C):
##    numFeatures=len(dat[0])-1

    numFeatures=len(A)

    baseEntropy=calcShannonEnt(dat)
    bestInfoGain=0.0;bestFeature=-1

    score={}
    Psc={}
    for i in A:
          featList=[example[i] for example in dat]
          uniqueVal=set(featList)

          newEntropy=0.0
          for value in uniqueVal:
               subDataSet=splitDataSet(dat,i,value)
               Ns=NoisyCount(subDataSet,ep)
               Nt=NoisyCount(dat,ep)
               prob=Ns/Nt
               newEntropy+=prob*calcShannonEnt(subDataSet)
          infoGain=baseEntropy-newEntropy
          score[i]=infoGain
    sens=math.log(len(dat)+1,2)+1/math.log1p(2)
##    sens=float(math.log(len(C),2))
    for k in score.keys():

        if ep*score[k] >709:
            h=ep/len(A)
            Psc[k]=math.e**(h*score[k])
        else:
            Psc[k]=math.e**(ep*score[k])


##        Psc[k]=math.e**(ep*score[k]/sens)

    for k in Psc.keys():
         if Psc[k]==max(Psc.values()):
            return k

##    return random.choice(list(Psc.keys()))



def calcShannonEnt(dataSet):
    numentries=len(dataSet)
    labelCounts={}
    for feat in dataSet:
        currentLabel=feat[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=1
        else:
            labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numentries
        shannonEnt-=prob*math.log(prob,2)
    return  shannonEnt

def splitDataSet(dataSet,axis,value):
     retDataSet=[]
     for feature in dataSet:
          if feature[axis]==value:
               reducedFeat=feature[:axis]
               reducedFeat.extend(feature[axis+1:])
               retDataSet.append(reducedFeat)
     return retDataSet

def maxoperator(dat,A,ep):
    dic={}
    for i in A:
        ftlist=[ex[i]for ex in dat]
        uniq=set(ftlist)
        sumVal=0
        for value in uniq:
            subDataSet=splitDataSet(dat,i,value)
            cl=[ex[-1]for ex in subDataSet]
            setcl=set(cl)
            con=[]
            for k in setcl:
                g=NoisyCount1(cl.count(k),ep)
                con.append(g)
            sumVal+=max(con)
        dic[i]=sumVal

    for k in dic.keys():
        if ep*dic[k] >709:
            h=ep/len(A)
            dic[k]=math.e**(h*dic[k])
        else:
            dic[k]=math.e**(ep*dic[k])
    for k in dic.keys():
        if dic[k]==max(dic.values()):
            return k

def NoisyCount1(T,ep):
    Nt=T+numpy.random.laplace(1,1/ep)

    return Nt


def majorityCnt1(classList):
     classCount={}
     for vote in classList:
          if vote not in classCount.keys():
            classCount[vote]=1
          else:
            classCount[vote]+=1
     for k in classCount.keys():
        if classCount[k]==max(classCount.values()):
            return k





def gini_index(dat,A,ep):
    gg=gini1(dat)
    numfeature=len(dat[0])-1
    dic={}
    Psc={}
    for i in A:
        featList=[example[i] for example in dat]
        unique=set(featList)
        gi=0
        for v in unique:
            subdata=splitDataSet(dat,i,v)
            ll=NoisyCount(subdata,ep)
            gi+=float(ll*gini1(subdata))

        gini2=gg-gi
        dic[i]=gini2

    sens=2
    for k in dic.keys():
        Psc[k]=math.e**(ep*dic[k]/sens)

    for k in Psc.keys():
         if Psc[k]==max(Psc.values()):
            return k

def gini1(dat):
    classlist=[ex[-1]for ex in dat]
    classes=set(classlist)
    dic={}
    for i in classes:
        dic[i]=classlist.count(i)
    sum=0
    for v in dic.values():
        sum=sum+(v/len(dat))**2
    return 1-sum
