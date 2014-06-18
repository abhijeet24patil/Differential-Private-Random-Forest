import random
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


def details(filename,k):
    dataSet=createDataSet(filename)
    vc=vvv(dataSet)
    k_fold=[]
    le=[]
    for t in vc:
        f=len(t)
        le.append(f)
    for i in range(k):
        f=part(vc,le,k)
        k_fold.append(f)
    return k_fold

def details1(dats,k):
##    dataSet=createDataSet(filename)
    vc=vvv(dats)
    k_fold=[]
    le=[]
    for t in vc:
        f=len(t)
        le.append(f)
    for i in range(k):
        f=part(vc,le,k)
        k_fold.append(f)
    return k_fold



def part(dat,le,k):
    par=[]
    for i in range(len(dat)):
        z=random.sample(dat[i],int(le[i]/k))
        par.extend(z)
        for j in z:
            ind=dat[i].index(j)
            del(dat[i][ind])
    return par



def vvv(dataset):
    clist=[ex[-1]for ex in dataset]
    w2=set(clist)
    ccc=[]
    for i in w2:
        d1=split1(i,dataset)
        ccc.append(d1)
    return ccc

def split1(i,dataSet):
	xx=[]
	for j in range(len(dataSet)):
		if dataSet[j][-1]==i:
			t=dataSet[j]
			xx.append(t)
	return xx

def parti(data,i):
    x=data[i+1:]
    y=data[:i]
    kf=y+x
    return data[i],kf
