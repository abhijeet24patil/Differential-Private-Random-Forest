import kfold as k
import DiffPRandomForest as dg

dat=dg.createDataSet('car.txt')
fold=k.details1(dat[1:],10)

l=dat[0]
z1,kf=k.parti(fold,2)
ss=[]
for f in kf:
    ss.extend(f)
ss.insert(0,l)


avg=0
me=[]
print(len(dat))
for i in range(1):
    g=dg.randomforest(ss)

    v=[]
    for f in z1:
        f1=f[:-1]
        f2=f[-1]
        b=dg.vote(f1,g,l)
        if b==f2:
            a='t'
        else:a='f'
        v.extend(a)
    tf=['t','f']
    t1=v.count('t')
    fa=v.count('f')
    ##print(t1/len(z1))
    me.append((t1/len(z1))*100)
    avg=avg+(t1/len(z1))


print('avg accuracy:',(avg/1)*100)
import numpy as np,math
print(me)
print(np.std(me))


