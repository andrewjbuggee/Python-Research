#! /usr/bin/env python
"""init_optest.py -  do the simple optimal estimation calculation"""
def optest_calc():
    import os
    import numpy as np

# *****************
# Open the output f.out ascii file
    fileout="f.out"
    file_object = open(fileout,'a')

    nopr=1

# *****************
# Obtain the data from init_calc.py
#   list1=[nvar,nobs,yobs,xa,kmatrix,se,sa,xactual,perc,apcov,errcov]
#   print("list1 from init_calc.py ",list1)
    import pickle
    filepickle = open("stateinit",'rb')
    list1=pickle.load(filepickle)
#   file.close(filepickle)
    filepickle.close()

    nvar=list1[0]
    nobs=list1[1]
    yobs=list1[2]
    xa=list1[3]
    kmatrix=list1[4]
    se=list1[5]
    sa=list1[6]
    xactual=list1[7]
    perc=list1[8]
    apcov=list1[9]
    errcov=list1[10]

# *****************
# the solution vector
    xsolution=np.zeros(nvar)

# The percent difference in the xsolution and the xactual values
    xperc=np.zeros(nvar)

# *****************
# xsoln = xa + (KT Se-1 K  + Sa-1)-1 (KT Se-1) (y - K xa)

    ydiff=np.zeros(nobs)
    ksa = np.dot(kmatrix,xa)
    ydiff=yobs-ksa

    kt=kmatrix.T
    sem1 = np.linalg.inv(se)
    ktsem1=np.dot(kt,sem1)

    bmat=np.dot(ktsem1,ydiff)

    part1 = np.dot(ktsem1,kmatrix)
    sam1 = np.linalg.inv(sa)
    sum = part1 + sam1 
    summ1 = np.linalg.inv(sum)

    xsolution = xa + np.dot(summ1,bmat)

# *****************
# The percent difference in the xsolution and xactual values
    xperc=100.0*(xsolution-xactual)/xactual

# *****************
# Write out results
    if nopr == 1:

        listn=[nvar,nobs]

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" optest_calc: nvar,nobs\n ")
        file_object.write(str(listn))

        file_object.write("\n")
        file_object.write(" optest_calc: xsolution\n ")
        file_object.write(str(xsolution))

        file_object.write("\n")
        file_object.write(" optest_calc: xactual\n ")
        file_object.write(str(xactual))

        file_object.write("\n")
        file_object.write(" optest_calc: random perc\n ")
        file_object.write(str(perc))

        file_object.write("\n")
        file_object.write(" optest_calc: xperc\n ")
        file_object.write(str(xperc))

# *****************
    ichecktype=0
    if ichecktype == 1:
        print(xsolution.dtype)

# *****************
# Close the output f.out ascii file
    file_object.close()

optest_calc() 
