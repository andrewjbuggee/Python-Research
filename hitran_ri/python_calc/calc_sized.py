#! /usr/bin/env python
"""calc_sized.py -  calculate the particle size distribution"""
def calc_sized():
    import os
    import math
    import numpy as np
    import pickle

# *****************
# Open the output f.out ascii file
    fileout="f.out"
    file_object = open(fileout,'a')

    nopr=1

# *****************
# Obtain the data from init_calc.py
#   list1=[iwave,w1,w2,dw,den1,rad1,sig1,den2,rad2,sig2,r1,r2]
#   print("list1 from init_calc.py ",list1)

    filepickle = open("sizedistparam",'rb')
    list1=pickle.load(filepickle)
    filepickle.close()

    iwave=list1[0]
    w1=list1[1]
    w2=list1[2]
    dw=list1[3]
    den1=list1[4]
    rad1=list1[5]
    sig1=list1[6]
    den2=list1[7]
    rad2=list1[8]
    sig2=list1[9]
    r1=list1[10]
    r2=list1[11]

# *****************
# Write out input
    iwrinput=1
    if iwrinput == 1:
        listn1=[iwave,w1,w2,dw]
        listn2=[den1,rad1,sig1]
        listn3=[den2,rad2,sig2]
        listn4=[r1,r2]

#       file_object.write("\n")
#       file_object.write("\n")
#       file_object.write(" calc_sized: iwave,w1,w2,dw\n ")
#       file_object.write(str(listn1))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_sized: input den1,rad1,sig1\n ")
        file_object.write(str(listn2))

        file_object.write("\n")
        file_object.write(" calc_sized: input den2,rad2,sig2\n ")
        file_object.write(str(listn3))

        file_object.write("\n")
        file_object.write(" calc_sized: input r1,r2\n ")
        file_object.write(str(listn4))

# *****************
    r1log=math.log(r1)
    r2log=math.log(r2)

    ndist2=200

    radr2=np.zeros(ndist2)
    sized2=np.zeros(ndist2)
    dr2=np.zeros(ndist2)

    pi=3.14159265
    const=math.sqrt(2.0*pi)

    s1=math.log(sig1)
    if sig2 > 1.0:
        s2=math.log(sig2)
    else: 
        pass
    if sig2 < 1.0:
        den2=0.0
    else: 
        pass

    dlog=(r2log-r1log)/(1.00*ndist2)

# Special write 
    iwrsp=0
    if iwrsp == 1:
        listn5=[r1log,r2log,ndist2,s1,s2,dlog]

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_sized: r1log,r2log,ndist2,s1,s2,dlog\n ")
        file_object.write(str(listn5))
    
# Calculate the size distribution
    ndist2m1=ndist2-1
    for i in range(0,ndist2):

        term1=0.0
        term2=0.0

        rlog=r1log+(i*dlog)
        b1=math.exp(rlog)
        radr2[i]=b1

        if den1 >= 1.0e-6:
            alpha1=(math.log(b1/rad1))/s1
            a2=(alpha1*alpha1)/2.0
            term1=den1*(1.0/(b1*s1*const))*math.exp(-a2)

        if den2 >= 1.0e-6:
            alpha2=(math.log(b1/rad2))/s2
            a4=(alpha2*alpha2)/2.0
            term2=den2*(1.0/(b1*s2*const))*math.exp(-a4)

        sum=term1+term2
        sized2[i]=sum

# *************************
# Calculate the radii bin
    for i in range(0,ndist2m1):
        i2=i+1
        dr2[i]=radr2[i2]-radr2[i]

    i3=ndist2m1
    i2=ndist2m1-1
    dr2[i3]=dr2[i2]

# *************************
# Just use values that are greater than 1.0e-12

# **
    ifirst=-99
    i=-1
    while i >= -1:
        i=i+1
        if sized2[i] >= 1.0e-12:
            ifirst=i
            break

# **
    ilast=-99
    i=ndist2
    while i >= 0:
        i=i-1
        if sized2[i] >= 1.0e-12:
            ilast=i
            break

# **
#   ifirst=0
#   ilast=ndist2m1

    if iwrsp == 1:
        listn8=[ifirst,ilast]
        file_object.write("\n")
        file_object.write(" calc_sized: ifirst,ifirst\n ")
        file_object.write(str(listn8))

# **
# Redefine the size distribution
    iok=0
    if ifirst >= 0 and  ilast >= 0:
        iok=1
        ndist=ilast-ifirst+1
        radr=np.zeros(ndist)
        sized=np.zeros(ndist)
        dr=np.zeros(ndist)
        j=-1
        i3=ilast+1
        for i in range(ifirst,i3):
            c1=radr2[i]
            c2=sized2[i]
            c3=dr2[i]

            j=j+1
            radr[j]=c1
            sized[j]=c2
            dr[j]=c3

# *************************
# Write out results
    if nopr == 1:

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_sized: ndist\n ")
        file_object.write(str(ndist))

        nskip=int(ndist/20)

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_sized: i,radr,dr,sized ")
        for i in range(0,ndist,nskip):
            file_object.write("\n")
            a0="% 4d" %i
            a1="% 10.3e" %radr[i]
            a2="% 10.3e" %dr[i]
            a3="% 10.3e" %sized[i]
            listn1=[a0,a1,a2,a3]
            file_object.write(str(listn1))

# *****************
    ichecktype=0
    if ichecktype == 1:
        print(radr.dtype)
        print(sized.dtype)
        print(dr.dtype)

# *****************
# One way to pass data out of a function is to   pickle   the data
    list4=[ndist,radr,sized,dr]
#   print("list4 from calc_sized.py ",list4)

    filepickle = open("sizedist",'wb')
    pickle.dump(list4,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

calc_sized() 
