#! /usr/bin/env python
"""calc_samewave.py -  the rvelengths on same grid"""
def calc_samewave():
    import os
    import math
    import numpy as np
    import pickle

# *****************
# Append to the output f.out ascii file
    fileout="f.out"
    file_object = open(fileout,'a')

    nopr=1

# *****************
# Since nlinesi ne nlinesr, you will use samewave.py to get values of rn at ri wavelengths
#   list6=[nlinesr,wcmr,wavelengthr,rndat,nlinesi,wcmi,wavelengthi,ridat,titlegr]
#   print("list6 from rd-xxxx.py ",list6)

    filepickle = open("diffindxwave",'rb')
    list6=pickle.load(filepickle)
    filepickle.close()

    nlinesr=list6[0]
    wcmr=list6[1]
    wavelengthr=list6[2]
    rndat=list6[3]

    nlinesi=list6[4]
    wcmi=list6[5]
    wavelengthi=list6[6]
    ridat=list6[7]

    titlegr=list6[8]

# *************************
# Write out results
    iwrsp=1
    if iwrsp == 1:

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_samewave: nlinesi,nlinesr ")
        a0="% 4d" %nlinesi
        a1="% 4d" %nlinesr
        listn=[a0,a1]
        file_object.write(str(listn))

        iskipi=int(nlinesi/20)
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_samewave: i,wavelengthi,wcmi,ridat ")
        for i in range(0,nlinesi,iskipi):
            file_object.write("\n")
            a0="% 4d" %i
            a1="% 11.4f" %wavelengthi[i]
            a2="% 11.4f" %wcmi[i]
            a3="% 10.4f" %ridat[i]
            listn1=[a0,a1,a2,a3]
            file_object.write(str(listn1))

        iskipr=int(nlinesr/20)
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_samewave: i,wavelengthr,wcmr,rndat ")
        for i in range(0,nlinesr,iskipr):
            file_object.write("\n")
            a0="% 4d" %i
            a1="% 11.4f" %wavelengthr[i]
            a2="% 11.4f" %wcmr[i]
            a3="% 10.4f" %rndat[i]
            listn1=[a0,a1,a2,a3]
            file_object.write(str(listn1))
    
# *************************
# Put values into temporary rndat2 and ridat2 arrays
    rndat2=np.zeros(nlinesr)
    for i in range(0,nlinesr):
        rndat2[i]=rndat[i]

    ridat2=np.zeros(nlinesi)
    for i in range(0,nlinesi):
        ridat2[i]=ridat[i]

# *************************
# Note ranges to work with
    wavelminr=1.0e6
    wavelmaxr=-1.0e6
    for i in range(0,nlinesr):
        if wavelengthr[i] < wavelminr:
            wavelminr=wavelengthr[i]
        if wavelengthr[i] > wavelmaxr:
            wavelmaxr=wavelengthr[i]

    wavelmini=1.0e6
    wavelmaxi=-1.0e6
    for i in range(0,nlinesi):
        if wavelengthi[i] < wavelmini:
           wavelmini=wavelengthi[i]
        if wavelengthi[i] > wavelmaxi:
            wavelmaxi=wavelengthi[i]

# print,'  samewave: nlinesi,nlinesr ',nlinesi,nlinesr
# print,'  samewave: wavelmini,wavelminr  ',wavelmini,wavelminr 
# print,'  samewave: wavelmaxi,wavelmaxr  ',wavelmaxi,wavelmaxr 

# *************************
# Case where imaginary wavelength range is less than real range
# e.g. rdliu_soa_acp.pro
    if wavelmini <= wavelminr and wavelmaxi <= wavelmaxr:

        for i in range(0,nlinesi):
            if wavelengthi[i] >= wavelengthr[0]:
                istart=i
                break

# *****************
        nlines=nlinesi-istart

        wcmdat=np.zeros(nlines)
        wavedat=np.zeros(nlines)
        rnval=np.zeros(nlines)
        rival=np.zeros(nlines)

# *****************
# The imaginary values are the primary ones to work with
        m=istart-1
        for i in range(0,nlines):
            m=m+1
            rival[i]=ridat2[m]
            wcmdat[i]=wcmi[m]
            wavedat[i]=wavelengthi[m]

# *****
# Find first match
        i=1
        diffmin=1.0e6
        istart=-99
        for ii in range(0,nlinesr):
            diff=abs(wavelengthr[ii]-wavedat[i])
            if diff < diffmin:
                istart=ii
                diffmin=diff
# **
# Specify the rival values
        idiff=5
        ii=istart
        for i in range(0,nlines):
   
           j1=ii-idiff
           j2=ii+idiff
           if j1 < 0:
               j1=0
           if j2 > nlinesr:
               j2=nlinesr

           diffmin=1.0e6
           iuse=-99
           for j in range(j1,j2):
              diff=abs(wavelengthr[j]-wavedat[i])
              if diff < diffmin:
                  iuse=j
                  ii=j
                  diffmin=diff

           rnval[i]=0.0
           if iuse >= 0 and iuse < nlinesr:
               rnval[i]=rndat2[iuse]

# *************************
    ndat=nlines

# *************************
# Write out results
    if nopr == 1:
        a0=titlegr
        a1='  '
        a2=ndat
        listn0=[a0,a1,a2]
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_samewave: titlegr,ndat")
        file_object.write(str(listn0))

        nskip=int(ndat/20)
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_samewave: i,wavedat,wcmdat,rnval,rival ")
        for i in range(0,ndat,nskip):
            file_object.write("\n")
            a0="% 4d" %i
            a1="% 11.4f" %wavedat[i]
            a2="% 11.4f" %wcmdat[i]
            a3="% 10.4f" %rnval[i]
            a4="% 10.4f" %rival[i]
            listn1=[a0,a1,a2,a3,a4]
            file_object.write(str(listn1))

# *****************
    ichecktype=0
    if ichecktype == 1:
        print(wavedat.dtype)
        print(wcm.dtype)
        print(rndat.dtype)
        print(ridat.dtype)

# *****************
# Write out results to the list5 pickle
    list5=[ndat,wavedat,wcmdat,rnval,rival,titlegr]
#   print("list5 from calc_samewave.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

calc_samewave() 
