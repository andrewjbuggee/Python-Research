#! /usr/bin/env python
"""rd_begemannal2o3.py -  specify the refractive indices"""
def rd_begemannal2o3():
    import sys
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
# Will select the jset compund to work with
    listset0=['0   Begemann Al2O3 Compact ',\
    '1   Begemann Al2O3 Porous']

    listset=['exoplanets/begemann_al2o3_compact.dat',\
    'exoplanets/begemann_al2o3_porous.dat']

# *******
# write out all possible files to choose from
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_begemannal2o3: file listset ")
        for i in range(0,2):
            file_object.write("\n")
            a0=listset0[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_begemannal2o3: file listset ")
        for i in range(0,2):
            a0=listset0[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 1")

# Will specify jset e.g. jset=3
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr=listset0[jset] 

# *****************
# Obtain the data from init_calc.py
#   list0=[fileasciidir,fileasciiroot]
#   print("list0 from init_calc.py ",list0)

    filepickle = open("subdirectories",'rb')
    list0=pickle.load(filepickle)
    filepickle.close()

    fileasciidir=list0[0]
    froot=list0[1]

# *****************
    fdat=listset[jset]

    filework=froot+fdat

    icheckfile=1
    if icheckfile ==1:
        file_object.write("\n")
        file_object.write(" filework\n ")
        file_object.write(str(filework))
#       sys.exit()

# *****************
# Will read in from one of the Begemann ascii files
    ndat=64

# Since you read in from one file, set ncomp=1 and jset=0
    ncomp=1
    jset=0

# Will store all of the data in the rndat and ridat arrays
# There are ndat wavelengths
    wavedat2=np.zeros((ndat,ncomp))
    wcmdat2=np.zeros((ndat,ncomp))
    rndat=np.zeros((ndat,ncomp))
    ridat=np.zeros((ndat,ncomp))

# Output arrays
    wavedat=np.zeros(ndat)
    wcmdat=np.zeros(ndat)
    rnval=np.zeros(ndat)
    rival=np.zeros(ndat)

# Open input ascii file for reading
    f = open(filework,'r')

# *****************
#Reference: Begemann, B., J. Dorschner, Th. Henning, H. Mutschke
#J. Gurtler, C. Kompe, and R. Nass
#Aluminum oxide and the opacity of oxygen-rich circumstellar dust in
#the 12-17 micron range
#Astrophysical Journal, v476, p199-208, 1997

#Real and Imaginary Refractive Indices of compact Al2O3

#Contact: H. Mutschke (harald.mutschke@uni-jena.de)

#Format: 64 lines 2x,2(1x,f9.2),2(1x,f10.4)

#      cm-1    microns     real    imaginary
#    1282.05      7.80     1.3430     0.0780
#    1250.00      8.00     1.3230     0.0860
#    1219.51      8.20     1.3010     0.0950

# *****************
# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

        nadd1=0
        nadd2=ncomp
        i1=14
        i2=i1+ndat-1
        idiff=i1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            a2=float(columns[1])
            jj=1
            for j in range(nadd1,nadd2):
                wavedat2[ii,j]=1.0e4/a1
                wcmdat2[ii,j]=a1
                jj=jj+1
                rndat[ii,j]=float(columns[jj])
                jj=jj+1
                ridat[ii,j]=float(columns[jj])

# Close the input ascii file
    f.close()

# *****************
# Place the input data  into the rnval and rival vectors
    a1=wavedat2[0,jset]
    a2=wavedat2[1,jset]
    if a1 < a2:
        for i in range(0,ndat):
            wavedat[i]=wavedat2[i,jset]
            wcmdat[i]=wcmdat2[i,jset]
            rnval[i]=rndat[i,jset]
            rival[i]=ridat[i,jset]
    if a2 < a1:
        i2=ndat
        for i in range(0,ndat):
            i2=i2-1
            wavedat[i]=wavedat2[i2,jset]
            wcmdat[i]=wcmdat2[i2,jset]
            rnval[i]=rndat[i2,jset]
            rival[i]=ridat[i2,jset]
    
# *****************
# Write out results
    if nopr == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_begemannal2o3: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_begemannal2o3: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_begemannal2o3: filework ")
        file_object.write(str(filework))

        nskip=int(ndat/20)

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_begemannal2o3: i,wavedat,wcmdat,rndex,ridex ")
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
        print(wcmdat.dtype)
        print(rnval.dtype)
        print(rival.dtype)

# *****************
# One way to pass data out of a function is to   pickle   the data
    list5=[ndat,wavedat,wcmdat,rnval,rival,titlegr]
#   print("list5 from rd_begemannal2o3.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_begemannal2o3() 
