#! /usr/bin/env python
"""rd_normanhno3.py -  specify the refractive indices"""
def rd_normanhno3():
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
# here therer is only  one input data file
    listset=["norman_hno3_h2o.dat"]

    nsets=1
    listset0=["Norman hno3 h2o"]

# *******
# write out all possible files to choose from
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_normanhno3: file listset ")
        for i in range(0,nsets):
            file_object.write("\n")
            a0=listset0[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Juset have one input data file here
    jset=0
#   titlegr=listset0[jset] 

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
    listsub="single_files/"
    fdat=listset[jset]

    filework=froot+listsub+fdat

    icheckfile=1
    if icheckfile ==1:
        file_object.write("\n")
        file_object.write(" filework\n ")
        file_object.write(str(filework))
#       sys.exit()

    if jset == 0:
      ncomp=5
      listset=['0   35% 220 K ','1   45% 220 K ','2   54% 220 K ','3   63% 220 K ','4   70% 220 K']

# real values 
    ndat1=2047

# imaginary values 
    ndat2=ndat1

    i1=18
    i2=i1+ndat1-1

    i3=2068
    i4=i3+ndat2-1

# Will read in the original data, then will need to interpolate to imaginary wavelength scale

# *******
# write out all possible hmo3 values
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_normanhno3: weight hno3, temperature listset ")
        for i in range(0,ncomp):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_normanhno3: weight hno3, temperature listset ")
        for i in range(0,ncomp):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset, 0,1,2,..  ")

# Will specify jset e.g. jset=2
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Norman HNO3 "+listset[jset] 

# *****************
# Will read in from the input ascii file

# Will store all of the data in the rndat and ridat arrays

# note use of ndat1
    wavedat2real=np.zeros((ndat1,ncomp))
    wcmdat2real=np.zeros((ndat1,ncomp))
    rndat=np.zeros((ndat1,ncomp))

# note use of ndat2
    wavedat2imag=np.zeros((ndat2,ncomp))
    wcmdat2imag=np.zeros((ndat2,ncomp))
    ridat=np.zeros((ndat2,ncomp))

# Output arrays
# note use of ndat2
    ndat=ndat2
    wavedat=np.zeros(ndat)
    wcmdat=np.zeros(ndat)
    rnval=np.zeros(ndat)
    rival=np.zeros(ndat)

# Open input ascii file for reading
    f = open(filework,'r')

# *****************

# Data: Real and imaginary indices of aqueous HNO3/H2O at 220 K
# from 754 to 4700 cm-1 for 35, 45, 54, 63, and 70% HNO3
# by weight.

# Reference: Norman, M. L., J. Qian, R. E. Miller, and
# D. R. Worsnop, Infrared complex refractive indices of
# supercooled liquid HNO3/H2O aerosols, J. Geophys. Res.,
# vol. 104, pgs. 30571-30584, 1999.

# Email contact person: R. E. Miller (remiller@unc.edu)

# Format: 2047 real indices (2x,f7.2,2x,f10.4,5(2x,f5.3))
#         2047 imaginary indices (2x,f7.2,2x,f10.4,5(1x,e10.3))

#  cm-1        microns        real indices
#                      35%    45%    54%    63%    70%
#  754.21     13.2588  1.511  1.551  1.577  1.559  1.520
#  756.14     13.2250  1.510  1.550  1.579  1.560  1.522

# *****************
# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

        nadd1=0
        nadd2=ncomp
        idiff=i1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            jj=1
            for j in range(nadd1,nadd2):
                wavedat2real[ii,j]=1.0e4/a1
                wcmdat2real[ii,j]=a1
                jj=jj+1
                rndat[ii,j]=float(columns[jj])

        nadd1=0
        nadd2=ncomp
        idiff=i3
        if i >= i3 and i <= i4:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            jj=1
            for j in range(nadd1,nadd2):
                wavedat2imag[ii,j]=1.0e4/a1
                wcmdat2imag[ii,j]=a1
                jj=jj+1
                ridat[ii,j]=float(columns[jj])

# Close the input ascii file
    f.close()

# *****************
# check that you read in the input data correctly 
    icheckdat=0
    if icheckdat == 1:

# real values
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_normanhno3: i,wavedat2real,wcmdat2real,rndat ")
        nskip=int(ndat1/30)
        for i in range(1,ndat1,nskip):
            file_object.write("\n")
            a0="% 4d" %i
            a1="% 11.4f" %wavedat2real[i,0]
            a2="% 11.4f" %wcmdat2real[i,0]
            a3="% 10.4f" %rndat[i,0]
            listn1=[a0,a1,a2,a3]
            file_object.write(str(listn1))

# imaginary values
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_normanhno3: i,wavedat2imag,wcmdat2imag,ridat ")
        nskip=int(ndat2/30)
        for i in range(0,ndat2,nskip):
            file_object.write("\n")
            a0="% 4d" %i
            a1="% 11.4f" %wavedat2imag[i,0]
            a2="% 11.4f" %wcmdat2imag[i,0]
            a3="% 10.4f" %ridat[i,0]
            listn1=[a0,a1,a2,a3]
            file_object.write(str(listn1))

        sys.exit()

# *****************
# First, find the real indices in the imaginary wavelength grid
# interpolate for rn in the imaginary wavelength scale
    ndat=ndat2

# Output arrays
    wavedat=np.zeros(ndat)
    wcmdat=np.zeros(ndat)
    rnval=np.zeros(ndat)
    rival=np.zeros(ndat)

# Place the input data into the rnval and rival vectors
    a1=wavedat2imag[0,jset]
    a2=wavedat2imag[1,jset]
    if a1 < a2:
        for i in range(0,ndat):
            wavedat[i]=wavedat2imag[i,jset]
            wcmdat[i]=wcmdat2imag[i,jset]
            rnval[i]=rndat[i,jset]
            rival[i]=ridat[i,jset]
    if a2 < a1:
        i2=ndat
        for i in range(0,ndat):
            i2=i2-1
            wavedat[i]=wavedat2imag[i2,jset]
            wcmdat[i]=wcmdat2imag[i2,jset]
            rnval[i]=rndat[i2,jset]
            rival[i]=ridat[i2,jset]
    
# *****************
# Write out results
    if nopr == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_normanhno3: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_normanhno3: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_normanhno3: filework ")
        file_object.write(str(filework))

        nskip=int(ndat/30)

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_normanhno3: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_normanhno3.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_normanhno3() 
