#! /usr/bin/env python
"""rd_palmerwilliams.py -  specify the refractive indices"""
def rd_palmerwilliams():
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
    listset=["palmer_williams_h2so4.dat"]

    nsets=1
    listset0=["Palmer  Williams H2SO4 indices"]

# *******
# write out all possible files to choose from
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_palmerwilliams: file listset ")
        for i in range(0,nsets):
            file_object.write("\n")
            a0=listset0[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Just have one input data file here
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
      ncomp=6
      listset=['0   25% 300 K ','1   38% 300 K ','2   50% 300 K ','3   75% 300 K ','4   84.5% 300 K',\
               '5   95.6% 300 K ']

# real values 
    ndat1=227

# imaginary values 
    ndat2=227

    i1=17
    i2=i1+ndat1-1

    i3=249
    i4=i3+ndat2-1

# Will read in the original data, then will need to interpolate to imaginary wavelength scale

# *******
# write out all possible h2so4-temperature combinations
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_palmerwilliams: weight h2so4, temperature listset ")
        for i in range(0,ncomp):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_palmerwilliams: weight h2so4, temperature listset ")
        for i in range(0,ncomp):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset, 0,1,2,..  ")

# Will specify jset e.g. jset=2
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Palmer Williams H2SO4 "+listset[jset] 

# *****************
# Will read in from the input data file

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

# Data: Real and imaginary indices of refraction of sulfuric acid
# solutions at 25, 38, 50, 75, 84.5, and 95.6% H2SO4, by weight.

# Reference: K. F. Palmer and Dudley Williams, Optical constants of
# sulfuric acid; Application to the clouds of Venus?, Applied Optics,
# vol. 14, pgs. 208-219, 1975.

# Format: 227 lines of real indices (2x,f6.0,2x,f6.3,2x,6(2x,f5.3))
#    227 lines of imaginary indices (2x,f6.0,2x,f6.3,2x,6(1x,e9.2))


# Real indices of refraction for sulfuric acid solutions

#   cm-1  microns   25%    38%    50%    75%    84.5%  95.6%

#   400.  25.000    1.700  1.749  1.806  1.930  1.938  1.896
#   410.  24.390    1.696  1.744  1.808  1.939  1.954  1.880

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
        file_object.write(" rd_palmerwilliams: i,wavedat2real,wcmdat2real,rndat ")
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
        file_object.write(" rd_palmerwilliams: i,wavedat2imag,wcmdat2imag,ridat ")
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
        file_object.write(" rd_palmerwilliams: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_palmerwilliams: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_palmerwilliams: filework ")
        file_object.write(str(filework))

        nskip=int(ndat/30)

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_palmerwilliams: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_palmerwilliams.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_palmerwilliams() 
