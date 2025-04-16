#! /usr/bin/env python
"""rd_remsberg.py -  specify the refractive indices"""
def rd_remsberg():
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
    listset=["remsberg_h2so4_hno3.dat"]

    nsets=1
    listset0=["0   h2so4 or hno3"]

# *******
# write out all possible files to choose from
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_remsberg: file listset ")
        for i in range(0,nsets):
            file_object.write("\n")
            a0=listset0[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=0
#   if iwrsp == 2:
#       print(" rd_remsberg: file listset ")
#       for i in range(0,nsets):
#           a0=listset0[i]
#           listn1=[a0]
#           print(str(listn1))
#       print(" enter jset from 0 to 7")

# Will specify jset e.g. jset=2
#   jsetstr=input(" jset: ")
#   jset=int(jsetstr)

# Just work with one input data file 
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
      ncomp=3
      listset=['0   55% H2SO4 300 K ','1   90% H2SO4 300 K ','2  68% HNO3 300 K']

    ndat1=184
    ndat2=211
    ndat3=44

# used below
    ndats=[ndat1,ndat2,ndat3]

    i1=23
    i2=i1+ndat1-1

    i3=210
    i4=i3+ndat2-1

    i5=424
    i6=i5+ndat3-1

# *******
# write out all possible h2so4-temperature combinations
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_remsberg: weight h2so4, temperature listset ")
        for i in range(0,ncomp):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_remsberg: weight h2so4, temperature listset ")
        for i in range(0,ncomp):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset, 0,1,2,..  ")

# Will specify jset e.g. jset=2
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Remsberg "+listset[jset] 

# *****************
# Will read in from the input data file

# Will store all of the data in the rndat and ridat arrays
# There are up to ndat wavelengths
    ndat=500
    wavedat2=np.zeros((ndat,ncomp))
    wcmdat2=np.zeros((ndat,ncomp))
    rndat=np.zeros((ndat,ncomp))
    ridat=np.zeros((ndat,ncomp))

# Open input ascii file for reading
    f = open(filework,'r')

# *****************

# Data: Real and imaginary indices of refraction of sulfuric acid
# solutions at 75 and 90% H2SO4, by weight, plus the standard
# deviations of the measurements.
#       Real and imaginary indices of refraction of nitric acid
# solutions at 68% HNO3, by weight, plus the standard
# deviations of the measurements.

# Reference: E. E. Remsberg, D. Lavery, and B. Crawford, Optical
# constants for sulfuric and nitric acids, J. Chem. and Engin. Data,
# vol.19, pgs. 263-255, 1974.

# Format: wavenumber(cm-1), wavelength(microns), imaginary index,
# imaginary index standard deviation, real index, real index standard
# deviation, correlation coefficient.
# 184 lines at 75% H2SO4
# 211 lines at 90% H2SO4
#  44 lines at 68% HNO3
# 2x,f6.1,2x,f6.3,5(2x,f6.4)

#  OPTICAL CONSTANTS FOR 75% AQUEOUS H2SO4
#  cm-1    microns    k     sk     n        sn      rho
# 1571.0   6.365   .1435   .0051  1.3957   .0072  -.6634
# 1567.8   6.378   .1423   .0051  1.3927   .0072  -.6672
# 1551.6   6.445   .1403   .0051  1.3870   .0072  -.6749

# *****************
# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

# 75% h2so4 (note j=0)
        idiff=i1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            j=0
            wavedat2[ii,j]=1.0e4/a1
            wcmdat2[ii,j]=a1
            ridat[ii,j]=float(columns[2])
            rndat[ii,j]=float(columns[4])

# 90% h2so4 (note j=1)
        idiff=i3
        if i >= i3 and i <= i4:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            j=1
            wavedat2[ii,j]=1.0e4/a1
            wcmdat2[ii,j]=a1
            ridat[ii,j]=float(columns[2])
            rndat[ii,j]=float(columns[4])

# 68% hno3 (note j=2)
        idiff=i5
        if i >= i5 and i <= i6:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            j=2
            wavedat2[ii,j]=1.0e4/a1
            wcmdat2[ii,j]=a1
            ridat[ii,j]=float(columns[2])
            rndat[ii,j]=float(columns[4])


# Close the input ascii file
    f.close()

# *****************
# Output arrays
    ndat=ndats[jset]

    wavedat=np.zeros(ndat)
    wcmdat=np.zeros(ndat)
    rnval=np.zeros(ndat)
    rival=np.zeros(ndat)

# Place the input data into the rnval and rival vectors
    a1=wavedat2[0,jset]
    a2=wavedat2[1,jset]

# yes, this makes sense
    if a1 < a2:
        for i in range(0,ndat):
            wavedat[i]=wavedat2[i,jset]
            wcmdat[i]=wcmdat2[i,jset]
            rnval[i]=rndat[i,jset]
            rival[i]=ridat[i,jset]

# This does make sense for this data set
#   if a2 < a1:
#       i2=ndat
#       for i in range(0,ndat):
#           i2=i2-1
#           wavedat[i]=wavedat2[i2,jset]
#           wcmdat[i]=wcmdat2[i2,jset]
#           rnval[i]=rndat[i2,jset]
#           rival[i]=ridat[i2,jset]
    
# *****************
# Write out results
    if nopr == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_remsberg: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_remsberg: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_remsberg: filework ")
        file_object.write(str(filework))

        nskip=int(ndat/30)

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_remsberg: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_remsberg.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_remsberg() 
