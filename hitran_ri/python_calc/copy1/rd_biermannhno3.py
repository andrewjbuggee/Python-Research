#! /usr/bin/env python
"""rd_biermannhno3.py -  specify the refractive indices"""
def rd_biermannhno3():
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
    listset=["hno3T213.biermann","hno3T223.biermann","hno3T233.biermann",\
             "hno3T253.biermann"," hno3T263.biermann","hno3T273.biermann",\
             "hno3T293.biermann"]

    nsets=7
    listset0=["0 213 K  Wght HNO3   40   45",\
              "1 223 K  Wght HNO3   40   45   50",\
              "2 233 K  Wght HNO3   30   40   45   50",\
              "3 253 K  Wght HNO3   30   40   45   50",\
              "4 263 K  Wght HNO3    0   20",\
              "5 273 K  Wght HNO3    0   10   20   30   40   45   50",\
              "6 293 K  Wght HNO3    0   10   20   30   40   45   50"]

# *******
# write out all possible files to choose from
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_biermannhno3: file listset ")
        for i in range(0,nsets):
            file_object.write("\n")
            a0=listset0[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_biermannhno3: file listset ")
        for i in range(0,nsets):
            a0=listset0[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 6")

# Will specify jset e.g. jset=2
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

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
    listsub="biermann_hno3/"
    fdat=listset[jset]

    filework=froot+listsub+fdat

    icheckfile=1
    if icheckfile ==1:
        file_object.write("\n")
        file_object.write(" filework\n ")
        file_object.write(str(filework))
#       sys.exit()

    if jset == 0:
      ncomp=2
      listset=['0   40% 213 K ','1   45% 213 K ']

    if jset == 1:
      ncomp=3
      listse3=['0   40% 223 K ','1   45% 223 K ','2   50% 223 K ']

    if jset == 2:
      ncomp=4
      listset=['0   30% 233 K ','1   40% 233 K ','2   45% 233 K ','3   50% 233 K ']

    if jset == 3:
      ncomp=4
      listset=['0   30% 253 K ','1   40% 253 K ','2   45% 253 K ','3   50% 253 K ']

    if jset == 4:
      ncomp=2
      listset=['0    0% 263 K ','1   20% 263 K ']

    if jset == 5:
      ncomp=7
      listset=['0    0% 273 K ','1   10% 273 K ','2  20% 273 K ','3   30% 273 K ',\
               '4   40% 273 K ','5   45% 273 K ','6  50% 273 K ']

    if jset == 6:
      ncomp=7
      listset=['0    0% 293 K ','1   10% 293 K ','2  20% 293 K ','3   30% 293 K ',\
               '4   40% 293 K ','5   45% 293 K ','6  50% 293 K ']


# real values from 0 to 16382 cm-1
    ndat1=8192

# imaginary values from 431 to 5028 cm-1
    ndat2=4768

    i1=19
    i2=i1+ndat1-1

    i3=8214
    i4=i3+ndat2-1

# Will read in the original data, then will need to interpolate to imaginary wavelength scale

# *******
# write out all possible h2so4-temperature combinations
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_biermannhno3: weight h2so4, temperature listset ")
        for i in range(0,ncomp):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_biermannhno3: weight h2so4, temperature listset ")
        for i in range(0,ncomp):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset, 0,1,2,..  ")

# Will specify jset e.g. jset=2
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Biermann HNO3 "+listset[jset] 

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

# Data: Real indices of liquid HNO3/H2O from 0 to 16382 cm-1
# and imaginary indices from 432 to 5028 cm-1 at 213 K
# as a function of the HNO3 concentration by weight.

# Reference: Biermann, U. M., B. P. Luo, and Th. Peter,
# Absorption Spectra and Optical Constants of Binary and
# Ternary Solutions of H2SO4, HNO3, and H2O in the Mid
# Infrared at Atmospheric Temperatures, J. Phys. Chem. A,
# vol. 104, pgs 783-793, 2000.

# Email contact person: B. P. Luo (luo@mpch-mainz.mog.de)

# Format: 8192 real indices (2x,f8.2,2x,f10.4,12(1x,f5.3))
#         4768 imaginary indices (2x,f8.2,2x,f10.4,12(1x,e10.3))

#  cm-1       microns  real
#                      40%   45%
#     0.00      0.0000 1.800 1.734
#     2.00   5000.0000 1.800 1.734
#     4.00   2500.0000 1.800 1.734

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
            if a1 < 1.0e-4:
                a1=1.0e-4
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
        file_object.write(" rd_biermannhno3: i,wavedat2real,wcmdat2real,rndat ")
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
        file_object.write(" rd_biermannhno3: i,wavedat2imag,wcmdat2imag,ridat ")
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
    rninterp=np.zeros(ndat)

# real indices
    ndatm1=ndat1-1

    j1=200
    j2=ndatm1
# loop over imaginary  indices cm-1 grid (wcm increses with increase in i)
    for i in range(0,ndat):
        wcm=wcmdat2imag[i,jset]
# loop over real  indices cm-1 grid (wcm increases with increse in j)
        for j in range(j1,j2):
           wcma=wcmdat2real[j,jset]
           j3=j+1
           wcmb=wcmdat2real[j3,jset]
           if wcm >= wcma and wcm < wcmb:
               rninterp[i]=rndat[j,jset]
               j1=j-1
               break

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
            rnval[i]=rninterp[i]
            rival[i]=ridat[i,jset]
    if a2 < a1:
        i2=ndat
        for i in range(0,ndat):
            i2=i2-1
            wavedat[i]=wavedat2imag[i2,jset]
            wcmdat[i]=wcmdat2imag[i2,jset]
            rnval[i]=rninterp[i2]
            rival[i]=ridat[i2,jset]
    
# *****************
# Write out results
    if nopr == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_biermannhno3: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_biermannhno3: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_biermannhno3: filework ")
        file_object.write(str(filework))

        nskip=int(ndat/30)

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_biermannhno3: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_biermannhno3.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_biermannhno3() 
