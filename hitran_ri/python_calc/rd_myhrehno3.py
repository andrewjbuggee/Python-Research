#! /usr/bin/env python
"""rd_myhrehno3.py -  specify the refractive indices"""
def rd_myhrehno3():
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
    listset=["myhrebin30hno3.dat","myhrebin54hno3.dat","myhrebin64hno3.dat"]

    listset0=["0  293, 273, 253, 243, 233, 223 K    30% HNO3, 70% H2O", \
              "1  293, 273, 253, 248, 243      K    54% HNO3, 46% H2O",\
              "2  293, 273, 253, 233, 238      K    64% HNO3, 36% H2O"]

#   /ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/myhre_hno3/myhrebin30hno3.dat'
#   /ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/myhre_hno3/myhrebin54hno3.dat'
#   /ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/myhre_hno3/myhrebin64hno3.dat'

# *******
# Write out all possible files to choose from
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_myhrehno3: file listset ")
        for i in range(0,3):
            file_object.write("\n")
            a0=listset0[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_myhrehno3: file listset ")
        for i in range(0,3):
            a0=listset0[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 2")

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
    listsub="myhre_hno3/"
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
      listset=['0   293 K ','1   273 K ','2   253 K ','3   243 K ','4   233 K','5   223 K']

    if jset == 1:
      ncomp=5
      listset=['0   293 K ','1   273 K ','2   253 K ','3   248 K ','4   243 K']

    if jset == 2:
      ncomp=5
      listset=['0   293 K ','1   273 K ','2   253 K ','3   233 K ','4   239 K']

    ndat=6051
    i1=15
    i2=i1+ndat-1
    i3=6068
    i4=i3+ndat-1

# *******
# write out all possible temperatures
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_myhrehno3: temperature listset ")
        for i in range(0,ncomp):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_myhrehno3: temperature listset ")
        for i in range(0,ncomp):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset, 0,1,2,..  ")

# Will specify jset e.g. jset=2
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Myhre HNO3 "+listset[jset] 

# *****************
# Will read in from the input data file

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
# Data: Real and imaginary indices of binary HNO3/H2O
# at 223, 233, 243, 253, 273 and 293 K from 450 to 6500 cm-1  30% HNO3, 70% H2O

# Reference: Myhre, C. E. Lund, H. Grothe, A. A. Gola, and C. J.
# Nielsen, Optical Constants of HNO3/H2O and H2SO4/HNO3/H2O at Low
# Temperatures in the Infrared Region, J. Phys. Chem., volume 109,
# pgs. 7166-7171, 2005.

# Email contact person: C. J. Nielsen (c.j.nielsen@kjemi.uio.no.)

# Format: 6051 real indices (2x,f7.2,2x,f10.4,6(2x,f7.5))
#         6051 imaginary indices (2x,f7.2,2x,f10.4,4(1x,e11.4))

# cm-1        microns  real293  real273  real253  real243  real233  real223
# 6500.00      1.5385  1.37170  1.38310  1.40010  1.39730  1.39540  1.40150
# 6499.00      1.5387  1.37170  1.38320  1.39910  1.39760  1.39370  1.40140

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
                wavedat2[ii,j]=1.0e4/a1
                wcmdat2[ii,j]=a1
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
#               wavedat2[ii,j]=a1
#               wcmdat2[ii,j]=1.0e4/a1
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
        file_object.write(" rd_myhrehno3: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_myhrehno3: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_myhrehno3: filework ")
        file_object.write(str(filework))

        nskip=int(ndat/30)

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_myhrehno3: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_myhrehno3.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_myhrehno3() 
