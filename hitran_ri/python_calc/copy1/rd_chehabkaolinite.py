#! /usr/bin/env python
"""rd_chehabkaolinite.py -  specify the refractive indices"""
def rd_chehabkaolinite():
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
# Will read in from the jovanovic_pluto_aerosol.dat ascii file

# Will store all of the data in the rndat and ridat arrays
# There are ndat wavelengths
    ndat=8465
    ncomp=1
    wavedat2=np.zeros((ndat,ncomp))
    wcmdat2=np.zeros((ndat,ncomp))
    rndat=np.zeros((ndat,ncomp))
    ridat=np.zeros((ndat,ncomp))

    jset=0

# Output arrays
    wavedat=np.zeros(ndat)
    wcmdat=np.zeros(ndat)
    rnval=np.zeros(ndat)
    rival=np.zeros(ndat)

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
# Open the input work.dat ascii file that has the wavelength scale and size distribution info
    fdat="single_files/chehab_kaolinite.dat"
    titlegr="Chehab_kaolinite "

    filework=froot+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************
#
#  Data: Real and imaginary indices of kaolinite
#  from 216 to 40000 cm-1

#  Reference: Chehab M, Herbin H, Deguine A, Gosselin S, Bizet V
#  Petitprez D
#  First complex refractive indices retrieval from FIR to UV
#  Application to kaolinite particles
#  Aerosol Science and Technology, 2024;58(5):498
#
#  Email contact person: M. Chehab (maria.chehab@univ-lille.fr)
#
#  Format: 8465 real indices (2x,f8.2,2x,f10.4,1(2x,f5.3))
#          8465 imaginary indices (2x,f8.2,2x,f10.4,1(1x,e10.3))
#
#    cm-1       microns        real indices
#                        kaolinite
#    216.00     46.2963  3.712
#    217.00     46.0830  1.802

# *****************
# ndat=8465
    i1=1
    i2=17
# real indices
    i3=18
    i4=8482
# 3 headers
    i5=8483
    i6=8485
# imaginary indices
    i7=8486
    i8=16950

# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

#  15 headers and 2 more lines 
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()

# read in real indices
        nadd1=0
        nadd2=ncomp
        idiff=i3
        if i >= i3 and i <= i4:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            a2=float(columns[1])
            jj=1
            for j in range(nadd1,nadd2):
                wavedat2[ii,j]=a2
                wcmdat2[ii,j]=a1
                jj=jj+1
                rndat[ii,j]=float(columns[jj])

# read in 3 headers
        if i >= i5 and i <= i6:
            line=line.strip()
            columns=line.split()


        nadd1=0
        nadd2=ncomp
        idiff=i7
        if i >= i7 and i <= i8:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            a2=float(columns[1])
            jj=1
            for j in range(nadd1,nadd2):
#               wavedat2[ii,j]=a2
#               wcmdat2[ii,j]=a1
                ridat[ii,j]=float(columns[jj])

# Close the input ascii file
    f.close()

# *****************
# Place the input data into the rnval and rival vectors
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
        file_object.write(" rd_chehabkaolinite: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_chehabkaolinite: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_chehabkaolinite: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_chehabkaolinite: i,wavedat,wcmdat,rndex,ridex ")
        for i in range(0,ndat):
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
#   print("list5 from rd_chehabkaolinite.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_chehabkaolinite() 
