#! /usr/bin/env python
"""rd_wagnersuper.py -  specify the refractive indices"""
def rd_wagnersuper():
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
    listset=['0   238 K ','1   252 K ','2   258 K ','3   269 K ']

# *******
# write out all possible compounds
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_wagnersuper: listset ")
        for i in range(0,4):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_wagnersuper: listset ")
        for i in range(0,4):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 3")

# Will specify jset e.g. jset=3
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="wagner_supercooled "+listset[jset]

# *****************
# Will read in from the wagnersuper.dat ascii file

# Will store all of the data in the rndat and ridat arrays
# There are ndat wavelengths
    ndat=442
    ncomp=4
    wavedat2=np.zeros((ndat,ncomp))
    wcmdat2=np.zeros((ndat,ncomp))
    rndat=np.zeros((ndat,ncomp))
    ridat=np.zeros((ndat,ncomp))

# Output arrays
    wavedat=np.zeros(ndat)
    wcmdat=np.zeros(ndat)
    rnval=np.zeros(ndat)
    rival=np.zeros(ndat)

# *****************
# Obtain the data from init_calc.py
#   list1=[fileasciidir]
#   print("list0 from init_calc.py ",list1)

    filepickle = open("subdirectories",'rb')
    list1=pickle.load(filepickle)
    filepickle.close()

    fileasciidir=list1[0]

# *****************
# Open the input work.dat ascii file that has the wavelength scale and size distribution info
#   fileasciidir="/ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/single_files/"

    fdat="wagner_supercooled.dat"

    filework=fileasciidir+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************
# Data: Real and imaginary indices of supercooled water
# at 238, 252, 258, and 269 K from 1101 to 4503 cm-1

# Reference: Wagner,R., S. Benz, O. Muhler, H. Saathoff, M. Schnaiter,
# and U. Schurath, Mid-Infrared Extinction Spectra and Optical
# Constants of Supercooled Water Droplets, J. Phys. Chem., volume 109,
# pgs. 7099-7112, 2005.

# Email contact person: Robert Wagner (Robert.Wagner@imk.fzk.de)

# Format: 442 real indices (2x,f7.2,2x,f10.4,4(2x,f7.5))
#         442 imaginary indices (2x,f7.2,2x,f10.4,4(1x,e11.4))

# cm-1        microns  real238, real252, real258, real269,
# 4503.04      2.2207  1.28210  1.28723  1.28864  1.29003
# 4495.33      2.2245  1.28177  1.28694  1.28833  1.28973

# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

        nadd1=0
        nadd2=4
        i1=15
        i2=456
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
        nadd2=4
        i1=459
        i2=900
        idiff=i1
        if i >= i1 and i <= i2:
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
# Place the input data into the rnval and rival vectors
    for i in range(0,ndat):
        wavedat[i]=wavedat2[i,jset]
        wcmdat[i]=wcmdat2[i,jset]
        rnval[i]=rndat[i,jset]
        rival[i]=ridat[i,jset]
    
# *****************
# Write out results
    if nopr == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_wagnersuper: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_wagnersuper: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_wagnersuper: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_wagnersuper: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_wagnersuper.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_wagnersuper() 
