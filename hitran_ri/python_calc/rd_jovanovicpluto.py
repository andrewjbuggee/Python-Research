#! /usr/bin/env python
"""rd_jovanovicpluto.py -  specify the refractive indices"""
def rd_jovanovicpluto():
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
    ndat=367
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
#   fdat="/ur/massie/hitran/hitran_2024/hitran_ri/ascii/single_files/jovanovic_pluto_aerosol.dat"

    fdat="single_files/jovanovic_pluto_aerosol.dat"
    titlegr="Jovanovic_pluto "

    filework=froot+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************
#
# Data: Real and imaginary indices of Pluto tholins
# from 0.27 to 2.1 microns
#
# Reference:  Jovanovic L, Gautier T, Broch L, Protopapa S,
# Bertrand T, Rannou P, Fayolle M, Quirico E, Johann L, Naciri A,
# Carrasco, N
#  ptical constants of Pluto aerosol analogues from UV to near-IR
# Icarus, 2021;362:114398
#
# Email contact person: L. Jovanovic (lora.jovanpvic@latmos.ipsl.fr)
#
# Format: 367 real indices (2x,f8.2,2x,f10.4,1(2x,f5.3))
#         367 imaginary indices (2x,f8.2,2x,f10.4,1(1x,e10.3))
#
#   cm-1       microns   real indices
#                      Pluto
# 37037.00      0.2700  1.712
# 36364.00      0.2750  1.720

# *****************
# ndat=367
    i1=1
    i2=17
# real indices
    i3=18
    i4=384
# 3 headers
    i5=385
    i6=387
# imaginary indices
    i7=388
    i8=754

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
        file_object.write(" rd_jovanovicpluto: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_jovanovicpluto: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_jovanovicpluto: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_jovanovicpluto: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_jovanovicpluto.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_jovanovicpluto() 
