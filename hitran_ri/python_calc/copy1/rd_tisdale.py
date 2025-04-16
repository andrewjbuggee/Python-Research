#! /usr/bin/env python
"""rd_tisdale.py -  specify the refractive indices"""
def rd_tisdale():
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
    listset=['0 45.2 % ','1 50 % ','2 55.1 % ','3 61.6 % ', \
             '4 65.3 % ','5 70.2 % ','6 75 % ','7 80 %']

# *******
# write out all possible compounds
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_tisdale: listset ")
        for i in range(0,8):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_tisdale: listset ")
        for i in range(0,8):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 7")

# Will specify jset e.g. jset=3
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Tisdale H2SO4 "+listset[jset] 

# *****************
# Will read in from the input data file

# Will store all of the data in the rndat and ridat arrays
# There are ndat wavelengths
    ndat=1685
    ncomp=8
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
#   fdat="/ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/single_files/tisdale_h2so4.dat"

    fdat="tisdale_h2so4.dat"

    filework=fileasciidir+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************
# Data: Real and imaginary indices of liquid H2SO4/H2O
# at 215 K from 499 to 6996 cm-1 as a function of the
# H2SO4 concentration by weight.

# Reference: Tisdale, R. T., D. L. Glandorf, M. A. Tolbert,
# and O. B. Toon, Infrared optical constants of low-temperature
# H2SO4 solutions representative of stratospheric sulfate
# aerosols, J. Geophys. Res., vol. 103, pgs. 25353-25370,
# 1998.

# Email contact person: M. Tolbert (tolbert@spot.colorado.edu)

# Format: 1685 real indices (2x,f7.2,2x,f10.4,8(2x,f5.3))
#         1685 imaginary indices (2x,f7.2,2x,f10.4,8(1x,e10.3))

#  cm-1       microns                        real indices
#                      45.2%  50%    55.1%  61.6%  65.3%  70.2%  75%    80%
#  499.50     20.0200  1.989  2.013  2.032  2.067  2.098  2.100  2.103  2.089
#  503.30     19.8689  1.887  1.905  1.920  1.958  1.992  2.000  2.013  2.005


# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

        nadd1=0
        nadd2=ncomp
        i1=19
        i2=i1+ndat-1
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
        i1=1707
        i2=i1+ndat-1
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
        file_object.write(" rd_tisdale: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_tisdale: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_tisdale: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_tisdale: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_tisdale.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_tisdale() 
