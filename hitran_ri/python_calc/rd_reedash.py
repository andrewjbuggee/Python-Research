#! /usr/bin/env python
"""rd_reedash.py -  specify the refractive indices"""
def rd_reedash():
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
# Phooey, need to replace greek beta by beta
    listset=['0 askja  ','1 aso ','2 eyjafjallajokull Reed 1 ',\
             '3 eyjafjallajokull Reed 2 ','4 grimsvotn ','5 nisyros ',\
             '6 spurr ','7 tongariro ']

# Number of compounds in the data file
    nsets=8

# *******
# write out all possible compounds
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_reedash: listset ")
        for i in range(0,nsets):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_reedash: listset ")
        for i in range(0,nsets):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 7")

# Will specify jset e.g. jset=1 for humic
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Reed volcanic ash "+listset[jset] 

# *****************
# Will read in from the input data file

# Will store all of the data in the rndat and ridat arrays
# There are ndat wavelengths
    ndat=1933
    ncomp=nsets
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
#   fdat="/ur/massie/hitran/hitran_2024/hitran_ri/ascii/single_files/reed_volcanic_ash.dat"

    fdat="reed_volcanic_ash.dat"

    filework=fileasciidir+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************

#  Data: Real and imaginary indices of volcanic ash
#  from 528  to 25000 cm-1 for eight samples

#  askja-ash_Reed
#  aso-ash_Reed
#  eyjafjallajokull-ash_Reed_1
#  eyjafjallajokull-ash_Reed_2
#  grimsvotn-ash_Reed
#  nisyros-ash_Reed
#  spurr-ash_Reed
#  tongariro-ash_Reed

#  Reference: Reed B, Peters D, McPheat R, Grainger R.
#  The complex refractive index of volcanic ash aerosol retrieved
#  from spectral mass extinction
#  J Geophys Res 2018;123:13390

#  Email contact person: R. Grainger (grainger@atm.ox.ac.uk)

#  Format: 1933 real indices (2x,f8.2,2x,f10.4,8(2x,f5.3))
#          1933 imaginary indices (2x,f8.2,2x,f10.4,8(1x,e10.3))

#     cm-1       microns        real indices
#                        Asjka  Aso    Eygaf1 Eygaf2 Grim   Nisyos Spurr  Tong
#    528.00     18.9394  1.890  1.865  2.234  2.497  2.515  1.357  2.496  1.801
#    529.00     18.9036  1.867  1.783  2.219  2.387  2.418  1.366  2.497  1.810

# *****************
# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

        nadd1=0
        nadd2=ncomp
# first data point
        i1=26
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
# first imaginary indexdata point
        i1=1962
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
        file_object.write(" rd_reedash: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_reedash: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_reedash: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_reedash: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_reedash.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_reedash() 
