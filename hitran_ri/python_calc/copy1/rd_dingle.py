#! /usr/bin/env python
"""rd_dingle.py -  specify the refractive indices"""
def rd_dingle():
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
    ncomp=25
    listset=["0 Lgf     H2O2     48 4      30 1         I ", \
             "1 Lgf     H2O2     37 3      10 1         II  ",\
             "2 Lgf     HONO     86 3      0.1 0.1      III ",\
             "3 Lgf     HONO     76 1      0.09 0.01    IVi  ",\
             "4 a-P     H2O2     110 7     22 1         I ",\
             "5 a-P     H2O2     129 3     12 2         II ",\
             "6 a-P     H2O2     48 2      4 1          III",\
             "7 a-P     H2O2     40 1      4 1          IV ",\
             "8 1-MN    H2O2     80 5      21 3         I ",\
             "9 1-MN    H2O2     61 75      1           II ",\
             "10 1-MN    H2O2     54 5      4 1          III ",\
             "11 1-MN    H2O2     22 13      0.1         IV ",\
             "12 1-MN    HONO     300 20    0.4 0.008    V ",\
             "13  1-MN    HONO     240 5     0.3 0.004    VI ",\
             "14 Phe     H2O2     210 2     60 1         I ",\
             "15 Phe     H2O2     160 4     27 1         II ",\
             "16 Phe     H2O2     31 2      10 1         III  ",\
             "17 Phe     HONO     120 6     0.2 0.003    IV ",\
             "18 Phe     HONO     110 5     0.1 0.002    V ",\
             "19 Tol     H2O2     240 3     50 1         II ",\
             "20 Tol     H2O2     240 3     50 1         II ",\
             "21 Tol     H2O2     72 1      N/A          III ",\
             "22 Tol     H2O2     43 3      18 2         IV ",\
             "23 Tol     HONO     163 20    0.2 0.02     V ",\
             "24 Tol     HONO     110 7     0.1 0.04     VI "]

# *******
# write out all possible compounds
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_dingle: listset ")
        for i in range(0,ncomp):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_dingle: listset ")
        for i in range(0,ncomp):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 24")

# Will specify jset e.g. jset=3
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Dingle SOA "+listset[jset] 

# *****************
# Will read in from the deguine.dat ascii file

# Will store all of the data in the rndat and ridat arrays
# There are ndat wavelengths
    ndat=1
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
#   fdat="/ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/single_files/dingle_soa.dat"

    fdat="dingle_soa.dat"

    filework=fileasciidir+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************
# Reference: Justin H. Dingle, Stephen Zimmerman, Alexander L. Frie,
# Justin Min, Heejung Jung and Roya Bahreini
# Complex refractive index, single scattering albedo, and mass
# absorption coefficient of secondary organic aerosols generated from
# oxidation of biogenic and anthropogenic precursors
# Aer Sci Tech, v53, 449-463, 2019
# https://doi.org/10.1080/02786826.2019.1571680

# Real and Imaginary Refractive Indices of SOA Aerosol

# Contact: Roya Bahreini ( Roya.Bahreini@ucr.edu )

# Laboratory conditions (ntoe that you removed the plus or minus no-ascii character)
#    1,  Lgf     H2O2     48 4      30 1         I
#    2,  Lgf     H2O2     37 3      10 1         II
#    3,  Lgf     HONO     86 3      0.1 0.1      III
#    4,  Lgf     HONO     76 1      0.09 0.01    IVi
#    5,  a-P     H2O2     110 7     22 1         I
#    6,  a-P     H2O2     129 3     12 2         II
#    7,  a-P     H2O2     48 2      4 1          III
#    8,  a-P     H2O2     40 1      4 1          IV
#    9,  1-MN    H2O2     80 5      21 3         I
#   10,  1-MN    H2O2     61 75      1           II
#   11,  1-MN    H2O2     54 5      4 1          III
#   12,  1-MN    H2O2     22 13      0.1         IV
#   13,  1-MN    HONO     300 20    0.4 0.008    V
#   14,  1-MN    HONO     240 5     0.3 0.004    VI
#   15,  Phe     H2O2     210 2     60 1         I
#   16,  Phe     H2O2     160 4     27 1         II
#   17,  Phe     H2O2     31 2      10 1         III
#   18,  Phe     HONO     120 6     0.2 0.003    IV
#   19,  Phe     HONO     110 5     0.1 0.002    V
#   20,  Tol     H2O2     270 5     55 4         I
#   21,  Tol     H2O2     240 3     50 1         II
#   22,  Tol     H2O2     72 1      N/A          III
#   23,  Tol     H2O2     43 3      18 2         IV
#   24,  Tol     HONO     163 20    0.2 0.02     V
#   25,  Tol     HONO     110 7     0.1 0.04     VI
#   ilab   Hydrocarboan  Hydroxyl Radical Source
#   HCo (ppbv) and [HC/NOx]o and Plot Reference
#   HCo is the initial hydrocarbon concentration
#   [HC/NOx]o is the initial hydrocarbon to NOx ratio
#   LgF=Longifolene a-P=alpha Pinene 1MN=1-Methylnaphthalene
#   Phe=Phenol, Tol=Toluene

# nlines       25
# j,wcm375(j),wavelength375(j),rn375(j),ri375(j),wcm632(j),wavelength632(j),rn632(j)
#  1     26666.6660     0.3750     1.4500     0.0000     15822.7842     0.6320     1.4900
#  2     26666.6660     0.3750     1.4800     0.0000     15822.7842     0.6320     1.4800
#  3     26666.6660     0.3750     1.5000     0.0020     15822.7842     0.6320     1.4700

# *****************
# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

        i1=48
        i2=72
        jj=-1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()

# note that column[0] is number (1,2,3  etc)
# wcm
            a1=float(columns[1])
# microns
            a2=float(columns[2])
# real
            a3=float(columns[3])
# imaginary
            a4=float(columns[4])

            ii=0
            jj=i-i1
            wavedat2[ii,jj]=a2
            wcmdat2[ii,jj]=a1
            rndat[ii,jj]=a3
            ridat[ii,jj]=a4

# Close the input ascii file
    f.close()

# *****************
# Place the input data into the rnval and rival vectors
# Note that there  is only  one wavelength read  in  here
    i=0
    wavedat[i]=wavedat2[i,jset]
    wcmdat[i]=wcmdat2[i,jset]
    rnval[i]=rndat[i,jset]
    rival[i]=ridat[i,jset]

# *****************
# Write out results
    if nopr == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_dingle: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_dingle: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_dingle: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_dingle: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_dingle.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_dingle() 
