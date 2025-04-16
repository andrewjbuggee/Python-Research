#! /usr/bin/env python
"""rd_liusoaacp.py -  specify the refractive indices"""
def rd_liusoaacp():
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
    listset=['0  A1   toulene   5.0        0.0            n/a ',\
             '1  A2   toulene   5.0        2.5            14  ',\
             '2  A3   toulene   5.0        5.0            7.0 ',\
             '3  A4   toulene   5.0       10.0            3.5 ',\
             '4  B1   m-xylene  5.0        0.0            n/a ',\
             '5  B2   n-xylene  5.0        2.5            16  ',\
             '6  B3   n-xylene  5.0        5.0            8.0 ',\
             '7  B4   n-xylene  5.0       10.0            4.0 ',\
             '                HC0(ppm)   NO0(ppm)   NC0/NO0 (ppbc ppN-1) ']

# *******
# write out all possible compounds
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_liusoaacp: listset ")
        for i in range(0,9):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_liusoaacp: listset ")
        for i in range(0,9):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 7")

# Will specify jset e.g. jset=3
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Liu SOA ACP "+listset[jset] 

# *****************
# Will read in from the liusoaacp.dat ascii file

# Will store all of the data in the rndat and ridat arrays
# There are ndat wavelengths
#   ndat=571
    ncomp=8

    nlinesi=571
    wavelengthi2=np.zeros((nlinesi,ncomp))
    wcmi2=np.zeros((nlinesi,ncomp))
    ridat2=np.zeros((nlinesi,ncomp))
    wavelengthi=np.zeros((nlinesi))
    wcmi=np.zeros((nlinesi))
    ridat=np.zeros((nlinesi))

    nlinesr=93
    wavelengthr2=np.zeros((nlinesr,ncomp))
    wcmr2=np.zeros((nlinesr,ncomp))
    rndat2=np.zeros((nlinesr,ncomp))
    wavelengthr=np.zeros((nlinesr))
    wcmr=np.zeros((nlinesr))
    rndat=np.zeros((nlinesr))

# Output arrays
#   wavedat=np.zeros(ndat)
#   wcmdat=np.zeros(ndat)
#   rnval=np.zeros(ndat)
#   rival=np.zeros(ndat)

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
#   fdat="/ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/single_files/liu_soa_ACP2015.dat"

    fdat="liu_soa_ACP2015.dat"

    filework=fileasciidir+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************
# Reference: P. F. Liu, N. Abdelmalki, H.-M. Hung, Y. Wang, W. H. Brune
# and S. T. Martin
# Ultraviolet and visible complex refractive indices of secondary
# organic material produced by photooxidation of the aromatic compounds
# toluene and m-xylene
# Atmos Chem Phys, 15, 1435-1446, 2015
# www.atmos-chem-phys.net/15/1435/2015/
# doi:10.5194/acp-15-1435-2015

# Real and Imaginary Refractive Indices of SOA Aerosol

# Contact: S. T. Martin (scot_martin@harvard.edu)

# Laboratory conditions
#    1  A1   toulene   5.0        0.0            n/a
#    2  A2   toulene   5.0        2.5            14
#    3  A3   toulene   5.0        5.0            7.0
#    4  A4   toulene   5.0       10.0            3.5
#    5  B1   m-xylene  5.0        0.0            n/a
#    6  B2   n-xylene  5.0        2.5            16
#    7  B3   n-xylene  5.0        5.0            8.0
#    8  B4   n-xylene  5.0       10.0            4.0
#    ilab            HC0(ppm)   NO0(ppm)   NC0/NO0 (ppbc ppN-1)
 
#  nlines      571
#  i,wcmnk(i),wavelengthk(i),rkA1(i),rkA2(i),rkA3(i),rkA4(i),rkB1(i),rkB2(i),rkB3(i),rkB4(i)
#    1 43478.2617     0.2300     0.2281     0.1674     0.2092     0.1778     0.2281     0.1674     0.2092     0.1778
#    2 43290.0430     0.2310     0.2316     0.1638     0.2095     0.1599     0.2316     0.1638     0.2095     0.1599

# *****************
# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

# for imaginary part
        nadd1=0
        nadd2=ncomp
        i1=27
        i2=i1+nlinesi-1
        idiff=i1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[1])
            a2=float(columns[2])
            jj=2
            for j in range(nadd1,nadd2):
                wavelengthi2[ii,j]=1.0e4/a1
                wcmi2[ii,j]=a1
                jj=jj+1
                ridat2[ii,j]=float(columns[jj])

        nadd1=0
        nadd2=ncomp
        i1=601
        i2=i1+nlinesr-1
        idiff=i1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[1])
            a2=float(columns[2])
            jj=2
            for j in range(nadd1,nadd2):
                wavelengthr2[ii,j]=1.0e4/a1
                wcmr2[ii,j]=a1
                jj=jj+1
                rndat2[ii,j]=float(columns[jj])

# Close the input ascii file
    f.close()

# *****************
# Place the input data  into the rnval and rival vectors
    a1=wavelengthr2[0,jset]
    a2=wavelengthr2[1,jset]

# This is the case here
    if a1 < a2:
        for i in range(0,nlinesr):
            wavelengthr[i]=wavelengthr2[i,jset]
            wcmr[i]=wcmr2[i,jset]
            rndat[i]=rndat2[i,jset]
        for i in range(0,nlinesi):
            wavelengthi[i]=wavelengthi2[i,jset]
            wcmi[i]=wcmi2[i,jset]
            ridat[i]=ridat2[i,jset]

    if a2 < a1:
        i2=nlinesr
        for i in range(0,nlinesr):
            i2=i2-1
            wavelengthr[i]=wavelengthr2[i2,jset]
            wcmr[i]=wcmr2[i2,jset]
            rndat[i]=rndat2[i2,jset]
        i2=nlinesi
        for i in range(0,nlinesi):
            i2=i2-1
            wavelengthi[i]=wavelengthi2[i2,jset]
            wcmi[i]=wcmi2[i2,jset]
            ridat[i]=ridat2[i2,jset]

# *****************
# Since nlinesi ne nlinesr, you will use calc_samewave.py to get values at similar wavelengths
    list6=[nlinesr,wcmr,wavelengthr,rndat,nlinesi,wcmi,wavelengthi,ridat,titlegr]
#   print("list6 from rd_liusoaacp.py ",list6)

    filepickle = open("diffindxwave",'wb')
    pickle.dump(list6,filepickle)
    filepickle.close()

# *****
# Will match wavelength scales in calc_samewave.py, which reads in diffindxwave pickle
    import calc_samewave

# *****************
# Write out results
    if nopr == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_liusoaacp: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_liusoaacp: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_liusoaacp: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_liusoaacp: jset,listset[jset] ")
        a0=listset[jset]
        listn1=[jset,a0]
        file_object.write(str(listn1))

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_liusoaacp() 
