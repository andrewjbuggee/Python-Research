#! /usr/bin/env python
"""rd_querrydiesel.py -  specify the refractive indices"""
def rd_querrydiesel():
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
    ncomp=4
    listset=["0 11 lines, Querry 1987 Diesel Soot - NMSU unheated ",\
             "1 9 lines, Querry 1987 Diesel Soot - UMKC Querry ",\
             "2 11 lines, Querry 1987 Diesel Soot - NMSU heated ",\
             "3 11 lines, Querry 1987 Diesel Soot - Felske "]

# specify the lines to work work with
    ndatkk=[11,9,11,11]

# *******
# write out all possible compounds
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_querrydiesel: listset ")
        for i in range(0,ncomp):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_querrydiesel: listset ")
        for i in range(0,ncomp):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 3")

# Will specify jset e.g. jset=3
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Querry diesel "+listset[jset] 

# *****************
# Will read in from the deguine.dat ascii file

# Will store all of the data in the rndat and ridat arrays
# There are up to ndat wavelengths
    ndat=11
    wavedat2=np.zeros((ndat,ncomp))
    wcmdat2=np.zeros((ndat,ncomp))
    rndat=np.zeros((ndat,ncomp))
    ridat=np.zeros((ndat,ncomp))

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
#   fdat="/ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/single_files/querry_diesel_soot.dat"

    fdat="querry_diesel_soot.dat"

    filework=fileasciidir+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************

# Data: Real and imaginary indices of refraction of Diesel Soot

# Reference: Marvin R. Querry, Optical Constants of Minerals
# and Other Materials From The Millimeter To The Ultraviolet
# CRDEC-CR-88009, November 1987.

# Contact: Steven Massie (Steven.Massie@lasp.colorado.edu)

# Format: wavenumber(cm-1), wavelength(micron), real index, imaginary index
# 11 lines, Querry 1987 Diesel Soot - NMSU unheated
#  9 lines, Querry 1987 Diesel Soot - UMKC Querry
# 11 lines, Querry 1987 Diesel Soot - NMSU heated
# 11 lines, Querry 1987 Diesel Soot - Felske

# Querry 1987 Diesel Soot - NMSU unheated
# 11 2x,4(1x,f10.4)
# cm-1  microns   n      k
#   22222.2227     0.4500     1.5600     0.2600
#    5000.0000     2.0000     1.9100     0.3500
#    3333.3333     3.0000     2.0200     0.3400
#    2500.0000     4.0000     2.0800     0.3300

# *****************
# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

        i1=15
        i2=72
        if i >= i1 and i <= i2:

            line=line.strip()
            columns=line.split()

# ######
            if i >= 19 and i <= 29:
                ii=i-19
                jj=0
                
# wcm
                a1=float(columns[0])
# microns
                a2=float(columns[1])
# real
                a3=float(columns[2])
# imaginary
                a4=float(columns[3])

                listn1=[a1,a2,a3,a4]
                file_object.write("\n")
                file_object.write(str(listn1))

# store data
                wavedat2[ii,jj]=a2
                wcmdat2[ii,jj]=a1
                rndat[ii,jj]=a3
                ridat[ii,jj]=a4

# ######
            if i >= 34 and i <= 42:
                ii=i-34
                jj=1
                
# wcm
                a1=float(columns[0])
# microns
                a2=float(columns[1])
# real
                a3=float(columns[2])
# imaginary
                a4=float(columns[3])

                listn1=[a1,a2,a3,a4]
                file_object.write("\n")
                file_object.write(str(listn1))

# store data
                wavedat2[ii,jj]=a2
                wcmdat2[ii,jj]=a1
                rndat[ii,jj]=a3
                ridat[ii,jj]=a4

# ######
            if i >= 47 and i <= 57:
                ii=i-47
                jj=2
                
# wcm
                a1=float(columns[0])
# microns
                a2=float(columns[1])
# real
                a3=float(columns[2])
# imaginary
                a4=float(columns[3])

                listn1=[a1,a2,a3,a4]
                file_object.write("\n")
                file_object.write(str(listn1))

# store data
                wavedat2[ii,jj]=a2
                wcmdat2[ii,jj]=a1
                rndat[ii,jj]=a3
                ridat[ii,jj]=a4

# ######
            if i >= 62 and i <= 72:
                ii=i-62
                jj=3
                
# wcm
                a1=float(columns[0])
# microns
                a2=float(columns[1])
# real
                a3=float(columns[2])
# imaginary
                a4=float(columns[3])

                listn1=[a1,a2,a3,a4]
                file_object.write("\n")
                file_object.write(str(listn1))

# store data
                wavedat2[ii,jj]=a2
                wcmdat2[ii,jj]=a1
                rndat[ii,jj]=a3
                ridat[ii,jj]=a4


# Close the input ascii file
    f.close()

# *****************
    jset=int(jset)

# Output arrays
    ndat=ndatkk[jset]
    wavedat=np.zeros(ndat)
    wcmdat=np.zeros(ndat)
    rnval=np.zeros(ndat)
    rival=np.zeros(ndat)

# Place the input data into the rnval and rival vectors
# Note that there  is only  one wavelength read  in  here
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
        file_object.write(" rd_querrydiesel: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_querrydiesel: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_querrydiesel: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_querrydiesel: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_querrydiesel.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_querrydiesel() 
