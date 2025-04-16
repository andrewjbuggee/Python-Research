#! /usr/bin/env python
"""rd_zarzana.py -  specify the refractive indices"""
def rd_zarzana():
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
    ncomp=9
    listset=["0 Zarzana   glyoxal-glycine ",\
             "1 Zarzana   glyoxal-methylamine ",\
             "2 Zarzana   methylglyoxal-glycine ",\
             "3 Zarzana   glyoxal-methylamine ",\
             "4 Dinar (2008) pollution HULIS ",\
             "5 Dinar (2008) smoke HULIS ",\
             "6 Dinar (2008) K-puszta HULIS ",\
             "7 Hoffer (2006) HULIS-day ",\
             "8 Hoffer (2006) HULIS-night "]

# *******
# write out all possible compounds
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_zarzana: listset ")
        for i in range(0,ncomp):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_zarzana: listset ")
        for i in range(0,ncomp):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 8")

# Will specify jset e.g. jset=3
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Zarzana SOA "+listset[jset] 

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
#   fdat="/ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/single_files/zarzana_soa.dat"

    fdat="zarzana_soa.dat"

    filework=fileasciidir+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************
# Data: Real and imaginary indices at 532 nm of proxies of secondary
# organic aerosol, and other determinations of HULIS indices

# Reference: Kyle J. Zarzana, David O. DeHaan, Miriam A. Freedman,
# Christa A. Hasenkopf, and Margaret A. Tolbert.
# Optical Propertiers of the Products of alpha-Dicarbonyl and Amine
# Reactions in Simulated Cloud Droplets, Env Sci Tech,46,4845-4851,2012

# Contact: Kyle Zarzana (kyle.zarzana@colorado.edu)

# Note: See Table 1 of the paper which also reports values from others.
# HULIS are HUmic LIke Substances
# case index    case
#  0            Zarzana   glyoxal-glycine
#  1            Zarzana   glyoxal-methylamine
#  2            Zarzana   methylglyoxal-glycine
#  3            Zarzana   glyoxal-methylamine
#  4            Dinar (2008) pollution HULIS
#  5            Dinar (2008) smoke HULIS
#  6            Dinar (2008) K-puszta HULIS
#  7            Hoffer (2006) HULIS-day
#  8            Hoffer (2006) HULIS-night

# Format: 1 line 2x,2(1x,f9.3),2(1x,f10.4)
#
# 0            Zarzana   glyoxal-glycine
#      cm-1    microns     real imaginary
#   18796.992     0.532     1.6400     0.0440
#
# 1            Zarzana   glyoxal-methylamine
#      cm-1    microns     real imaginary
#   18796.992     0.532     1.6500     0.0350

# *****************
# Read in from the ascii file
    i=0
    k=0
    jj=-1
    for line in f:
        i=i+1

        i1=26
        i2=61
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()

# read  in three headers
            k=k+1
# this line will have the data
            if k == 4:

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

# reset k
                k=0

# store data
                ii=0
                jj=jj+1
                wavedat2[ii,jj]=a2
                wcmdat2[ii,jj]=a1
                rndat[ii,jj]=a3
                ridat[ii,jj]=a4

# Close the input ascii file
    f.close()

# *****************
    jset=int(jset)

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
        file_object.write(" rd_zarzana: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_zarzana: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_zarzana: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_zarzana: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_zarzana.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_zarzana() 
