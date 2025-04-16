#! /usr/bin/env python
"""rd_shettle.py -  specify the refractive indices"""
def rd_shettle():
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
    listset=['0 Water ','1 Ice ','2 NaCl ','3 Sea Salt ','4 Watersol ','5 Ammonium Sulfatae', \
             '6 Carbonaceous ','7 Volcanic Dust ','8 H2SO4 215K ','9 H2SO4 300K ', \
             '10 Meteoric ','11 Quartz O ','12 Quartz E ','13 Hematite O ','14 Hematite E', \
             '15 Sand O ','16 Sand E ','17 Dustlike']

# *******
# write out all possible compounds
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_shettle: listset ")
        for i in range(0,18):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_shettle: listset ")
        for i in range(0,18):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 17")

# Will specify jset e.g. jset=3
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

# *****************
# Will read in from the afcrl1987_shettle.dat ascii file

# Will store all of the data in the rndat and ridat arrays
# There are ndat wavelengths
    ndat=291-202+1
    ncomp=18
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
#   fdat="/ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/single_files/afcrl1987_shettle.dat"

    fdat="afcrl1987_shettle.dat"
    fdat2="afcrl1987_shettle "
    filework=fileasciidir+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************
# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

#       if i <= 201:
#          line=line.strip()
#          listw1=[line]
#          file_object.write("\n")
#          file_object.write(str(listw1))

#    0.200   1.396  1.10e-7    1.394  1.50e-8    1.790  3.1e-9    1.510  1.0e-4
        nadd1=0
        nadd2=4
        i1=202
        i2=291
        idiff=i1
        ndat1=i2-i1+1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            jj=0
            for j in range(nadd1,nadd2):
                wavedat2[ii,j]=a1
                wcmdat2[ii,j]=1.0e4/a1
                jj=jj+1
                rndat[ii,j]=float(columns[jj])
                jj=jj+1
                ridat[ii,j]=float(columns[jj])

        nadd1=4
        nadd2=7
        i1=303
        i2=363
        idiff=i1
        ndat2=i2-i1+1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            jj=0
            for j in range(nadd1,nadd2):
                wavedat2[ii,j]=a1
                wcmdat2[ii,j]=1.0e4/a1
                jj=jj+1
                rndat[ii,j]=float(columns[jj])
                jj=jj+1
                ridat[ii,j]=float(columns[jj])

        nadd1=7
        nadd2=11
        i1=373
        i2=434
        idiff=i1
        ndat3=i2-i1+1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            jj=0
            for j in range(nadd1,nadd2):
                wavedat2[ii,j]=a1
                wcmdat2[ii,j]=1.0e4/a1
                jj=jj+1
                rndat[ii,j]=float(columns[jj])
                jj=jj+1
                ridat[ii,j]=float(columns[jj])

        nadd1=11
        nadd2=15
        i1=445
        i2=512
        idiff=i1
        ndat4=i2-i1+1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            jj=0
            for j in range(nadd1,nadd2):
                wavedat2[ii,j]=a1
                wcmdat2[ii,j]=1.0e4/a1
                jj=jj+1
                rndat[ii,j]=float(columns[jj])
                jj=jj+1
                ridat[ii,j]=float(columns[jj])

        nadd1=15
        nadd2=18
        i1=523
        i2=590
        idiff=i1
        ndat5=i2-i1+1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            jj=0
            for j in range(nadd1,nadd2):
                wavedat2[ii,j]=a1
                wcmdat2[ii,j]=1.0e4/a1
                jj=jj+1
                rndat[ii,j]=float(columns[jj])
                jj=jj+1
                ridat[ii,j]=float(columns[jj])

# Close the input ascii file
    f.close()

# *****************
# Place the input data  into the rnval and rival vectors
    mdat=0
    for i in range(0,ndat):
        wavedat[i]=wavedat2[i,jset]
        wcmdat[i]=wcmdat2[i,jset]
        rnval[i]=rndat[i,jset]
        rival[i]=ridat[i,jset]
        if rnval[i] > 0.0:
            mdat=mdat+1

# redefine ndat
    ndat=mdat
    
# *****************
# Write out results
    if nopr == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_shettle: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_shettle: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_shettle: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_shettle: ndat ")
        file_object.write(str(ndat))
        file_object.write(" rd_shettle: i,wavedat,wcmdat,rndex,ridex ")
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
    titlegr=fdat2+listset[jset]
    list5=[ndat,wavedat,wcmdat,rnval,rival,titlegr]
#   print("list5 from rd_shettle.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_shettle() 
