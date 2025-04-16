#! /usr/bin/env python
"""rd_staggcarbon.py -  specify the refractive indices"""
def rd_staggcarbon():
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
    listset=['0 Pyro 25C ','1 Pyro 200C ','2 Pyro 400C ','3 Pyro 600C ', \
             '4 Amorphous 25C ','5 Amorphous 200C ','6 Amorphous 400C ','7 Amorphous 600K ', \
             '8 Propane 25C ','9 Propane 300C ','10 Propane 600C ', \
             '11 Soot 25C ','12 Soot 300C ','13 Soot 600C ']

# *******
# write out all possible compounds
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_staggcarbon: listset ")
        for i in range(0,14):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_staggcarbon: listset ")
        for i in range(0,14):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 13")

# Will specify jset e.g. jset=3
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

# *****************
# Will read in from the afcrl1987_staggcarbon.dat ascii file

# Will store all of the data in the rndat and ridat arrays
# There are ndat wavelengths
    ndat=10
    ncomp=14
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
#   fdat="/ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/single_files/stagg_carbon.dat"

    fdat="stagg_carbon.dat"
    fdat2="Stagg_Carbon "
    filework=fileasciidir+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************

#Data: Real and imaginary indices of refraction of pyrolytic graphite,
#amorphous carbon, propane and flame soot 400 to 700 nm

#Reference: B. J. Stagg and T. T. Charalampopoulis, Refraction Indices
#of Pyrolytic Graphite, Amorphous Carbon, and Flame Soot in the
#Temperature Range 25 to 600C, Combustion and Flame, 94:381-396, 1993

#Contact: Steven Massie (Steven.Massie@lasp.colorado.edu)

#Format: 10 lines 2x,2(1x,f8.2),4(1x,f10.4)
#        10 lines 2x,2(1x,f8.2),4(1x,f10.4)

#pyro
#     cm-1    microns                     real
#                         25C        200C       400C       600C
#  25000.00     0.40     2.3770     2.4040     2.4160     2.4230
#  23094.69     0.43     2.5290     2.5440     2.5620     2.5550
#  21413.28     0.47     2.6370     2.6560     2.6750     2.6730

# *****************
# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

        nadd1=0
        nadd2=4
        i1=17
        i2=i1+ndat-1
        i3=29
        i4=i3+ndat-1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-i1
            a1=float(columns[0])
            jj=1
            for j in range(nadd1,nadd2):
                wavedat2[ii,j]=1.0e4/a1
                wcmdat2[ii,j]=a1
                jj=jj+1
                rndat[ii,j]=float(columns[jj])

        if i >= i3 and i <= i4:
            line=line.strip()
            columns=line.split()
            ii=i-i3
            a1=float(columns[0])
            jj=1
            for j in range(nadd1,nadd2):
#               wavedat2[ii,j]=1.0e4/a1
#               wcmdat2[ii,j]=a1
                jj=jj+1
                ridat[ii,j]=float(columns[jj])

        nadd1=4
        nadd2=8
        i1=43
        i2=i1+ndat-1
        i3=55
        i4=i3+ndat-1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-i1
            a1=float(columns[0])
            jj=1
            for j in range(nadd1,nadd2):
                wavedat2[ii,j]=1.0e4/a1
                wcmdat2[ii,j]=a1
                jj=jj+1
                rndat[ii,j]=float(columns[jj])

        if i >= i3 and i <= i4:
            line=line.strip()
            columns=line.split()
            ii=i-i3
            a1=float(columns[0])
            jj=1
            for j in range(nadd1,nadd2):
#               wavedat2[ii,j]=1.0e4/a1
#               wcmdat2[ii,j]=a1
                jj=jj+1
                ridat[ii,j]=float(columns[jj])

        nadd1=8
        nadd2=11
        i1=69
        i2=i1+ndat-1
        i3=81
        i4=i3+ndat-1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-i1
            a1=float(columns[0])
            jj=1
            for j in range(nadd1,nadd2):
                wavedat2[ii,j]=1.0e4/a1
                wcmdat2[ii,j]=a1
                jj=jj+1
                rndat[ii,j]=float(columns[jj])

        if i >= i3 and i <= i4:
            line=line.strip()
            columns=line.split()
            ii=i-i3
            a1=float(columns[0])
            jj=1
            for j in range(nadd1,nadd2):
#               wavedat2[ii,j]=1.0e4/a1
#               wcmdat2[ii,j]=a1
                jj=jj+1
                ridat[ii,j]=float(columns[jj])

        nadd1=11
        nadd2=14
        i1=95
        i2=i1+ndat-1
        i3=107
        i4=i3+ndat-1
        idiff=i1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-i1
            a1=float(columns[0])
            jj=1
            for j in range(nadd1,nadd2):
                wavedat2[ii,j]=1.0e4/a1
                wcmdat2[ii,j]=a1
                jj=jj+1
                rndat[ii,j]=float(columns[jj])

        if i >= i3 and i <= i4:
            line=line.strip()
            columns=line.split()
            ii=i-i3
            a1=float(columns[0])
            jj=1
            for j in range(nadd1,nadd2):
#               wavedat2[ii,j]=1.0e4/a1
#               wcmdat2[ii,j]=a1
                jj=jj+1
                ridat[ii,j]=float(columns[jj])

# Close the input ascii file
    f.close()

# *****************
# Place the input data  into the rnval and rival vectors
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
        file_object.write(" rd_staggcarbon: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_staggcarbon: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_staggcarbon: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_staggcarbon: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_staggcarbon.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_staggcarbon() 
