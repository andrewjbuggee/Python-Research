#! /usr/bin/env python
"""rd_poschcatio3.py -  specify the refractive indices"""
def rd_poschcatio3():
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
# Will read in from the poschcatio3.dat ascii file

# Will store all of the data in the rndat and ridat arrays
# There are ndat wavelengths
    ndat=2000
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
#   fdat="/ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/exoplanets/posch_catio3.dat"

    fdat="exoplanets/posch_catio3.dat"
    titlegr="Posch CaTiO3 "

    filework=froot+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************
#Reference: Posch, Th., F. Kerschbaum, D. Fabian, H. Mutschke,
#J. Dorschner, A. Tamanai, Th. Henning, Infrared Properties of Solid
#Titanium Oxides: Exploring Potential Primarry Dust Condensates
#Astrophysical Jounral Supplement Series, v149, p437-445, 2003

#Real and Imaginary Refractive Indices of CaTiO3 (Perovskite)

#Contact: Thomas Posch (thomas.posch@univie.ac.at)

#Format: 2000 lines 2x,2(1x,f8.2),2(1x,f10.4)

#     cm-1    microns     real    imaginary
#   5000.00     2.00     2.2762     0.0002
#   4980.08     2.01     2.2760     0.0002
#   4960.24     2.02     2.2757     0.0002

# *****************
# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

        nadd1=0
        nadd2=ncomp
        i1=13
        i2=i1+ndat-1
        idiff=i1
        if i >= i1 and i <= i2:
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
                jj=jj+1
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
        file_object.write(" rd_poschcatio3: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_poschcatio3: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_poschcatio3: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_poschcatio3: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_poschcatio3.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_poschcatio3() 
