#! /usr/bin/env python
"""rd_ramireztitan.py -  specify the refractive indices"""
def rd_ramireztitan():
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
# Will read in from the ramireztitan.dat ascii file

# Will store all of the data in the rndat and ridat arrays
# There are ndat wavelengths
    ndat=161
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
#   list1=[fileasciidir,filesasciiraoot]
#   print("list0 from init_calc.py ",list1)

    filepickle = open("subdirectories",'rb')
    list1=pickle.load(filepickle)
    filepickle.close()

    fileasciidir=list1[0]
    froot=list1[1]

# *****************
# Open the input work.dat ascii file that has the wavelength scale and size distribution info
#   froot="/ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/"
#   fdat="/ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/exoplanets/ramirez_titan_aerosol.dat"

    fdat="ramirez_titan_aerosol.dat"
    titlegr="Ramirez Titan Aerosol "

    filework=froot+'exoplanets/'+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************
#Reference: Khare, B. N., C. Sagan, E. T. Arakawa, F. Suits,
#T. A. Callcott, and M. W. Williams
#Optical Constants of Organic Tholins Produced in a Simulated Titanian
#Atmosphere: from Soft X-Ray to Microwave Frequencies
#Icarus, v60, p127-137, 1984

#Real and Imaginary Refractive Indices of Titan Tholins

#Contact: Steven Massie (lasp.colorado.edu)

#Format: 90 lines 2x,2(1x,e10.4),2(1x,f10.4)

#      cm-1    microns       real    imaginary
#  4.8309e+05 2.0700e-02     0.9200     0.0490
#  3.2051e+05 3.1200e-02     0.8500     0.1400
#  2.4096e+05 4.1500e-02     0.8020     0.3100

# *****************
# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

        nadd1=0
        nadd2=ncomp
        i1=14
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
        file_object.write(" rd_ramireztitan: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_ramireztitan: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_ramireztitan: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_ramireztitan: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_ramireztitan.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_ramireztitan() 
