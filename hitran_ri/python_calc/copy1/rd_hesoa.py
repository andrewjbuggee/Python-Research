#! /usr/bin/env python
"""rd_hesoa.py -  specify the refractive indices"""
def rd_hesoa():
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
    listset=['0 beta-pinene ','1 beta-caryophyllene ','2 alpha-humulene ',\
             '3 isoprene ','4 delta3-carene ','5 alpha-cedrene ']

# Number of compounds in the data file
    nsets=6

# *******
# write out all possible compounds
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_hesoa: listset ")
        for i in range(0,nsets):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_hesoa: listset ")
        for i in range(0,nsets):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 5")

# Will specify jset e.g. jset=1 for humic
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Fang soa "+listset[jset] 

# *****************
# Will read in from the input data file

# Will store all of the data in the rndat and ridat arrays
# There are ndat wavelengths
    ndat=153
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
#   fdat="/ur/massie/hitran/hitran_2024/hitran_ri/ascii/single_files/he_soa_bvoc.dat"

    fdat="he_soa_bvoc.dat"

    filework=fileasciidir+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************

#  Data: Real and imaginary indices of secondary organic aerosol (SOA)
#  from 315 to 650 nm produced by six BVOC compounds

#  beta-pinene, beta-caryophyllene, alpha-humulene
#  isoprene, delta3-carene, alpha-cedrene

#  Reference: He Q, Tomaz S, Li C, Zhu M, Meidan D , Riva M,
#  Laskin A, Brown S, George C, Wang X, Rudich Y
#  Optical properties of secondary organic aerosol produced by
#  nitrate  radical oxidation of biogenic volatile organic compounds
#  Env Sci Tech, 2021;55(5):2878

#  Email contact person: Y. Rudich (yinon.rudich@weizmann.ac.il)

#  Format: 153 real indices (2x,f8.2,2x,f10.4,6(2x,f5.3))
#          153 imaginary indices (2x,f8.2,2x,f10.4,6(1x,e10.3))

#    cm-1       microns        real indices
#                        beta-pine beta-car  alpha-hum  isopr  delta3-car alpha-ced
#  15361.93      0.6510  1.440  1.427  1.479  1.486  1.503  1.436
#  15408.08      0.6490  1.453  1.431  1.480  1.484  1.504  1.437

# *****************
# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

        nadd1=0
        nadd2=ncomp
# first data point
        i1=21
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
        i1=177
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
        file_object.write(" rd_hesoa: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_hesoa: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_hesoa: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_hesoa: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_hesoa.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_hesoa() 
