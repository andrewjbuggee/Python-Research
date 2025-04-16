#! /usr/bin/env python
"""rd_niedzielah2so4.py -  specify the refractive indices"""
def rd_niedzielah2so4():
    import sys
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
    listset=["h2so4T200.niedziela","h2so4T210.niedziela","h2so4T220.niedziela", \
              "h2so4T230.niedziela","h2so4T240.niedziela","h2so4T260.niedziela", \
              "h2so4T280.niedziela","h2so4T300.niedziela"]

    listset0=["0   45, 50 Wght H2SO4 at 200 K", \
              "1   32, 39, 61 Wght H2SO4 at 210 K", \
              "2   38, 43, 50, 55, 66, 72 Wght H2SO4 at 220 K", \
              "3   75 Wght H2SO4 at 230 K", \
              "4   42, 50, 59, 69, 80 Wght H2SO4 at 240 K", \
              "5   42, 47, 55, 72, 76, 87 Wght H2SO4 at 260 K", \
              "6   50, 63, 70, 76, 85 Wght H2SO4 at 280 K", \
              "7   72, 75, 85 Wght H2SO4 at 2003 K"]

# *******
# write out all possible files to choose from
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_niedzielah2so4: file listset ")
        for i in range(0,7):
            file_object.write("\n")
            a0=listset0[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_niedzielah2so4: file listset ")
        for i in range(0,7):
            a0=listset0[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 7")

# Will specify jset e.g. jset=2
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

#   titlegr=listset0[jset] 

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
    listsub="niedziela_h2so4/"
    fdat=listset[jset]

    filework=froot+listsub+fdat

    icheckfile=1
    if icheckfile ==1:
        file_object.write("\n")
        file_object.write(" filework\n ")
        file_object.write(str(filework))
#       sys.exit()

    if jset == 0:
      ncomp=2
      listset=['0   45% 200 K ','1   50% 200 K ']

    if jset == 1:
      ncomp=3
      listset=['0   32% 210 K ','1   39% 210 K ','2   61% 210 K ']

    if jset == 2:
      ncomp=6
      listset=['0   38% 220 K ','1   43% 220 K ','2   50% 220 K ','3   55% 220 K ','4   66% 220 K','5  72% 220 K']

    if jset == 3:
      ncomp=1
      listset=['0   75% 230 K ']

    if jset == 4:
      ncomp=5
      listset=['0   42% 240 K ','1   50% 240 K ','2   59% 240 K ','3   69% 240 K ','4   80% 240 K']

    if jset == 5:
      ncomp=6
      listset=['0   42% 260 K ','1   47% 260 K ','2  55% 260 K ','3   72% 260 K ','4   76% 260 K','5  87% 260 K']

    if jset == 5:
      ncomp=5
      listset=['0   50% 280 K ','1   63% 280 K ','2  70% 280 K ','3   76% 280 K ','4   85% 280 K']

    if jset == 6:
      ncomp=3
      listse3=['0   72% 300 K ','1   75% 300 K ','2  85% 300 K']

    ndat=2010
    i1=19
    i2=i1+ndat-1
    i3=2032
    i4=i3+ndat-1

# *******
# write out all possible h2so4-temperature combinations
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_niedzielah2so4: weight h2so4, temperature listset ")
        for i in range(0,ncomp):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_niedzielah2so4: weight h2so4, temperature listset ")
        for i in range(0,ncomp):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset, 0,1,2,..  ")

# Will specify jset e.g. jset=2
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Niedziela H2SO4 "+listset[jset] 

# *****************
# Will read in from the input data file

# Will store all of the data in the rndat and ridat arrays
# There are ndat wavelengths
    wavedat2=np.zeros((ndat,ncomp))
    wcmdat2=np.zeros((ndat,ncomp))
    rndat=np.zeros((ndat,ncomp))
    ridat=np.zeros((ndat,ncomp))

# Output arrays
    wavedat=np.zeros(ndat)
    wcmdat=np.zeros(ndat)
    rnval=np.zeros(ndat)
    rival=np.zeros(ndat)

# Open input ascii file for reading
    f = open(filework,'r')

# *****************

#  Data: Real and imaginary indices of liquid H2SO4/H2O
#  at 200 K from 825 to 4700 cm-1.

#  Reference: Niedziela, R. F., M. L. Norman, C. L. deForest,
#  R. E. Miller, and D. R. Worsnop, A Temperature and
#  Composition-Dependent Study of H2SO4 Aerosol Optical
#  Constants Using Fourier Transform and Tunable Diode
#  Laser Infrared Spectroscopy, J. Phys. Chem. A,
#  vol. 103, pgs. 8030-8040, 1999.

#  Email contact person:  R. E. Miller (remiller@unc.edu)

#  Format: 2010 real indices (2x,f7.2,2x,f10.4,6(1x,f5.3))
#          2010 imaginary indices (2x,f7.2,2x,f10.4,6(1x,e10.3))

#   cm-1       microns  real
#                      45%   50%
#   825.59     12.1126 1.646 1.699
#   827.51     12.0844 1.648 1.701
#   829.44     12.0563 1.650 1.703

# *****************
# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

        nadd1=0
        nadd2=ncomp
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
        idiff=i3
        if i >= i3 and i <= i4:
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
        file_object.write(" rd_niedzielah2so4: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_niedzielah2so4: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_niedzielah2so4: filework ")
        file_object.write(str(filework))

        nskip=int(ndat/30)

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_niedzielah2so4: i,wavedat,wcmdat,rndex,ridex ")
        for i in range(0,ndat,nskip):
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
#   print("list5 from rd_niedzielah2so4.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_niedzielah2so4() 
