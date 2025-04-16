#! /usr/bin/env python
"""rd_myhreh2so4.py -  specify the refractive indices"""
def rd_myhreh2so4():
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
    listset=["myhreh2so4set1.dat","myhreh2so4wet2.dat","myhreh2so4set3.dat", \
              "myhreh2so4set4.dat","myhreh2so4wet5.dat","myhreh2so4set6.dat", \
              "myhreh2so4set7.dat"]

    listset0=["0   81, 81, 81, 76, 76 Wght H2SO4 at  298,  273,  267,  298,  273 K", \
              "1   76, 76, 76, 72, 72 Wght H2SO4 at  233,  213,  203,  298,  253 K", \
              "2   72, 72, 72, 72, 72 Wght H2SO4 at  245,  233,  223,  213,  203 K", \
              "3   65, 65, 65, 65, 58 Wght H2SO4 at  298,  263,  243,  223,  298 K", \
              "4   58, 58, 48, 48, 48 Wght H2SO4 at  243,  233,  298,  273,  234 K", \
              "5   48, 38, 38, 38, 38 Wght H2SO4 at  213,  298,  277,  257,  243 K", \
              "6   38, 38             Wght H2SO4 at  223,  213                   K"]

#   /ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/myhre_h2so4/myhreh2so4set1.dat
#   /ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/myhre_h2so4/myhreh2so4set2.dat
#   /ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/myhre_h2so4/myhreh2so4set3.dat
#   /ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/myhre_h2so4/myhreh2so4set4.dat
#   /ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/myhre_h2so4/myhreh2so4set5.dat
#   /ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/myhre_h2so4/myhreh2so4set6.dat
#   /ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/myhre_h2so4/myhreh2so4set7.dat

# *******
# write out all possible files to choose from
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_myhreh2so4: file listset ")
        for i in range(0,7):
            file_object.write("\n")
            a0=listset0[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_myhreh2so4: file listset ")
        for i in range(0,7):
            a0=listset0[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 6")

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
    listsub="myhre_h2so4/"
    fdat=listset[jset]

    filework=froot+listsub+fdat

    icheckfile=1
    if icheckfile ==1:
        file_object.write("\n")
        file_object.write(" filework\n ")
        file_object.write(str(filework))
#       sys.exit()

    if jset == 0:
      ncomp=5
      listset=['0   81% 298 K ','1   81% 273 K ','2   81% 267 K ','3   76% 298 K ','4   76% 273 K']

    if jset == 1:
      ncomp=5
      listset=['0   76% 233 K ','1   76% 213 K ','2   76% 203 K ','3   72% 298 K ','4   72% 253 K']

    if jset == 2:
      ncomp=5
      listset=['0   72% 245 K ','1   72% 233 K ','2   72% 223 K ','3   72% 213 K ','4   72% 203 K']

    if jset == 3:
      ncomp=5
      listset=['0   65% 298 K ','1   65% 263 K ','2   65% 243 K ','3   65% 223 K ','4   68% 298 K']

    if jset == 4:
      ncomp=5
      listset=['0   58% 243 K ','1   58% 233 K ','2   48% 298 K ','3   48% 273 K ','4   48% 234 K']

    if jset == 5:
      ncomp=5
      listset=['0   48% 213 K ','1   38% 298 K ','2   38% 277 K ','3   38% 257 K ','4   38% 243 K']

    if jset == 6:
      ncomp=2
      listset=['0   38 % 223 K ','1   38% 213 K ']

    ndat=4733
    i1=16
    i2=i1+ndat-1
    i3=4751
    i4=i3+ndat-1

# *******
# write out all possible temperatures
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_myhreh2so4: weight h2so4, temperature listset ")
        for i in range(0,ncomp):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_myhreh2so4: weight h2so4, temperature listset ")
        for i in range(0,ncomp):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset, 0,1,2,..  ")

# Will specify jset e.g. jset=2
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Myhre H2SO4 "+listset[jset] 

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
# Data: Real and imaginary indices of H2SO4/H2O droplets
#  0.81, 0.81, 0.81, 0.76, 0.76 Weight percent H2SO4
#  298,  273,  267,  298,  273 K

# Reference: Myhre, C. E. Lund, D. H. Christensen, F. M. Nicolaisen,
# and C. J. Nielsen, Spectroscopic Study of Aqueous H2SO4 at Different
# Temperatures and Compositions: Variations in Dissociation and
# Optical Properties, J. Phys. Chem., volume 107, pgs. 1979-1991, 2005.

# Email contact person: C. E. Lund Myhre (e.c.lund@iakh.uio.no.)

# Format: 4733 real indices (2x,f7.2,2x,f10.4,5(2x,f7.5))
#         4733 imaginary indices (2x,f7.2,2x,f10.4,5(1x,e11.4))

# cm-1        microns   rnw81t298, rnw81t273, rnw81t267, rnw76t298, rnw76t273
# 7498.52      1.3336  1.41749  1.41747  1.42001  1.41318  1.40669
# 7497.02      1.3339  1.41766  1.41764  1.42017  1.41319  1.40672

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
        file_object.write(" rd_myhreh2so4: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_myhreh2so4: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_myhreh2so4: filework ")
        file_object.write(str(filework))

        nskip=int(ndat/30)

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_myhreh2so4: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_myhreh2so4.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_myhreh2so4() 
