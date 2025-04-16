#! /usr/bin/env python
"""rd_dibiagiodesert.py -  specify the refractive indices"""
def rd_dibiagiodesert():
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
    listset=['0 Algeria  ','1 Arizona ','2 Atacama ',\
             '3 Australia  ','4 Bodele ','5 Ethiopia ',\
             '6 Gobi  ','7 Kuwait ','8 Libya ',\
             '9  Mali  ','10 Mauritania ','11 Morocco ',\
             '12  Namib-1  ','13 Namib-2 ','14 Niger ',\
             '15 Patagonia  ','16 Saudi Arabia ','17 Taklimakan ',\
             '18 Tunisia  ']

# Number of compounds in the data file
    nsets=19

# *******
# write out all possible compounds
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_dibiagiodesert: listset ")
        for i in range(0,nsets):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_dibiagiodesert: listset ")
        for i in range(0,nsets):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 18")

# Will specify jset e.g. jset=1 for humic
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Di Biagio desert "+listset[jset] 

# *****************
# Will read in from the input data file

# Will store all of the data in the rndat and ridat arrays
# There are ndat wavelengths
    ndat=601
    ncomp=nsets

    ncomp1=10
    ncomp2=9

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
#   fdat="/ur/massie/hitran/hitran_2024/hitran_ri/ascii/single_files/dibiagio_desert.dat"

    fdat="dibiagio_desert.dat"

    filework=fileasciidir+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************

#  Data: Real and imaginary indices of desert material
#  from 3.0 to 15.0 microns for 18 locations

#  Algeria
#  Arizona
#  Atacama
#  Australia
#  Bodele
#  Ethiopia
#  Gobi
#  Kuwait
#  Libya
#  Mali
#  Mauritania
#  Morocco
#  Namib-1
#  Namib-2
#  Niger
#  Patagonia
#  SaudiArabia
#  Taklimakan
#  Tunisia


#  Reference: Di Biagio C, Formenti P, Balkanski Y, Caponi L
#  Cazaunau, M., Pangui, E., Journet E, Nowak S, Caquineau S
#  Andreae M, Kandler K
#  Global scale variability of the mineral dust long-wave refractive
#  index: a new dataset of in situ measurements or climate modeling
#  and remotte sensing
#  Atm Chem Phys 2017;17(3):1901

#  Email contact person: C. Di Bagio (claudia.dibiagio@lisa.ipsl.fr)

#  Format: 601 real indices (2x,f8.2,2x,f10.4,10(2x,f5.3))
#          601 real indices (2x,f8.2,2x,f10.4,9(2x,f5.3))
#          601 imaginary indices (2x,f8.2,2x,f10.4,10(1x,e10.3))
#          601 imaginary indices (2x,f8.2,2x,f10.4,9(1x,e10.3))

#    cm-1       microns        real indices
#                        Alg    Ariz   Ata    Aust   Bod    Eth    Gob    Kuw    Lib    Mal
#   3333.33      3.0000  1.458  1.476  1.470  1.490  1.459  1.488  1.498  1.470  1.445  1.425
#   3311.26      3.0200  1.450  1.460  1.473  1.486  1.451  1.487  1.465  1.455  1.456  1.430

# *****************
# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

        nadd1=0
        nadd2=ncomp1
# first data points of real indices
        i1=43
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

        nadd1=ncomp1
        nadd2=ncomp1+ncomp2
# second set of data points of real indices
        i1=647
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
        nadd2=ncomp1
# first data points of imaginary indices
        i1=1251
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

        nadd1=ncomp1
        nadd2=ncomp1+ncomp2
# second set of data points of imaginary indices
        i1=1855
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
        file_object.write(" rd_dibiagiodesert: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_dibiagiodesert: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_dibiagiodesert: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_dibiagiodesert: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_dibiagiodesert.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_dibiagiodesert() 
