#! /usr/bin/env python
"""rd_toonpsc.py -  specify the refractive indices"""
def rd_toonpsc():
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
    listset=['0 Beta NAT ','1 NAD ','2 alpha NAT ','3 NAM ','4 Water ice ', \
             '5 A NAT ','6 A NAD ','7 A NAM ']
    ilinesj=[25, 139, 276, 392, 531, 708, 832, 949 ]
    nlinesj=[110, 133, 112, 135, 173, 120, 113, 124 ]

# *******
# write out all possible compounds
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_toonpsc: listset ")
        for i in range(0,8):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_toonpsc: listset ")
        for i in range(0,8):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 7")

# Will specify jset e.g. jset=3
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Toon PSC "+listset[jset]

# *****************
# Will read in from the toonpsc.dat ascii file

# Will store all of the data in the rndat and ridat arrays
# There are ndat max wavelengths
    ndat=200
    ndat200=200
    ncomp=8
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
#   fdat="/ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/single_files/toon_psc.dat"

    fdat="toon_psc.dat"

    filework=fileasciidir+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************

# Data: Real and imaginary indices of refraction of H2O-ice, amorphous
# nitric acid solutions, and nitric acid hydrates.

# Reference: O. B. Toon, M. A. Tolbert, B. G. Koehler, A. M. Middlebrook,
# and J. Jordan, The infrared optical constants of H2O-ice, amorphous
# acid solutions, and nitric acid hydrates, J. Geophys. Res., accepted
# for publication, 1994.

# Format: wavenumber(cm-1), real index,imaginary index
#  110 lines, BETA NAT film  at 196 K
#  133 lines, NAD film at 184 K
#  112 lines, ALPHA NAT film at 181 k
#  135 lines, NAM film at 179 K
#  173 lines, water ice film at 163 K
#  120 lines, A NAT film at 153 K
#  113 lines, A NAD film at 153 K
#  124 lines, A NAM film at 153 K
# 2x,f6.1,2x,f5.3,2x,e8.2


# BETA NAT film at 196 K
#  110    2x,f6.1,2x,f5.3,2x,e8.2
#  cm-1   n      k
#  482.0  1.950  2.22E-01
#  494.0  1.832  2.54E-01

# *****************
# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

        for mset in range(0,ncomp):
            i1=ilinesj[mset]
            i2=i1+nlinesj[mset]-1
            idiff=i1
            if i >= i1 and i <= i2:
                line=line.strip()
                columns=line.split()
                ii=i-idiff
                a1=float(columns[0])
                jj=0
                wavedat2[ii,mset]=1.0e4/a1
                wcmdat2[ii,mset]=a1
                jj=jj+1
                rndat[ii,mset]=float(columns[jj])
                jj=jj+1
                ridat[ii,mset]=float(columns[jj])

# Close the input ascii file
    f.close()

# *****************
# Place the input data into the rnval and rival vectors
    a1=wavedat2[0,jset]
    a2=wavedat2[1,jset]
    ndatj=nlinesj[jset]
    ndat=ndatj
    if a1 < a2:
        for i in range(0,ndatj):
            wavedat[i]=wavedat2[i,jset]
            wcmdat[i]=wcmdat2[i,jset]
            rnval[i]=rndat[i,jset]
            rival[i]=ridat[i,jset]
    if a2 < a1:
        i2=ndatj
        for i in range(0,ndatj):
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
        file_object.write(" rd_toonpsc: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_toonpsc: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_toonpsc: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_toonpsc: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_toonpsc.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_toonpsc() 
