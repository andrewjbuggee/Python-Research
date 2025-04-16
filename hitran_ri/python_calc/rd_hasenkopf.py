#! /usr/bin/env python
"""rd_hasenkopf.py -  specify the refractive indices"""
def rd_hasenkopf():
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
# Vuitton just has imaginary index so ignore, put ncomp=5
    ncomp=5
    listset=["0 Hasenkopf   Titan ",\
             "1 Hasenkopf   early Earth ",\
             "2 Khare (1984)  10% CH4 in N2 ",\
             "3 Ramirez (2002)  2% CH4 in N2 ",\
             "4 Tran (2003)  1.8% CH4 in N2 ",\
             "5 Vuitton (2009) 1.8% CH4 in N2 "]

# *******
# write out all possible compounds
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_hasenkopf: listset ")
        for i in range(0,ncomp):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_hasenkopf: listset ")
        for i in range(0,ncomp):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 8")

# Will specify jset e.g. jset=3
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Hasenkopf file "+listset[jset] 

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
#   fdat="/ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/single_files/zhasenkopf_organic_haze.dat""

    fdat="hasenkopf_organic_haze.dat"

    filework=fileasciidir+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************

#  Data: Real and imaginary indices at 532 nm of organic haze
#  for Titan and early Earth, and other determinations

#  Reference: C. Hasenkopf, M. Beaver, M. Trainer, H. Dewit, M. Freedman
#  O. Toon, C. McKay, M. Tolbert
#  Optical properties of Titan and early earth haze laboratory analogs in
#  the mid-visible, Icarus, vol 207, 903-913, 2010.

# Contact: Christa Hasenkopf (Christa.Hasenkopf@colorado.edu)

# Note: See Table 1 of the paper which also reports values from others.

# case index    case
#  0            Hasenkopf   Titan
#  1            Hasenkopf   early Earth
#  2            Khare (1984)  10% CH4 in N2
#  3            Ramirez (2002)  2% CH4 in N2
#  4            Tran (2003)  1.8% CH4 in N2
#  5            Vuitton (2009) 1.8% CH4 in N2

# Format: 1 line 2x,2(1x,f8.2),6(1x,f10.4)

#      cm-1    microns     real
#                           Titan      EEarth      Khare     Ramirez    Tran     Vuitton
#   18796.992     0.532     1.3500     1.8100     1.7100     1.5690     1.5800   -99.0000
#
#      cm-1    microns     imaginary
#                           Titan      EEarth      Khare     Ramirez    Tran     Vuitton
#   18796.992     0.532     0.0230     0.0550     0.0320     0.0025     0.0540     0.0220

# *****************
# Hardwire
    ii=0
    a1=18796.992
    a2=0.532

    for jj in range(0,ncomp):
        wavedat2[ii,jj]=a2
        wcmdat2[ii,jj]=a1

    rndat[ii,0]=1.350
    rndat[ii,1]=1.81
    rndat[ii,2]=1.71
    rndat[ii,3]=1.5690
    rndat[ii,4]=1.58

    ridat[ii,0]=0.0230
    ridat[ii,1]=0.0550
    ridat[ii,2]=0.0320
    ridat[ii,3]=0.0540
    ridat[ii,4]=0.0220

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
        file_object.write(" rd_hasenkopf: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_hasenkopf: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_hasenkopf: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_hasenkopf: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_hasenkopf.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_hasenkopf() 
