#! /usr/bin/env python
"""rd_warren.py -  specify the refractive indices"""
def rd_warren():
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
# Will read in from the warren_ice.dat ascii file

# Will store all of the data in the rndat and ridat arrays
# There are ndat wavelengths
    ndat=498-13+1

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
#   fdat="/ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/single_files/warren_ice.dat"

    fdat="warren_ice.dat"
    titlegr="warren_ice"
    filework=fileasciidir+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************
 
# Data: Real and Imaginary indices of Ice Ih at 266 K
 
# Reference: Warren SG, Brandt RE. Optical constants of ice from the
# ultraviolet to the microwave: A revised compilation.
# J Geophys Res 2008;113:D14220,doi:10.1029/2007JD009744.

# Email contact person: Stephen G. Warren (sgw@atmos.washington.edu)

# Format: 486 real and imaginary indices (2(1x,e11.4),1(1x,f6.4),1(1x,e10.3))

# cm-1        microns    real    imaginary
# 2.2573e+05  4.4300e-02 0.8228  1.640e-01
# 2.2173e+05  4.5100e-02 0.8250  1.730e-01

# Read in from the ascii file
    i=0
    for line in f:
        i=i+1

#       if i <= 12:
#          line=line.strip()
#          listw1=[line]
#          file_object.write("\n")
#          file_object.write(str(listw1))

        i1=13
        i2=498
        ndat=498-13+1
        idiff=i1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            a2=float(columns[1])
            wavedat[ii]=a2
            wcmdat[ii]=a1
            rnval[ii]=float(columns[2])
            rival[ii]=float(columns[3])


# Close the input ascii file
    f.close()

# *****************
# Write out results
    if nopr == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_warren: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_warren: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_warren: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_warren: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_warren.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_warren() 
