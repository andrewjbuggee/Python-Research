#! /usr/bin/env python
"""init_calc.py - specify input to the HITRAN-RI calculation"""
def init_calc():
    import os
    import numpy as np
    import pickle

# *****************
# Open the output f.out ascii file
    fileout="f.out"
    file_object = open(fileout,'a')

    nopr=1

# *****************
# Sepcify the wavelength scale and the size distribution by hand

# Specify iwave. iwave=1 for wavenumbers, iwave=2 for wavelength
#   iwave=2
# Specify w1,w2,dw values (e.g. range of wavenumber,wavenumber spacing in spectra
# or range of wavelength in microns, wavelength spacing in spectra in microns)
#   w1=3.0
#   w2=12.0
#   dw=0.5

#   listn1=[iwave,w1,w2,dw]

# Specify the parameters of the log-normal size distribution, both modes
# The first mode (number/cm3, mean radii (microns), sigma width)  (den1,rad1,sig1)
#   den1=56.3
#   rad1=0.0179
#   sig1=2.35
# The second mode (number/cm3, mean radii (microns), sigma width) (den2,rad2,sig2)
#   den2=0.309
#   rad2=0.411
#   sig2=1.28
# Specify the radii range r1,r2 of the size distribution in microns
#   r1=0.005
#   r2=3.0

# *****************
# Open the input work.dat ascii file that has the wavelength scale and size distribution info
    filework="work.dat"
    f = open(filework,'r')

    nopr=1

# *******
# note, coma erases the "cartridge return"
# note, line=line.strip() gets rid of the \n character
    i=0
    for line in f:
        i=i+1
#       if i == 1 or i == 2:
#          line=line.strip()
#          listw1=[line]
#          file_object.write("\n")
#          file_object.write(str(listw1))
        if i == 3:
            line=line.strip()
            columns=line.split()
            iset=int(columns[0])
#       if i == 4 or i == 5:
#          line=line.strip()
#          listw1=[line]
#          file_object.write("\n")
#          file_object.write(str(listw1))
        if i == 6:
            line=line.strip()
            columns=line.split()
            iwave=int(columns[0])
#       if i == 7 or i == 8:
#          line=line.strip()
#          listw1=[line]
#          file_object.write("\n")
#          file_object.write(str(listw1))
        if i == 9:
            line=line.strip()
            columns=line.split()
            w1=float(columns[0])
            w2=float(columns[1])
            dw=float(columns[2])
#       if i >= 10 and i <= 12:
#          line=line.strip()
#          listw1=[line]
#          file_object.write("\n")
#          file_object.write(str(listw1))
        if i == 13:
            line=line.strip()
            columns=line.split()
            den1=float(columns[0])
            rad1=float(columns[1])
            sig1=float(columns[2])
#       if i == 14:
#          line=line.strip()
#          listw1=[line]
#          file_object.write("\n")
#          file_object.write(str(listw1))
        if i == 15:
            line=line.strip()
            columns=line.split()
            den2=float(columns[0])
            rad2=float(columns[1])
            sig2=float(columns[2])
#       if i == 16:
#          line=line.strip()
#          listw1=[line]
#          file_object.write("\n")
#          file_object.write(str(listw1))
        if i == 17:
            line=line.strip()
            columns=line.split()
            r1=float(columns[0])
            r2=float(columns[1])

    f.close()

# *****************
# Open the input work.dat ascii file that has the wavelength scale and size distribution info
    filework="directory.dat"
    f = open(filework,'r')

    nopr=1

# *******
# note, coma erases the "cartridge return"
# note, line=line.strip() gets rid of the \n character
    i=0
    for line in f:
        i=i+1
#       if i == 1 or i == 2:
#          line=line.strip()
#          listw1=[line]
#          file_object.write("\n")
#          file_object.write(str(listw1))
        if i == 3:
            line=line.strip()
            columns=line.split()
            fileasciidir=columns[0]
        if i == 5:
            line=line.strip()
            columns=line.split()
            fileasciiroot=columns[0]

    f.close()

# Write out the fileasciidir string to the subdirectories pickle
    list0=[fileasciidir,fileasciiroot]
    print("list0 from init_calc.py",list0)

    filepickle = open("subdirectories",'wb')
    pickle.dump(list0,filepickle)
    filepickle.close()

# *****************
    listn0=[iset]
    listn1=[iwave,w1,w2,dw]
    listn2=[den1,rad1,sig1]
    listn3=[den2,rad2,sig2]
    listn4=[r1,r2]

# Write out results
    if nopr == 1:

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" init_calc: fileasciidir\n ")
        file_object.write(str(list0))

        file_object.write("\n")
        file_object.write(" init_calc: iset\n ")
        file_object.write(str(listn0))

        file_object.write("\n")
        file_object.write(" init_calc: iwave,w1,w2,dw\n ")
        file_object.write(str(listn1))

        file_object.write("\n")
        file_object.write(" init_calc: iwave,w1,w2,dw\n ")
        file_object.write(str(listn1))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" init_calc: den1,rad1,sig1\n ")
        file_object.write(str(listn2))

        file_object.write("\n")
        file_object.write(" init_calc: den2,rad2,sig2\n ")
        file_object.write(str(listn3))

        file_object.write("\n")
        file_object.write(" init_calc: r1,r2\n ")
        file_object.write(str(listn4))

# *****************
    ichecktype=0
    if ichecktype == 1:
        print(iwave.dtype)
        print(w1.dtype)
        print(rad1.dtype)
        print(r1.dtype)

# *****************
# One way to pass data out of a function is to   pickle   the data
    list1=[iwave,w1,w2,dw,den1,rad1,sig1,den2,rad2,sig2,r1,r2,iset]
#   print("list1 from init_calc.py ",list1)

    filepickle = open("sizedistparam",'wb')
    pickle.dump(list1,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

init_calc() 
