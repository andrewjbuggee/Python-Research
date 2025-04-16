#! /usr/bin/env python
"""calc_wave.py -  calculate the wavelength and wavenumber scales"""
def calc_wave():
    import os
    import math
    import numpy as np
    import pickle

# *****************
# Open the output f.out ascii file
    fileout="f.out"
    file_object = open(fileout,'a')

    nopr=1

# *****************
# Obtain the data from init_calc.py
#   list1=[iwave,w1,w2,dw,den1,rad1,sig1,den2,rad2,sig2,r1,r2,iset]
#   print("list1 from init_calc.py ",list1)

    filepickle = open("sizedistparam",'rb')
    list1=pickle.load(filepickle)
    filepickle.close()

    iwave=list1[0]
    w1=list1[1]
    w2=list1[2]
    dw=list1[3]
    iset=list1[12]

# *****************
# Write out input
    iwrinput=0
    if iwrinput == 1:
        listn1=[iwave,w1,w2,dw]

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_wave: input iwave,w1,w2,dw\n ")
        file_object.write(str(listn1))

# *****************
# Specify iwave. iwave=1 for wavenumbers, iwave=2 for wavelength
# 2
# Specify w1,w2,dw values (e.g. range of wavenumber,wavenumber spacing in spectra
# or range of wavelength in microns, wavelength spacing in spectra in microns)
# 3.0 12.0 0.1

    nwave=int((w2-w1)/dw)

    wcm=np.zeros(nwave)
    wavelength=np.zeros(nwave)

# Special write 
    iwrsp=0
    if iwrsp == 1:
        listn2=[nwave]

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_wave: nwave\n ")
        file_object.write(str(listn2))
    
# Calculate the size distribution
    nwavem1=nwave-1

# Input is for wcm in wavenumbers
    if iwave == 1:
        for i in range(0,nwave):
            a1=float(w1+(i*dw))
            wcm[i]=a1
            wavelength[i]=1.0e4/a1
    else: 
        pass

# Input is for wavelength in microns
    if iwave == 2:
        for i in range(0,nwave):
            a1=float(w1+(i*dw))
            wavelength[i]=a1
            wcm[i]=1.0e4/a1
    else: 
        pass

# *****************
# Write out results
    if nopr == 1:

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_wave: nwave\n ")
        file_object.write(str(nwave))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_wave: i,wavelength,wcm ")
        for i in range(0,nwave):
            file_object.write("\n")
            a0="% 4d" %i
            a1="% 10.4f" %wavelength[i]
            a2="% 10.4f" %wcm[i]
            listn1=[a0,a1,a2]
            file_object.write(str(listn1))

# *****************
    ichecktype=0
    if ichecktype == 1:
        print(wcm.dtype)
        print(wavelength.dtype)

# *****************
# One way to pass data out of a function is to   pickle   the data
    list2=[iwave,nwave,wcm,wavelength,iset]
#   print("list2 from calc_wave.py ",list2)
    filepickle = open("wavescale",'wb')
    pickle.dump(list2,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

# *****************
calc_wave() 
