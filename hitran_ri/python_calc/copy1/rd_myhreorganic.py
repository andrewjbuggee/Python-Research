#! /usr/bin/env python
"""rd_myhreorganic.py -  specify the refractive indices"""
def rd_myhreorganic():
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
    nwavemax=10000
    wcm=np.zeros(nwavemax)
    wavelength=np.zeros(nwavemax)

    ndat=1001
    ncomp=21
    ndatr=1001
    ndati=851
    npairs=10

    nlinesr=ndatr
    nlinesi=ndati

# integer
#   ireal=np.zeros(npairs,dtype=int32)
#   iimag=np.zreros(npairs,dtype=int32)

# real
#   numlines=np.zeros(npairs)
#   wcmdatr=np.zeros((ndatr,ncomp))
#   wavedatr=np.zeros((ndatr,ncomp))
#   wcmdati=np.zeros((ndati,ncomp))
#   wavedati=np.zeros((ndati,ncomp))
#   rnval=np.zeros((ndatr,ncomp))
#   rival=np.zeros((ndatr,ncomp))

# real
    wcmr2=np.zeros(ndatr)
    wavelengthr2=np.zeros(ndatr)
    rndat2=np.zeros(ndatr)

    wcmr=np.zeros(ndatr)
    wavelengthr=np.zeros(ndatr)
    rndat=np.zeros(ndatr)

# imaginary
    wcmi2=np.zeros(ndati)
    wavelengthi2=np.zeros(ndati)
    ridat2=np.zeros(ndati)

    wcmi=np.zeros(ndati)
    wavelengthi=np.zeros(ndati)
    ridat=np.zeros(ndati)

    numlines=[851, 851, 1001,1001,1001,851, 851, 1001,1001,1001,851, \
    1001,1001,851, 851,851, 1001,851, 1001,851,851 ]

# file indices for real indices (0,..)
    ireal=[2,3,4, 7,8,9,   11,12,  16,  18 ]

# file indices for imag indices
    iimag=[5,5,5, 10,10,10, 13,13,  17,  19 ]

# ***
# Obtain the listing of the compounds
# Will ask the user to choose which compound to work with

# The different compositions
    listset=['0   Imaginary indices of ammoniumsulfate    ',\
    '1   Imaginary indices of benzoic acid       ',\
    '2   Real indices of 25 wt% Glutaric acid    ',\
    '3   Real indices of 50 wt% Glutaric acid    ',\
    '4   Real indices of 5 wt% Glutaric acid     ',\
    '5   Imaginary indices of glutaric acid K    ',\
    '6   Imaginary indices of hydroxymalonic acid' ,\
    '7   Real indices of 10 wt% Malonic acid     ',\
    '8   Real indices of 25 wt% Malonic acid     ',\
    '9   Real indices of 60 wt% Malonic acid    ',\
    '10  Imaginary indices of malonic acid      ',\
    '11  Real indices of 10 wt% Oxalic acid     ',\
    '12  Real indices of 5 wt% Oxalic acid      ',\
    '13  Imaginary indices of oxalic acid       ',\
    '14  Imaginary indices of phthalic acid     ',\
    '15  Imaginary indices of pinonic acid      ',\
    '16  Real indices of 5 wt% Pyruvic acid     ',\
    '17  Imaginary indices of pyruvic acid      ',\
    '18  Real indices of 8 wt% Succinic acid    ',\
    '19  Imaginary indices of succinic acid     ',\
    '20  Imaginary indices of water at 293K     ']

# The different compositions
    listset2=['0  Ammoniumsulfate    ',\
    '1   Benzoic acid       ',\
    '2   25 wt% Glutaric acid    ',\
    '3   50 wt% Glutaric acid    ',\
    '4   5 wt% Glutaric acid     ',\
    '5   Glutaric acid K    ',\
    '6   Hydroxymalonic acid' ,\
    '7   10 wt% Malonic acid     ',\
    '8   25 wt% Malonic acid     ',\
    '9   60 wt% Malonic acid    ',\
    '10  Malonic acid      ',\
    '11  10 wt% Oxalic acid     ',\
    '12  5 wt% Oxalic acid      ',\
    '13  Oxalic acid       ',\
    '14  Phthalic acid     ',\
    '15  Pinonic acid      ',\
    '16  5 wt% Pyruvic acid     ',\
    '17  Pyruvic acid      ',\
    '18  8 wt% Succinic acid    ',\
    '19  Succinic acid     ',\
    '20  Water at 293K     ']

    filesd=['organic_acids/ammoniumsulfate_imag.myhre',\
    'organic_acids/benzoic_imag.myhre',\
    'organic_acids/glutaric_25%_real.myhre',\
    'organic_acids/glutaric_50%_real.myhre',\
    'organic_acids/glutaric_5%_real.myhre',\
    'organic_acids/glutaric_imag.myhre',\
    'organic_acids/hydroxymalonic_imag.myhre',\
    'organic_acids/malonic_10%_real.myhre',\
    'organic_acids/malonic_25%_real.myhre',\
    'organic_acids/malonic_60%_real.myhre',\
    'organic_acids/malonic_imag.myhre',\
    'organic_acids/oxalic_10%_real.myhre',\
    'organic_acids/oxalic__5%_real.myhre',\
    'organic_acids/oxalic_imag.myhre',\
    'organic_acids/phthalic_imag.myhre',\
    'organic_acids/pinonic_imag.myhre',\
    'organic_acids/pyruvic_5%_real.myhre',\
    'organic_acids/pyruvic_imag.myhre',\
    'organic_acids/succinic_8%_real.myhre',\
    'organic_acids/succinic_imag.myhre',\
    'organic_acids/water_imag.myhre']

# *******
# Write out all possible compounds
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_myhreorganic: listset ")
        for i in range(0,21):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print("  To obtain a pair of indices (real and imaginary) ")
        print("  set mset to 2,3,4 7,8,9, 11,12, 16, or 18")
        print(" rd_myhreorganic: listset ")
        for i in range(0,21):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter mset to 2,3,4, 7,8,9,   11,12,  16,  or 18")

# Will specify mset e.g. mset=3
    msetstr=input(" mset: ")
    mset=int(msetstr)

    titlegr="Myhre "+listset2[mset]

# For real file mset value ,find the corresponding ipair file index for the iamg file
# to do a Mie calculation
    ipair=-99
    for i in range(0,npairs):
        if mset == ireal[i]:
            ipair=iimag[i]

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
# Data: Real indices of 50 wt% Glutaric acid at 293K from
# 0.25 to 1.25 microns

# Reference: C.E. Lund Myhre and C.J. Nielsen
# Optical properties in the UV and visible spectral region
# of organic acids relevant to tropospheric aerosols
# Atmos. Chem. Phys. volume 4, pgs. 1759-1769, 2004.

# Email contact person: C. J. Nielsen (claus.nielsen@kjemi.uio.no)

# Format: 1001 lines (2x,f8.2,2x,f10.4,2x,1p,e10.3)
#   cm-1        microns   Real
#  8000.00      1.2500   1.38830
#  8006.40      1.2490   1.38830
#  8012.82      1.2480   1.38830
#  8019.25      1.2470   1.38830

# *****************
# Read in the real indices
    fdat=filesd[mset]
    filework=froot+fdat
    fileworkr=filework

# Open input ascii file for reading
    f = open(filework,'r')

# Loop over the data lines
    i=0
    for line in f:
        i=i+1

        nadd1=0
        nadd2=1
        i1=13
        i2=i1+ndatr-1
        idiff=i1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            a2=float(columns[1])
            jj=1
            for j in range(nadd1,nadd2):
                wavelengthr2[ii]=1.0e4/a1
                wcmr2[ii]=a1
                jj=jj+1
                rndat2[ii]=float(columns[jj])

# Close the input ascii file
    f.close()

# *****************
# Data: Imaginary indices of glutaric acid at 293K from
# 0.25 to 1.1 microns

# Reference: C.E. Lund Myhre and C.J. Nielsen
# Optical properties in the UV and visible spectral region
# of organic acids relevant to tropospheric aerosols
# Atmos. Chem. Phys. volume 4, pgs. 1759-1769, 2004.

# Email contact person: C. J. Nielsen (claus.nielsen@kjemi.uio.no)

# Format: 851 lines (2x,f8.2,2x,f10.4,2x,1p,e10.3)
#   cm-1        microns   Imaginary
#  9090.91      1.1000   7.000e-09
#  9099.18      1.0990   7.000e-09
#  9107.47      1.0980   7.000e-09
#  9115.77      1.0970   7.000e-09

# *****************
# Read in the imaginary indices
# Note that the number of lines differs from the real indcies
    fdat=filesd[ipair]
    filework=froot+fdat
    fileworki=filework

# Open input ascii file for reading
    f = open(filework,'r')

# Loop over the data lines
    i=0
    for line in f:
        i=i+1

        nadd1=0
        nadd2=1
        i1=13
        i2=i1+ndati-1
        idiff=i1
        if i >= i1 and i <= i2:
            line=line.strip()
            columns=line.split()
            ii=i-idiff
            a1=float(columns[0])
            a2=float(columns[1])
            jj=1
            for j in range(nadd1,nadd2):
                wavelengthi2[ii]=1.0e4/a1
                wcmi2[ii]=a1
                jj=jj+1
                ridat2[ii]=float(columns[jj])

# Close the input ascii file
    f.close()

# **********************************
# Place the input data  into the rndat and ridat vectors
    a1=wavelengthr2[0]
    a2=wavelengthr2[1]

    if a1 < a2:
        for i in range(0,nlinesr):
            wavelengthr[i]=wavelengthr2[i]
            wcmr[i]=wcmr2[i]
            rndat[i]=rndat2[i]
        for i in range(0,nlinesi):
            wavelengthi[i]=wavelengthi2[i]
            wcmi[i]=wcmi2[i]
            ridat[i]=ridat2[i]

    if a2 < a1:
        i2=nlinesr
        for i in range(0,nlinesr):
            i2=i2-1
            wavelengthr[i]=wavelengthr2[i2]
            wcmr[i]=wcmr2[i2]
            rndat[i]=rndat2[i2]
        i2=nlinesi
        for i in range(0,nlinesi):
            i2=i2-1
            wavelengthi[i]=wavelengthi2[i2]
            wcmi[i]=wcmi2[i2]
            ridat[i]=ridat2[i2]

# **********************************
# Since the number of real and imag indices are different, need to 
# 
# *****************
# Since nlinesi ne nlinesr, you will use calc_samewave.py to get values at similar wavelengths
    list6=[nlinesr,wcmr,wavelengthr,rndat,nlinesi,wcmi,wavelengthi,ridat,titlegr]
#   print("list6 from rd_liusoaacp.py ",list6)

    filepickle = open("diffindxwave",'wb')
    pickle.dump(list6,filepickle)
    filepickle.close()

# *****
# Will match wavelength scales in calc_samewave.py, which reads in diffindxwave pickle
    import calc_samewave

# *****************
# Write out results
    if nopr == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_myhreorganic: fileworkr ")
        file_object.write(str(fileworkr))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_myhreorganic: fileworki ")
        file_object.write(str(fileworki))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_myhreorganic: mset,listset[mset] ")
        a0=listset[mset]
        listn1=[mset,a0]
        file_object.write(str(listn1))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_myhreorganic: ipair,listset[ipair] ")
        a0=listset[ipair]
        listn2=[ipair,a0]
        file_object.write(str(listn2))

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_myhreorganic()
