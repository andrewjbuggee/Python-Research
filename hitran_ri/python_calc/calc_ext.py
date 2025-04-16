#! /usr/bin/env python
"""calc_ext.py -  calculate the extinction spectrum"""
def calc_ext():
    import os
    import math
    import numpy as np
    import pickle
    import bhmie

# *****************
# Open the output f.out ascii file
    fileout="f.out"
    file_object = open(fileout,'a')

    nopr=1

# *************************
# Obtain data from calc_indices.py
#   list3=[iwave,nwave,wcm,wavelength,rndex,ridex,titlegr]
#   print("list3 from calc_indices.py ",list3)

    filepickle = open("indicesdat",'rb')
    list3=pickle.load(filepickle)
    filepickle.close()

    iwave=list3[0]
    nwave=list3[1]
    wcm=list3[2]
    wavelength=list3[3]
    rndex=list3[4]
    ridex=list3[5]
    titlegr=list3[6]

# *************************
# Obtain data from calc_sized.py
#   list4=[ndist,radr,sized,dr]
#   print("list4 from calc_sized.py ",list4)

    filepickle = open("sizedist",'rb')
    list4=pickle.load(filepickle)
    filepickle.close()

    ndist=list4[0]
    radr=list4[1]
    sized=list4[2]
    dr=list4[3]

# *************************
# Write out input
    iwrinput=0
    if iwrinput == 1:

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_ext: input nwave\n ")
        file_object.write(str(nwave))

        if iwave == 1:
            file_object.write("\n")
            file_object.write(" calc_ext: input wcm\n ")
            file_object.write(str(wcm))

        if iwave == 2:
            file_object.write("\n")
            file_object.write(" calc_ext: input wavelength\n ")
            file_object.write(str(wavelength))

        file_object.write("\n")
        file_object.write(" calc_ext: input rndex\n ")
        file_object.write(str(rndex))

        file_object.write("\n")
        file_object.write(" calc_ext: input ridex\n ")
        file_object.write(str(ridex))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_ext: input ndist\n ")
        file_object.write(str(ndist))

        file_object.write("\n")
        file_object.write(" calc_ext: input radr\n ")
        file_object.write(str(radr))

        file_object.write("\n")
        file_object.write(" calc_ext: input sized\n ")
        file_object.write(str(sized))

        file_object.write("\n")
        file_object.write(" calc_ext: input dr\n ")
        file_object.write(str(dr))

# *************************
    pi=3.14159265
    const=math.sqrt(2.00*pi)
    constx=2.00*pi*1.0e-4
    refmed=1.00

    nang=20

# Output arrays
    bext=np.zeros(nwave)
    babs=np.zeros(nwave)
    bsca=np.zeros(nwave)
    asym=np.zeros(nwave)
    back=np.zeros(nwave)
    omega=np.zeros(nwave)

# The Mie x parameter  (two pi radius / wavelength)
    xsize=np.zeros(ndist)

# The complex refractive index
    refrelval=np.zeros(nwave,dtype=complex)

# ****
# Loop over spectra grid
    for i in range(0,nwave):

# Initialize for each spectra grid point
        bext[i]=0.0
        babs[i]=0.0
        bsca[i]=0.0
        asym[i]=0.0
        back[i]=0.0
        omega[i]=0.0

# The wavenumber
        wavcm=wcm[i]

# The complex index of refraction
        refre=rndex[i]
        refim=ridex[i]
#       refrel=complex(refre,refim)/refmed   (refmd=1.00)
        refrel=complex(refre,refim)
        refrelval[i]=refrel

# Loop over the size distribution
        for j in range(0,ndist):

# The Mie x parameter (2 pi Radius / Wavelength)
            x=constx*radr[j]*wavcm
            xsize[j]=x

# Use the Bohren and Huffman BHMIE routine
# return s1,s2,qext,qsca,qback,gsca
            result=bhmie.bhmie(x,refrel,nang)
            qext=result[2]
            qsca=result[3]
            qback=result[4]
            gfac=result[5]
 
# The Q absorption efficieny factor 
            qabs=qext-qsca

# Calculate the extinction, scattering, absorption coefficient
# 1.0e-3 converts cm-1 to km-1
            rd2=radr[j]*radr[j]
            weight=pi*rd2*sized[j]*dr[j]*1.0e-3

# Add to the output arrays
            bext[i]=bext[i]+(weight*qext)
            babs[i]=babs[i]+(weight*qabs)
            bsca[i]=bsca[i]+(weight*qsca)
            back[i]=back[i]+(weight*qback)
            asym[i]=asym[i]+(weight*qsca*gfac)

# Loop over the size distribution bins is done

# ***
# The asymmetry factor
        asym[i]=asym[i]/bsca[i]

# The single scattering albedo
        omega[i]=bsca[i]/bext[i]

# Loop over spectra wavelength points is done

# *************************
# Write out results

# For wavenumber scale
    if nopr == 1 and iwave == 1:

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_ext: output nwave\n ")
        file_object.write(str(nwave))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_ext: i,wcm,rndex,ridex,refrelval ")
        for i in range(0,nwave):
            file_object.write("\n")
            a0="% 4d" %i
            a1="% 10.4f" %wcm[i]
            a2="% 10.3e" %rndex[i]
            a3="% 10.3e" %ridex[i]
            listn1=[a0,a1,a2,a3,refrelval[i]]
            file_object.write(str(listn1))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_ext: i,wcm,bext,babs,bsca,asym,omega ")
        for i in range(0,nwave):
            file_object.write("\n")
            a0="% 4d" %i
            a1="% 10.4f" %wcm[i]
            a2="% 10.3e" %bext[i]
            a3="% 10.3e" %babs[i]
            a4="% 10.3e" %bsca[i]
            a5="% 10.3e" %asym[i]
            a6="% 10.3e" %omega[i]
            listn2=[a0,a1,a2,a3,a4,a5,a6]
            file_object.write(str(listn2))

# For wavelength scale
    if nopr == 1 and iwave == 2:

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_ext: output nwave\n ")
        file_object.write(str(nwave))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_ext: i,wavelength,rndex,ridex,refrelval ")
        for i in range(0,nwave):
            file_object.write("\n")
            a0="% 4d" %i
            a1="% 10.4f" %wavelength[i]
            a2="% 10.3e" %rndex[i]
            a3="% 10.3e" %ridex[i]
            listn1=[a0,a1,a2,a3,refrelval[i]]
            file_object.write(str(listn1))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_ext: i,wavelength,bext,babs,bsca,asym,omega ")
        for i in range(0,nwave):
            file_object.write("\n")
            a0="% 4d" %i
            a1="% 10.4f" %wavelength[i]
            a2="% 10.3e" %bext[i]
            a3="% 10.3e" %babs[i]
            a4="% 10.3e" %bsca[i]
            a5="% 10.3e" %asym[i]
            a6="% 10.3e" %omega[i]
            listn2=[a0,a1,a2,a3,a4,a5,a6]
            file_object.write(str(listn2))

# *************************
# Close the output f.out ascii file
    file_object.close()

# *************************
# Do a simple graph of the extinction spectrum

# To not do the graph put igraph=0 here in calc_ext.py
    igraph=1

    if igraph == 1:
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np

# For wavelength on x axix
        if iwave == 1:
            t = wcm
            s = 1.0e4*bext

            fig, ax = plt.subplots()
            ax.plot(t, s)

            ax.set(xlabel='Wavenumber (cm-1)', ylabel='Extinction (km-1 x 1.0e4)',
            title=titlegr)
            ax.grid()

            fig.savefig("ext.png")
            plt.show()

# For wavelength on x axix
        if iwave == 2:
            t = wavelength
            s = 1.0e4*bext

            fig, ax = plt.subplots()
            ax.plot(t, s)

            ax.set(xlabel='Wavelength (microns)', ylabel='Extinction (km-1 x 1.0e4)',
            title=titlegr)
            ax.grid()

            fig.savefig("ext.png")
            plt.show()

# *************************
calc_ext()
