#! /usr/bin/env python
"""rd_liusoaest.py -  specify the refractive indices"""
def rd_liusoaest():
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
    listset=['0 Alpha Pinene ','1 Limonene ','2 Catechol']
# First line of data
    ilinesj=[19, 178, 377 ]
# Number of wavelengths
    nlinesj=[155, 195, 195 ]

# *******
# write out all possible compounds
    iwrsp=1
    if iwrsp == 1:
        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_liusoaest: listset ")
        for i in range(0,3):
            file_object.write("\n")
            a0=listset[i]
            listn1=[a0]
            file_object.write(str(listn1))

# *******
# Write out to console
    iwrsp=2
    if iwrsp == 2:
        print(" rd_liusoaest: listset ")
        for i in range(0,3):
            a0=listset[i]
            listn1=[a0]
            print(str(listn1))
        print(" enter jset from 0 to 2")

# Will specify jset e.g. jset=3
    jsetstr=input(" jset: ")
    jset=int(jsetstr)

    titlegr="Liu SOA est  "+listset[jset]

# *****************
# Will read in from the liusoaest.dat ascii file

# Will store all of the data in the rndat and ridat arrays
# There are ndat max wavelengths
    ndat=200
    ncomp=3
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
#   fdat="/ur/massie/hitran/2020/hitran_aerosol_tarfile/ascii/single_files/liu_soa_est.dat"

    fdat="liu_soa_est.dat"

    filework=fileasciidir+fdat

# Open input ascii file for reading
    f = open(filework,'r')

# *****************
# Reference: Pengfei Liu, Yue Zhang, and Scot T. Martin
# Complex Refractive Indices of Thin Films of Secondary Organic
# Materials by Spectroscopic Ellipsometry from 220 to 1200 nm
# Env. Sci. Teach., volume 47, 13594-13601, 2013
# dx.doi.org/10.1021/es403411e

# Real and Imaginary Refractive Indices of SOA Aerosol

# Contact: S. T. Martin (scot_martin@harvard.edu)

# Laboratory conditions
#    1  alpha pinene
#    2  limonene
#    3  catechol

# alpha pinene
# nlines      155
# i,wcm(i),wavelength(i),rn(i),ri(i)
#   1 43478.2617     0.2300     1.5978     0.0065
#   2 42553.1875     0.2350     1.5912     0.0055
#   3 41666.6641     0.2400     1.5852     0.0037
#   4 40816.3242     0.2450     1.5797     0.0032
#   5 40000.0000     0.2500     1.5746     0.0029

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
                a1=float(columns[1])
                a2=float(columns[2])
                jj=2
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
        file_object.write(" rd_liusoaest: fileasciidir ")
        file_object.write(str(fileasciidir))
        file_object.write("\n")
        file_object.write(" rd_liusoaest: fdat ")
        file_object.write(str(fdat))
        file_object.write("\n")
        file_object.write(" rd_liusoaest: filework ")
        file_object.write(str(filework))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" rd_liusoaest: i,wavedat,wcmdat,rndex,ridex ")
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
#   print("list5 from rd_liusoaest.py ",list5)

    filepickle = open("indicesorig",'wb')
    pickle.dump(list5,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

rd_liusoaest() 
