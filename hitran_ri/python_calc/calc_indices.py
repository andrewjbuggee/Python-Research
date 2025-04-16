#! /usr/bin/env python
"""calc_indices.py -  specify the refractive indices"""
def calc_indices():
    import sys
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
# Obtain the data from init_wave.py
#   list2=[iwave,nwave,wcm,wavelength,iset]
#   print("list2 from calc_wave.py ",list1)

    filepickle = open("wavescale",'rb')
    list2=pickle.load(filepickle)
    filepickle.close()

    iwave=list2[0]
    nwave=list2[1]
    wcm=list2[2]
    wavelength=list2[3]

    iset=list2[4]

# *****************
# Write out input
    iwrinput=0
    if iwrinput == 1:

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_indices: input iset\n ")
        file_object.write(str(iset))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_indices: input nwave\n ")
        file_object.write(str(nwave))

        file_object.write("\n")
        file_object.write(" calc_indices: input wcm\n ")
        file_object.write(str(wcm))

        file_object.write("\n")
        file_object.write(" calc_indices: input wavelength\n ")
        file_object.write(str(wavelength))

# *****************
    itest=0

    if itest == 1:
        rndex=np.zeros(nwave)
        ridex=np.zeros(nwave)

        for i in range(0,nwave):
            rndex[i]=1.33
            ridex[i]=0.05

# *****************
    if itest == 0:
        rndex=np.zeros(nwave)
        ridex=np.zeros(nwave)
        ifound=0

# Standard set of  17 materials compiled by Shettle
        if iset == 0:
            import rd_shettle
            ifound=1

# Water indices from  2 to 1000 microns
        if iset == 1:
            import rd_downingwater
            ifound=1

# Supercooled water
        if iset == 2:
            import rd_wagnersuper
            ifound=1

# Ice indices from 0.04 microns to 2 meters
        if iset == 3:
            import rd_warren
            ifound=1
 
# Ice indcies at various cold temperatures
        if iset == 4:
            import rd_clappice
            ifound=1

# Liquid H2SO4 at stratospheric temperatures
        if iset == 5:
            import rd_tisdale
            ifound=1

# Liquid H2SO4 at stratospheric temperatures
        if iset == 6:
            import rd_myhreh2so4
            ifound=1

# Liquid HNO3 at stratospheric tempratures
        if iset == 7:
            import rd_myhrehno3
            ifound=1

# Ternary solution (
        if iset == 8:
            import rd_myhreternary
            ifound=1

        if iset == 9:
            import rd_niedzielanad
            ifound=1

        if iset == 10:
            import rd_richwinenat
            ifound=1

        if iset == 11:
            import rd_toonpsc
            ifound=1

        if iset == 12:
            import rd_wagnersahara
            ifound=1

        if iset == 13:
            import rd_dibiagiodesert
            ifound=1

        if iset == 14:
            import rd_graingerash
            ifound=1

        if iset == 15:
            import rd_deguine
            ifound=1

        if iset == 16:
            import rd_reedash
            ifound=1

        if iset == 17:
            import rd_liusoaacp
            ifound=1

        if iset == 18:
            import rd_liusoaest
            ifound=1

        if iset == 19:
            import rd_hesoa
            ifound=1

        if iset == 20:
            import rd_fangsoa
            ifound=1

        if iset == 21:
            import rd_hashemivan
            ifound=1

        if iset == 22:
            import rd_myhreorganic
            ifound=1

        if iset == 23:
            import rd_alexander
            ifound=1

        if iset == 24:
            import rd_sutherlandbio
            ifound=1

        if iset == 25:
            import rd_shepherdwood
            ifound=1

        if iset == 26:
            import rd_magifire
            ifound=1

        if iset == 27:
            import rd_staggcarbon
            ifound=1

        if iset == 28:
            import rd_changsoot
            ifound=1

        if iset == 29:
            import rd_querrymineral
            ifound=1

        if iset == 30:
            import rd_toonmineral
            ifound=1

        if iset == 31:
            import rd_chehabkaolinite
            ifound=1

        if iset == 32:
            import rd_herbinquartz
            ifound=1

        if iset == 33:
            import rd_kharetholins
            ifound=1

        if iset == 34:
            import rd_ramireztitan
            ifound=1

        if iset == 35:
            import rd_imanaka
            ifound=1

        if iset == 36:
            import rd_jovanovicpluto
            ifound=1

        if iset == 37:
            import rd_querrykcl
            ifound=1

        if iset == 38:
            import rd_querryzns
            ifound=1

        if iset == 39:
            import rd_henningsio2
            ifound=1

        if iset == 40:
            import rd_zeidlersio2
            ifound=1

        if iset == 41:
            import rd_begemannal2o3
            ifound=1

        if iset == 42:
            import rd_henningfeo
            ifound=1

        if iset == 43:
            import rd_poschcatio3
            ifound=1

        if iset == 44:
            import rd_triaudfe2o3
            ifound=1

        if iset == 45:
            import rd_jenafe2sio4
            ifound=1

        if iset == 46:
            import rd_fabianfe2sio4
            ifound=1

        if iset == 47:
            import rd_fabianmgal2o4
            ifound=1

        if iset == 48:
            import rd_jagermg2sio4
            ifound=1

        if iset == 49:
            import rd_jagermgsio3
            ifound=1

        if iset == 50:
            import rd_zeidlertio2rut
            ifound=1

        if iset == 51:
            import rd_zeidlertio2ana
            ifound=1

        if iset == 52:
            import rd_poschtio2
            ifound=1

# supplementary indices

        if iset == 53:
#           import rd_kuo
#  for ifound=2, only imaginary indices are  in the input data file
            ifound=2

        if iset == 54:
#           import rd_sinyuk
            ifound=2

        if iset == 55:
            import rd_dingle
            ifound=3

        if iset == 56:
            import rd_zarzana
            ifound=3

        if iset == 57:
            import rd_querrydiesel
            ifound=1

        if iset == 58:
            import rd_niedzielah2so4
            ifound=1

        if iset == 59:
            import rd_biermannh2so4
            ifound=1

        if iset == 60:
            import rd_palmerwilliams
            ifound=1

        if iset == 61:
            import rd_normanhno3
            ifound=1

        if iset == 62:
            import rd_biermannhno3
            ifound=1

        if iset == 63:
            import rd_querrytylerhno3
            ifound=1

        if iset == 64:
            import rd_remsberg
            ifound=1

        if iset == 65:
            import rd_hasenkopf
            ifound=3

# oops, 
        if ifound == 0:
            print(" ")
            print(" Did not find refractive index data file, will stop ")
            print(" Respecify iset in work.dat ")
            file_object.write("\n")
            file_object.write("\n")
            file_object.write(" calc_indices: ifound=1, will stop ")
            file_object.write("\n")
            file_object.write(" calc_indices: Repecify iset in work.dat ")
            sys.exit()

# oops, 
        if ifound == 2:
            print(" ")
            print(" Only imaginary indices are in input data  file, will stop ")
            file_object.write("\n")
            file_object.write("\n")
            file_object.write(" calc_indices: Only imaginary indices are in input file")
            file_object.write("\n")
            file_object.write(" calc_indices: ifound=2, will stop ")
            sys.exit()

# oops, 
        if ifound == 3:
            print(" ")
            print(" There is only one wavelength to work with, will stop ")
            file_object.write("\n")
            file_object.write("\n")
            file_object.write(" calc_indices: Only one wavelenggth, will stpp")
            file_object.write("\n")
            file_object.write(" calc_indices: ifound=3, will stop ")
            sys.exit()

# *****************
# Read in from the pickle file
# Obtain the data from the rd routine
#   list5=[ndat,wavedat,wcmdat,rnval,rival,titlegr]
#   print("list5 from rd_shettle.py ",list5)

    filepickle = open("indicesorig",'rb')
    list5=pickle.load(filepickle)
    filepickle.close()

    ndat=list5[0]
    wavedat=list5[1]
    wcmdat=list5[2]
    rnval=list5[3]
    rival=list5[4]
    titlegr=list5[5]

    diffwi=np.zeros(nwave)

# *****************
# You have constructed the rd_xxx.py modules so that wavedat increases as j increases
# for wavelength (iwave=2)
# Using the rnval and rival values, interpolate to nwave grid
    if itest == 0 and iwave == 2:

# Check that the wwavelength range is consistent with the wavedat range

        if wavelength[0] < wavedat[0] or wavelength[nwave-1] > wavedat[ndat-1]:
            print(" ")
            print(" wavelength and wavedat ranges are different ")
            print(" Need to respecify w1,w2 in work.dat ")
            print(" Will stop in calc_indices.py ")
            file_object.write("\n")
            file_object.write("\n")
            file_object.write(" calc_indices: wavelength range ")
            listw1=[wavelength[0],wavelength[nwave-1]]
            file_object.write(str(listw1))
            file_object.write("\n")
            file_object.write(" calc_indices: wavedat range ")
            listw2=[wavedat[0],wavedat[ndat-1]]
            file_object.write(str(listw2))
            sys.exit()

        j1=0
        j2=ndat
        for i in range(0,nwave):
            w22=wavelength[i]
            for j in range(j1,j2):
                wj=wavedat[j]
                if wj >= w22: 
                    diffw=w22-wj
                    diffwi[i]=diffw
                    j3=j
                    j4=j+1
                    if j4 >= ndat:
                        j3=j-1
                        j4=j
                    diff0=wavedat[j4]-wavedat[j3]
                    deriv1=(rnval[j4]-rnval[j3])/diff0
                    deriv2=(rival[j4]-rival[j3])/diff0
                    adiff1=diffw*deriv1
#                   adiff1=0.0
                    rn2=rnval[j3]+adiff1
                    if rn2 < 0.0:
                        rn2=rnval[j3]
                    rndex[i]=rn2
                    adiff2=diffw*deriv2
#                   adiff2=0.0
                    ri2=rival[j3]+adiff2
                    if ri2 < 0.0:
                        ri2=rival[j3]
                    if ri2 < 0.0:
                        ri2=0.0
                    ridex[i]=ri2
                    j1=j
                    break

# Write out results
    if nopr == 1 and iwave == 2:

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_indices: output iset\n ")
        file_object.write(str(iset))

        file_object.write("\n")
        file_object.write(" calc_indices: output nwave\n ")
        file_object.write(str(nwave))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_indices: i,wavelength,rndex,ridex,diffwi ")
        for i in range(0,nwave):
            file_object.write("\n")
            a0="% 4d" %i
            a1="% 10.4f" %wavelength[i]
            a2="% 10.4f" %rndex[i]
            a3="% 10.4f" %ridex[i]
            a4="% 10.4f" %diffwi[i]
            listn1=[a0,a1,a2,a3,a4]
            file_object.write(str(listn1))


# *****************
# For iwave=1 the wavenumber scale is of interest
    if itest == 0 and iwave == 1:

# Check that the wwavelength range is consistent with the wavedat range

        if wcm[0] < wcmdat[ndat-1] or wcm[nwave-1] > wcmdat[0]:
            print(" ")
            print(" wcm and wcmdat ranges are different ")
            print(" Need to respecify w1,w2 in work.dat ")
            print(" Will stop in calc_indices.py ")
            file_object.write("\n")
            file_object.write("\n")
            file_object.write(" calc_indices: output wcm range ")
            listw1=[wcm[0],wcm[nwave-1]]
            file_object.write(str(listw1))
            file_object.write("\n")
            file_object.write(" calc_indices: wcmdat range ")
            listw2=[wcmdat[0],wcmdat[ndat-1]]
            file_object.write(str(listw2))
            sys.exit()

        j1=-1
        j2=ndat-1
        jskip=-1
        j2last=j2
        for i in range(0,nwave):
            w22=wcm[i]
            for j in range(j2,j1,jskip):
                wj=wcmdat[j]
                if wj >= w22: 
                    diffw=w22-wj
                    diffwi[i]=diffw
                    j3=j
                    j4=j-1
                    if j4 <= 0:
                        j3=j+1
                        j4=j
                    diff0=wcmdat[j4]-wcmdat[j3]
                    deriv1=(rnval[j4]-rnval[j3])/diff0
                    deriv2=(rival[j4]-rival[j3])/diff0
                    adiff1=diffw*deriv1
#                   adiff1=0.0
                    rn2=rnval[j3]+adiff1
                    if rn2 < 0.0:
                        rn2=rnval[j3]
                    rndex[i]=rn2
                    adiff2=diffw*deriv2
#                   adiff2=0.0
                    ri2=rival[j3]+adiff2
                    if ri2 < 0.0:
                        ri2=rival[j3]
                    if ri2 < 0.0:
                        ri2=0.0
                    ridex[i]=ri2
                    j2=j+1
                    if j2 >= j2last: 
                        j2=j2last
                    break

# *****************
# Write out results
    if nopr == 1 and iwave ==1:

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_indices: output iset\n ")
        file_object.write(str(iset))

        file_object.write("\n")
        file_object.write(" calc_indices: output nwave\n ")
        file_object.write(str(nwave))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" calc_indices: i,wcm,rndex,ridex,diffwi ")
        for i in range(0,nwave):
            file_object.write("\n")
            a0="% 4d" %i
            a1="% 10.4f" %wcm[i]
            a2="% 10.4f" %rndex[i]
            a3="% 10.4f" %ridex[i]
            a4="% 10.4f" %diffwi[i]
            listn1=[a0,a1,a2,a3,a4]
            file_object.write(str(listn1))

# If you just want to test this routine, stop here
#   sys.exit()

# *****************
    ichecktype=0
    if ichecktype == 1:
        print(rndex.dtype)
        print(ridex.dtype)

# *****************
# One way to pass data out of a function is to   pickle   the data
    list3=[iwave,nwave,wcm,wavelength,rndex,ridex,titlegr]
#   print("list3 from calc_indices.py ",list3)

    filepickle = open("indicesdat",'wb')
    pickle.dump(list3,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

calc_indices() 
