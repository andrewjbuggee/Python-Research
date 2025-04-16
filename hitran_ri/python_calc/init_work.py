#! /usr/bin/env python
"""init_work.py - read in input data from the work.dat file"""
def init_work():
    import os
    import numpy as np
    import pickle

# *****************
# Open the output f.out ascii file
    filework="work.dat"
    file_object = open(filework,'r')

    nopr=1

# *****************
    nlines=17

# *****************
# Write out results
    if nopr == 1:

        file_object.write("\n")
        file_object.write(" init_work: iwave,w1,w2,dw\n ")
        file_object.write(str(listn1))

        file_object.write("\n")
        file_object.write("\n")
        file_object.write(" init_work: den1,rad1,sig1\n ")
        file_object.write(str(listn2))

        file_object.write("\n")
        file_object.write(" init_work: den2,rad2,sig2\n ")
        file_object.write(str(listn3))

        file_object.write("\n")
        file_object.write(" init_work: r1,r2\n ")
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
    list1=[iwave,w1,w2,dw,den1,rad1,sig1,den2,rad2,sig2,r1,r2]
#   print("list1 from init_calc.py ",list1)

    filepickle = open("sizedistparam",'wb')
    pickle.dump(list1,filepickle)
    filepickle.close()

# *****************
# Close the output f.out ascii file
    file_object.close()

init_work() 
