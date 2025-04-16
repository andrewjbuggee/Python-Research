#! /usr/bin/env python
"""ri_main.py - main module"""
def main():

# Open the output f.out ascii file
    import os

    fileout="f.out"
    file_object = open(fileout,'w')
    file_object.write("\n")
    file_object.write(" /python_calc/ri_main.py\n")
    file_object.write(" Hitran-ri calculation\n")
    file_object.close()
    
# Initialize the problem. Will write out to pickle file
    import init_calc

# Calculate the wavelength and wavenumber scales
    import calc_wave

# Specify the refractive indices
    import calc_indices

# Calculate the size distribution
    import calc_sized

# Calculate the extinction spectrum
    import calc_ext

# Close the output f.out ascii file
    file_object.close()

    print("Done")

main() 
