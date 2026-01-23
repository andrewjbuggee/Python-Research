# Output Filename Reference

## Naming Convention
All output files follow this pattern:
```
wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_XXX.cdf
```

Where XXX is a 3-digit zero-padded alpha value.

## Complete List of 48 Output Files

### Alpha 1-40 (continuous)
```
wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_001.cdf
wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_002.cdf
wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_003.cdf
...
wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_038.cdf
wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_039.cdf
wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_040.cdf
```

### Alpha 45-60 (every 5)
```
wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_045.cdf
wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_050.cdf
wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_055.cdf
wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_060.cdf
```

### Alpha 80, 100, 125, 150
```
wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_080.cdf
wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_100.cdf
wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_125.cdf
wc_mieTable_gamma_rEff_1-35microns_gammaDist_alpha_150.cdf
```

## File Properties
- Format: NETCDF4 (same as input files)
- Extension: .cdf (compatible with libRadtran)
- Size: ~250-450 MB each
- Dimensions: 221 wavelengths × 35 radii

## Location
All files will be created in:
```
/projects/anbu8374/Matlab-Research/Radiative_Transfer_Physics/mieTables_gamma/netCDF_gammaDist_more_rEffs_moreAlpha/combined_by_alpha/
```

## Quick Check
To verify all files were created:
```bash
cd /projects/anbu8374/Matlab-Research/Radiative_Transfer_Physics/mieTables_gamma/netCDF_gammaDist_more_rEffs_moreAlpha/combined_by_alpha/

# Count files
ls -1 wc_mieTable_gamma_*.cdf | wc -l
# Should output: 48

# List all files
ls -lh wc_mieTable_gamma_*.cdf

# Check file sizes
du -sh wc_mieTable_gamma_*.cdf
```
