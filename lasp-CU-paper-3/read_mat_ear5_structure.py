import scipy.io
from pathlib import Path
import numpy as np

mat_dir = Path('/Volumes/My Passport/neural_network_training_data/combined_vocals_oracles_training_data_12_April_2026/')
path = next(f for f in sorted(mat_dir.glob('*.mat')) if not f.name.startswith('._'))

d = scipy.io.loadmat(path, squeeze_me=True)

# Top-level keys
print("Top-level keys:", [k for k in d.keys() if not k.startswith('_')])

# ERA5 structure
era5 = d['era5']
print("\nera5 type:", type(era5), "  shape:", getattr(era5, 'shape', 'N/A'))
print("era5 dtype:", era5.dtype)

# Try to access datProfiles
try:
    dp = era5['datProfiles']
    print("\ndatProfiles type:", type(dp))
    print("datProfiles dtype:", dp.dtype)
    
    T    = dp['T'][()]    if hasattr(dp, '__getitem__') else dp.item()['T']
    vap  = dp['vapor_concentration'][()] if hasattr(dp, '__getitem__') else dp.item()['vapor_concentration']
    gph  = dp['GP_height'][()] if hasattr(dp, '__getitem__') else dp.item()['GP_height']
    
    print("\nT             shape:", np.atleast_1d(T).shape,   " min/max:", T.min(), T.max(), "K")
    print("vapor_conc    shape:", np.atleast_1d(vap).shape, " min/max:", vap.min(), vap.max(), "molec/cm³")
    print("GP_height     shape:", np.atleast_1d(gph).shape, " min/max:", gph.min(), gph.max())
except Exception as e:
    print("Access error:", e)
    # Try alternate nested access
    print("\nTrying .item():")
    print(era5.item())
