"""Getting EPCAPE data: configuration, ARM Live downloads, and combining daily
files into one netCDF per product.

    config.py         config.yaml reader: campaign dates, machine paths, products
    credentials.py    ARM username/token lookup (~/.arm_credentials)
    armlive.py        ARM Live web-service client (query, saveData, mod)
    arm_files.py      ARM file names and local file discovery
    sync.py           resumable downloads (variable subsets or complete files)
    combine.py        daily files -> data/processed/<product>_<start>_<end>.nc
    download_data/download_arm.py   command line:  python download_data/download_arm.py <product>
    download_data/combine_product.py command line: python download_data/combine_product.py <product>
"""
