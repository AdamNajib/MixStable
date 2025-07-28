import os
import tkinter as tk
from tkinter import filedialog
from rpy2 import robjects
from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects.vectors import StrVector

def setup_r_environment(required_packages=None, show_dialog=True):
    """
    Sets up an R environment using a user-chosen folder as the custom library path.
    Installs and loads required R packages using rpy2.

    Parameters:
    -----------
    required_packages : list of str
        List of R package names to install and load.
    show_dialog : bool
        If True, opens a dialog to choose the R library folder.
        If False, uses default path or environment variable.
    
    Returns:
    --------
    loaded_packages : dict
        A dictionary mapping package names to their imported rpy2 objects.
    r_lib_path : str
        The path to the custom R library folder.
    """
    # === Step 0: Choose folder interactively ===
    if show_dialog:
        root = tk.Tk()
        root.withdraw()
        print("📁 Please choose or create a folder where R packages will be installed.")
        r_lib_path = filedialog.askdirectory(title="Select R Library Folder")
        if not r_lib_path:
            raise Exception("❌ No folder selected. Aborting.")
    else:
        r_lib_path = os.path.expanduser("~/R_libs")  # default fallback path

    # === Step 1: Ensure folder exists ===
    if not os.path.exists(r_lib_path):
        os.makedirs(r_lib_path)
        print(f"📁 Created R library folder at: {r_lib_path}")
    else:
        print(f"📁 R library folder already exists at: {r_lib_path}")

    print("\n✅ R packages will be installed to:")
    print(f"   {r_lib_path}\n")

    # === Step 2: Set R to use this custom library path ===
    robjects.r(f'.libPaths("{r_lib_path}")')

    # === Step 3: Define required R packages ===
    if required_packages is None:
        required_packages = [
            "base", "libstable4u", "stabledist", "cubature", "alphastable", "invgamma",
            "LaplacesDemon", "lubridate", "magrittr", "mltest", "evmix", "nprobust",
            "BNPdensity", "stats"
        ]

    # === Step 4: Install missing packages ===
    utils = importr("utils")
    utils.chooseCRANmirror(ind=1)

    missing = [pkg for pkg in required_packages if not isinstalled(pkg)]
    if missing:
        print(f"📦 Installing missing R packages: {missing}")
        utils.install_packages(StrVector(missing), lib=r_lib_path)
    else:
        print("✅ All required R packages are already installed.")

    # === Step 5: Import all packages ===
    loaded_packages = {pkg: importr(pkg) for pkg in required_packages}
    print("📚 All packages successfully loaded into the R environment.")

    return loaded_packages, r_lib_path



# Load required R packages
base = importr("base")
libstable4u = importr("libstable4u")
stabledist = importr("stabledist")
cubature = importr("cubature")
alphastable = importr("alphastable")
invgamma = importr("invgamma")
LaplacesDemon = importr("LaplacesDemon")
lubridate = importr("lubridate")
magrittr = importr("magrittr")
mltest = importr("mltest")
evmix = importr("evmix")
nprobust = importr("nprobust")
BNPdensity = importr("BNPdensity")
stats = importr("stats")