<div align="center">
<img src="https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/blob/main/images/LIFDlogo.png"></a>
<a href="https://www.cemac.leeds.ac.uk/">
  <img src="https://github.com/cemac/cemac_generic/blob/master/Images/cemac.png"></a>
  <br>
</div>

# Leeds Institute for Fluid Dynamics Machine Learning For Earth Sciences #

# Random Forests

[![GitHub release](https://img.shields.io/github/release/cemac/LIFD_GenerativeAdversarialNetworks.svg)](https://github.com/cemac/LIFD_GenerativeAdversarialNetworks/releases) [![GitHub top language](https://img.shields.io/github/languages/top/cemac/LIFD_GenerativeAdversarialNetworks.svg)](https://github.com/cemac/LIFD_GenerativeAdversarialNetworks) [![GitHub issues](https://img.shields.io/github/issues/cemac/LIFD_GenerativeAdversarialNetworks.svg)](https://github.com/cemac/LIFD_GenerativeAdversarialNetworks/issues) [![GitHub last commit](https://img.shields.io/github/last-commit/cemac/LIFD_GenerativeAdversarialNetworks.svg)](https://github.com/cemac/LIFD_GenerativeAdversarialNetworks/commits/master) [![GitHub All Releases](https://img.shields.io/github/downloads/cemac/LIFD_GenerativeAdversarialNetworks/total.svg)](https://github.com/cemac/LIFD_GenerativeAdversarialNetworks/releases) ![GitHub](https://img.shields.io/github/license/cemac/LIFD_GenerativeAdversarialNetworks.svg)[![DOI](https://zenodo.org/badge/366734586.svg)](https://zenodo.org/badge/latestdoi/366734586)

[![LIFD_ENV_ML_NOTEBOOKS](https://github.com/cemac/LIFD_GenerativeAdversarialNetworks/actions/workflows/python-package-conda-GAN.yml/badge.svg)](https://github.com/cemac/LIFD_GenerativeAdversarialNetworks/actions/workflows/python-package-conda-GAN.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cemac/LIFD_GenerativeAdversarialNetworks/HEAD?labpath=GANS.ipynb)

This notebook explores Random Forests to find out what variables control leaf temperature

## Recommended Background Reading

## Quick look

If you want a quick look at the contents inside the notebook before deciding to run it please view the [md file](https://github.com/cemac/LIFD_GenerativeAdversarialNetworks/blob/main/GANS.md) generated (*note some HTML code not fully rendered*)


### Quick start

**Binder**

You can run this notebook on your personal laptop or via the [binder](https://mybinder.readthedocs.io/en/latest/index.html#what-is-binder) link above (please allow a few minutes for set up).

**Running Locally**

If you're already familiar with git, anaconda and virtual environments the environment you need to create is found in GAN.yml and the code below to install activate and launch the notebook. The .yml file has been tested on the latest linux, macOS and windows operating systems.

```bash
git clone git@github.com:cemac/LIFD_GenerativeAdversarialNetworks.git
cd LIFD_GenerativeAdversarialNetworks
conda env create -f GANS.yml
conda activate GANS
jupyter-notebook
```

## Installation and Requirements

This notebook is designed to run on a laptop with no special hardware required therefore recommended to do a local installation as outlined in the repository [howtorun](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/howtorun.md) and [jupyter_notebooks](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/jupyter_notebooks.md) sections.


# Licence information #

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">LIFD_ENV_ML_NOTEBOOKS</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://cemac.leeds.ac.uk/" property="cc:attributionName" rel="cc:attributionURL">cemac</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

## Acknowledgements

Thanks to Caitlin Howarth for the basis of this tutorial. This tutorial is part of the [LIFD ENV ML NOTEBOOKS](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS) please refer to for full acknowledgements.
