# Note for example image reconstruction for Mapping human cerebral blood flow with high-density, multi-channel speckle contrast optical spectroscopy

To run the example SCOS image reconstruction notebook, first complete cedalion installation as described in the documentation below. You must be using the scos_validation branch to run the notebook. The example notebook can be found in the following directory within the repository:
examples/head_models/imagerecon_scos.ipynb

<p align="center">
    <img src="docs/img/IBS_clr_small.png" />
</p>

# cedalion - fNIRS analysis toolbox

To avoid misinterpretations and to facilitate studies in naturalistic environments, fNIRS measurements will increasingly be combined with recordings from physiological sensors and other neuroimaging modalities.
The aim of this toolbox is to facilitate this kind of analyses, i.e. it should allow the easy integration of machine learning techniques and provide unsupervised decomposition techniques for
multimodal fNIRS signals.

## Documentation

The [documentation](https://doc.ibs.tu-berlin.de/cedalion/doc/dev) contains
[installation instructions](https://doc.ibs.tu-berlin.de/cedalion/doc/dev/getting_started/installation.html) as
well as several [example notebooks](https://doc.ibs.tu-berlin.de/cedalion/doc/dev/examples.html)
that illustrate the functionality of the toolbox.
For discussions and help you can visit the [cedalion forum on openfnirs.org](https://openfnirs.org/community/cedalion/)


## Development environment

To create a conda environment with the necessary dependencies run:

```
$ conda env create -n cedalion -f environment_dev.yml
```

Afterwards activate the environment and add an editable install of `cedalion` to it:
```
$ conda activate cedalion
$ pip install -e .
$ bash install_nirfaster.sh CPU # or GPU
```

This will also install Jupyter Notebook to run the example notebooks.

If conda is too slow consider using the faster drop-in replacement [mamba](https://mamba.readthedocs.io/en/latest/).
If you have Miniconda or Anaconda you can install mamba with:
'''
$ conda install mamba -c conda-forge
'''
and then create the environment with
```
$ mamba env create -n cedalion -f environment_dev.yml
```
Please note: If this does not socceed there is another route to go:
Install the libmamba solver
'''
$ conda install -n base conda-libmamba-solver
'''
and then build the environment with the --solver=libmamba
```
$ conda env create -n cedalion -f environment_dev.yml --solver=libmamba
```

## How to cite Cedalion
A paper for the toolbox is currently in the making. If you use this toolbox for a publication in the meantime, please cite us using GitHub's  "Cite this repository" feature in the "About" section. If you want to contact us or learn more about the IBS-Lab please go to https://www.ibs-lab.com/

