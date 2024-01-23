###############
elastomer LGEDM
###############

|build| |license|

A repository that incorporates the composite uFJC model within a localizing gradient-enhanced damage model (LGEDM) for elastomers. This repository is dependent upon the latest version of the `composite-uFJC-scission <https://pypi.org/project/composite-ufjc-scission/>`_ Python package (1.4.0). It is also dependent upon the latest version of legacy FEniCS (2019.1.0).

*****
Setup
*****

Software and package installation is managed by Conda. The ``conda-install-instructions.tex`` file provides step-by-step instructions for how to install the necessary software and packages in this repository. These instructions sequentially include how to:

(i) install Miniconda,
(ii) create a Conda environment called ``elastomer-lgedm`` with Python 3.9 (a Python version that is compatible with legacy FEniCS),
(iii) install legacy FEniCS,
(iv) install pip, and
(v) install the ``composite-ufjc-scission`` PyPI package via pip.
(vi) clone this GitHub repository
(vi) After the Conda environment has been installed and this repository cloned, the ``elastomer-lgedm`` package needs to be installed in the environment by moving to the main project directory (where the setup and configuration files are located), and (with the Conda environment activated) executing ``pip install -e .`` for an editable installation or ``pip install .`` for a standard installation.

If need be, please refer to the general and most up-to-date Internet presence of FEniCS for assistance in the proper installation of legacy FEniCS (2019.1.0). 

*****
Usage
*****

The ``elastomer_lgedm`` directory contains the following files that constitute the ``elastomer-lgedm`` package: ``composite_ufjc.py``, ``rate_dependence_scission.py``, ``scission_model.py``, ``core.py``, ``composite_ufjc_network.py``, ``default_parameters.py``, ``problem.py``, and ``utility.py``. The ``notched-crack`` and ``refined-notched-crack`` directories respectively contain ``notched_crack.py`` and ``refined_notched_crack.py``, the main executible files for this repository. Each of these files creates the finite element problem for the elastomer LGEDM, solves the problem, and (to a certain extent) post-processes results. Before using these codes, it is highly recommended that you carefully examine each and every one of the aforementioned Python codes to understand how the code works as a whole, how certain parts of the code depend on other packages, how certain parts of the code relate to one another, and how the code is generally structured (in an object-oriented fashion). If necessary, feel free to modify any of these codes for your purposes.

In order to run either of the main executible files in serial, first activate the Conda environment, and then execute the following command in the terminal

::

    python3 {notched_crack, refined_notched_crack}.py

In order to run either of the main executible files in parallel (thanks to the parallel computing capabilities of FEniCS), first activate the Conda environment, and then execute the following command in the terminal

::

    mpirun -np number_of_cores python3 {notched_crack, refined_notched_crack}.py

Do note that the codes published as is in this repository are unable to be run in parallel (due to the ASCII-encoding of results in XDMF files, which is required for the most recent versions of ParaView to be able to open XDMF files produced via FEniCS).

***********
Information
***********

- `License <https://github.com/jasonmulderrig/elastomer-LGEDM/LICENSE>`__
- `Releases <https://github.com/jasonmulderrig/elastomer-LGEDM/releases>`__
- `Repository <https://github.com/jasonmulderrig/elastomer-LGEDM>`__

********
Citation
********

\Jason Mulderrig, Brandon Talamini, and Nikolaos Bouklas, ``composite-ufjc-scission``: the Python package for the composite uFJC model with scission, `Zenodo (2022) <https://doi.org/10.5281/zenodo.7335564>`_.

\Jason Mulderrig, Brandon Talamini, and Nikolaos Bouklas, Statistical mechanics-based gradient-enhanced damage for elastomeric materials, In preparation.

\Jason Mulderrig, Brandon Talamini, and Nikolaos Bouklas, A statistical mechanics framework for polymer chain scission, based on the concepts of distorted bond potential and asymptotic matching, `Journal of the Mechanics and Physics of Solids 174, 105244 (2023) <https://www.sciencedirect.com/science/article/pii/S0022509623000480>`_.

..
    Badges ========================================================================

.. |build| image:: https://img.shields.io/github/checks-status/jasonmulderrig/elastomer-LGEDM/main?label=GitHub&logo=github
    :target: https://github.com/jasonmulderrig/elastomer-LGEDM

.. |license| image:: https://img.shields.io/github/license/jasonmulderrig/elastomer-LGEDM?label=License
    :target: https://github.com/jasonmulderrig/elastomer-LGEDM/LICENSE