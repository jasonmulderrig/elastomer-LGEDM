###############
elastomer LGEDM
###############

|build| |license|

A repository that incorporates the composite uFJC model within a localizing gradient-enhanced damage model (LGEDM) for elastomers. This repository is dependent upon the latest version of the `composite-uFJC-scission <https://pypi.org/project/composite-ufjc-scission/>`_ Python package (1.4.0). It is also dependent upon the latest version of legacy FEniCS (2019.1.0).

**********************
Installation and Setup
**********************

Software and package installation is managed by conda. In what follows in this section are four subsections of instructions for installing conda via Miniconda, creating a conda environment, and then in that conda environment, installing legacy FEniCS, some other necessary dependencies, and the ``elastomer-lgedm`` package. It is highly recommended that these instructions be followed in the order they appear below.

As a disclaimer, and if need be, please refer to the general and most up-to-date Internet presence of FEniCS for further assistance in the proper installation of legacy FEniCS (2019.1.0). The following instructions worked successfully for me in January 2024, but there could be a chance that future FEniCS developments render these instructions unhelpful.

---------------------------------
Install Miniconda with Python 3.9
---------------------------------

I recommend that conda be installed via Miniconda. Moreover, I recommend that Miniconda be installed with Python 3.9 because this is a relatively up-to-date version of Python which legacy FEniCS is compatible with. The step-by-step instructions I used to install Miniconda are provided below:

(1) Gather your particular computer system information (my computer is a 64-bit Linux machine)
(2) Download the Miniconda installer shell script from the `Miniconda installer website <https://docs.conda.io/projects/miniconda/en/latest/miniconda-other-installer-links.html>`_ to your computer (I downloaded the Python 3.9 Miniconda3 Linux 64-bit installer)
(3) Go to the directory where the Miniconda installer shell script is located, and make the script executable by running ``chmod +x miniconda_installer_filename.sh``
(4) Run the Miniconda installer shell script via the command ``bash miniconda_installer_filename.sh``, and accept all of the default settings during the installation
(5) When the installation is complete, log out of the computer, and then log back in
(6) Test your installation by running ``conda list``. If conda has been installed correctly, a list of installed packages appears

--------------------------------------------------------------------------
Create the ``elastomer-lgedm`` conda environment and install legacy FEniCS
--------------------------------------------------------------------------

Once conda has been installed via Miniconda, a conda environment can be created with legacy FEniCS. Let's call this particular conda environment ``elastomer-lgedm``. The following set of instructions are used to install the ``elastomer-lgedm`` conda environment and then install legacy FEniCS in that environment:

(1) Create the ``elastomer-lgedm`` conda environment by running ``conda create -n elastomer-lgedm python=3.9`` (it is important that the ``python=3.9`` option be included in this command, for reasons described before)
(2) Open the ``elastomer-lgedm`` conda environment by running ``conda activate elastomer-lgedm``
(3) Install legacy FEniCS in the ``elastomer-lgedm`` conda environment by running ``conda install -c conda-forge fenics``
(4) Close the ``elastomer-lgedm`` conda environment by running ``conda deactivate``

---------------------------------------------------------------------------
Install necessary dependencies in the ``elastomer-lgedm`` conda environment
---------------------------------------------------------------------------

The next set of instructions are used to install additional necessary dependencies in the ``elastomer-lgedm`` conda environment:

(1) Open the ``elastomer-lgedm`` conda environment by running ``conda activate elastomer-lgedm``
(2) Install pip by running ``conda install pip``
(3) Explicitly verify the pip installed in the ``elastomer-lgedm`` conda environment by running ``which pip``. The output should be the particular pip that has been installed to the ``elastomer-lgedm`` conda environment, and this is the pip that needs to be explicitly called henceforth in the remaining instructions. Let's call this output as ``explicit_conda_pip``. The output should look similar to my analogous output: ``/home/jpm445/miniconda3/envs/elastomer-lgedm/bin/pip``
(4) Install the ``composite-ufjc-scission`` PyPI package by running ``explicit_conda_pip install composite-ufjc-scission``
(5) Install the ``matplotlib`` PyPI package by running ``explicit_conda_pip install matplotlib``
(6) Install the ``gmsh`` PyPI package by running ``explicit_conda_pip install --upgrade gmsh``
(7) Close the ``elastomer-lgedm`` conda environment by running ``conda deactivate``

------------------------------------------------------------------------------------
Install the ``elastomer-lgedm`` package in the ``elastomer-lgedm`` conda environment
------------------------------------------------------------------------------------

The final set of instructions are used to install the ``elastomer-lgedm`` package in the ``elastomer-lgedm`` conda environment:

(1) Clone or download the contents of this ``elastomer-LGEDM`` GitHub repository to your computer
(2) Move to the main project directory (where the setup and configuration files are located)
(3) Open the ``elastomer-lgedm`` conda environment by running ``conda activate elastomer-lgedm``
(4) Install the ``elastomer-lgedm`` package via an editable installation or a standard installation by running either ``explicit_conda_pip install -e .`` or ``explicit_conda_pip install .``, respectively. Note that ``explicit_conda_pip`` is what was determined from the prior set of instructions
(5) Close the ``elastomer-lgedm`` conda environment by running ``conda deactivate``

------------
Verification
------------

With the ``elastomer-lgedm`` conda environment opened, the output from running the command ``conda list`` should be identical to the content in the ``conda-requirements.txt`` file.

*****
Usage
*****

The ``elastomer_lgedm`` directory contains the following files that constitute the ``elastomer-lgedm`` package:

- ``composite_ufjc.py``
- ``rate_dependence_scission.py``
- ``scission_model.py``
- ``core.py``
- ``composite_ufjc_network.py``
- ``default_parameters.py``
- ``problem.py``
- ``utility.py``

The ``notched-crack`` and ``refined-notched-crack`` directories respectively contain ``notched_crack.py`` and ``refined_notched_crack.py``, the main executable files for this repository. Each of these files creates the finite element problem for the elastomer LGEDM, solves the problem, and (to a certain extent) post-processes results. Before using these codes, it is highly recommended that you carefully examine each and every one of the aforementioned Python codes to understand how the code works as a whole, how certain parts of the code depend on other packages, how certain parts of the code relate to one another, and how the code is generally structured (in an object-oriented fashion). If necessary, feel free to modify any of these codes for your purposes.

In order to run either of the main executable files in serial, first activate the Conda environment, and then execute the following command in the terminal

::

    python3 {notched_crack, refined_notched_crack}.py

In order to run either of the main executable files in parallel (thanks to the parallel computing capabilities of FEniCS), first activate the Conda environment, and then execute the following command in the terminal

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