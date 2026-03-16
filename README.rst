=========
Advection
=========


.. image:: https://img.shields.io/pypi/v/advection.svg
        :target: https://pypi.python.org/pypi/advection

.. image:: https://img.shields.io/travis/inkenbrandt/advection.svg
        :target: https://travis-ci.com/inkenbrandt/advection

.. image:: https://readthedocs.org/projects/advection/badge/?version=latest
        :target: https://advection.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

A lightweight Python toolkit for diagnosing **horizontal and vertical
advection** and correcting the **energy-balance closure gap** in
eddy-covariance data.

* Free software: MIT license
* Documentation: https://advection.readthedocs.io.


Purpose
-------

Eddy-covariance towers routinely under-sum the available energy—the measured
sensible (H) plus latent (LE) heat flux is typically 10–20% lower than net
radiation minus ground heat flux (Rn - G). This package provides functions
to calculate advective fluxes and apply corrections to improve energy balance
closure, based on methods described by Wang et al. (2024).

Setup
-----

To install the latest version from PyPI:

.. code-block:: bash

    pip install advection

For development, clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/inkenbrandt/advection.git
    cd advection
    pip install -e ".[dev]"

Usage
-----

Basic example of detecting advection and computing corrections:

.. code-block:: python

    import numpy as np
    from advection import advect_detect, advection

    # Example data
    main_data = {
        'H': [50, 60, -10],
        'LE': [200, 220, 50],
        'Rn': [300, 310, 100],
        'G': [20, 25, 10]
    }

    # Detect horizontal advection
    flags_h = advect_detect.detect_horizontal_advection(
        main_flux=main_data['H'],
        le_main=main_data['LE'],
        rn=main_data['Rn'],
        g=main_data['G']
    )

    # Compute advection fluxes
    out = advection.compute_advection_fluxes(
        main_data=main_data,
        detect_horizontal=flags_h
    )

    print(out['H_adv'])

Features
--------

* Detect horizontal and vertical advection using multi-tower or single-tower criteria.
* Compute advective heat and moisture fluxes.
* Apply corrections to energy balance components.
* Utility functions for atmospheric physics (air density, specific humidity, etc.).

Credits
-------

Based on work by Wang and others, 2024 (10.1016/j.agrformet.2024.110196)

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
