#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------

import numpy as np


class MarmotMaterial:
    """Material class for Marmot materials.

    Right now, this class only stores the material properties array and the material name.
    The underlying material model is accessed through Marmot finite elements.
    """

    def __init__(self, name: str, properties: np.ndarray):
        self._name = name  # set material name
        self._properties = properties  # set properties array

    @property
    def properties(self) -> np.ndarray:
        """Returns the material properties array.

        Returns
        -------
        np.ndarray
            Material properties array."""
        return self._properties

    @property
    def name(self) -> str:
        """Returns the material name.

        Returns
        -------
        str
            Material name."""
        return self._name
