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
# Created on Fri Jan 27 19:53:45 2017

# @author: Matthias Neuner

import numpy as np


class Node:
    """A basic node.
    It has a label, a spatial position, and may be associated with an arbitrary number of fields.

    Parameters
    ----------
    label
        The unique label for this node.
    coordinates
        The coordinates of this node.
    """

    def __init__(
        self,
        label: int,
        coordinates: np.ndarray,
    ):
        self.label = label
        self.coordinates = coordinates
        self.fields = {}

    def setFields(self, *fields):
        """Activate fields on this node.

        Parameters
        ----------
        fields
           The fields to activate
        """
        self.fields.update(fields)
