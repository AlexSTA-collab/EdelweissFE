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
"""
Created on Fri Feb 10 19:20:25 2017

@author: Matthias Neuner
"""

import importlib

solverLibrary = {
    "NIST": "nonlinearimplicitstatic",
    "NISTParallel": "nonlinearimplicitstaticparallelmk2",
    "NISTParallelForMarmotElements": "nonlinearimplicitstaticparallel",
    "NISTPArcLength": "nonlinearimplicitstaticparallelarclength",
}


def getSolverByName(name):
    solver = importlib.import_module("fe.solvers.{:}".format(solverLibrary[name]))
    return getattr(solver, name)
