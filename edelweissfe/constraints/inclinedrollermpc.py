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

"""Multi-point-constraint form of the interface-aligned roller.

Guides the same boundary as :mod:`~edelweissfe.constraints.inclinedroller` and enforces the same equations,

- guided faces: ``n . u = sin(theta) u_x + cos(theta) u_z = 0`` and ``u_y = 0``,
- Y-normal (symmetry) faces: ``u_y = 0``,

but by master-slave elimination instead of Lagrange multipliers: the slave DOF of each guided node (the
component with the larger normal part, ``u_z`` for angles below 45 degrees) follows ``u_x`` of the same node,
and every ``u_y`` is a slave that follows ``u_x`` with weight zero, i.e. it is zero. The model then has no extra scalar variables and no
zero diagonal, which iterative solvers (``blockamg``) cannot handle, and the solution is the one of the
multiplier form.

Both forms classify the node sets identically (:func:`~edelweissfe.constraints.inclinedroller.classifyRollerNodeSets`).
A slave DOF that is also prescribed by a boundary condition keeps the boundary condition; the record is then
redundant when the prescribed values already satisfy the roller equation, as they do for the load nodes of the
tilted shear model.
"""

import math

import numpy as np

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.constraints.base.multipointconstraintbase import (
    MultiPointConstraintBase,
)
from edelweissfe.constraints.inclinedroller import (
    InclinedRollerSchema,
    classifyRollerNodeSets,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.schema import buildSchemaFromOptions


class Constraint(MultiPointConstraintBase):
    """Guide a boundary so it can only move parallel to an inclined interface, by DOF elimination."""

    #: Same options as the multiplier form.
    schema = InclinedRollerSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        *,
        configuration: InclinedRollerSchema = InclinedRollerSchema(),
    ):
        self.name = name
        self._field = configuration.field
        self._model = model
        self._activeFromTime = configuration.activeFromTime
        if getFieldSize(self._field, model.domainSize) != 3:
            raise ValueError("InclinedRollerMPC requires a three-component vector field.")

        angle = math.radians(configuration.angle)
        self._normal_x = math.sin(angle)
        self._normal_z = math.cos(angle)

        cosTolerance = math.cos(math.radians(float(configuration.faceToleranceDegrees)))
        nodes_by_label, guided_labels, report = classifyRollerNodeSets(
            model, configuration.nSets, bool(configuration.legacy), cosTolerance
        )

        self._nodes = [nodes_by_label[label] for label in sorted(nodes_by_label)]
        self._guided = {label for label in guided_labels}
        self._records = [(node,) for node in self._nodes]

        print(
            f"InclinedRollerMPC '{name}': {len(self._nodes)} nodes, {len(self._guided)} guided by n.u = 0, "
            f"{len(self._nodes) - len(self._guided)} on Y-symmetry planes\n" + "\n".join(report)
        )

    @property
    def nodes(self):
        """The nodes of the roller, ordered by label."""
        return self._nodes

    @classmethod
    def fromConstraintDefinition(
        cls, name: str, definition: dict, model: FEModel, journal: "Journal" = None
    ) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(name, model, configuration=configuration)

    def getMultiPointConstraints(self, dofManager) -> list[tuple[int, list[tuple[int, float]]]]:
        # the condensation acts on the increment and is rebuilt at the start of every step, so a roller that
        # starts in a later step guides the motion from the state reached at that time
        if self._activeFromTime is not None and self._model.time < self._activeFromTime - 1e-12:
            return []
        fieldVariableIndices = dofManager.idcsOfFieldVariablesInDofVector

        # eliminate the component with the larger normal part: n.u = 0 -> u_s = -(n_m / n_s) u_m
        eliminateZ = abs(self._normal_z) >= abs(self._normal_x)

        records = []
        for node in self._nodes:
            if self._field not in node.fields or node.fields[self._field] not in fieldVariableIndices:
                raise KeyError(f"Constraint '{self.name}': node {node.label} has no active '{self._field}' DOFs.")
            ux, uy, uz = fieldVariableIndices[node.fields[self._field]]

            # u_y = 0 is a slave without a physical master; the condensation wants one, so it is u_x with weight 0
            records.append((int(uy), [(int(ux), 0.0)]))
            if node.label in self._guided:
                if eliminateZ:
                    records.append((int(uz), [(int(ux), -self._normal_x / self._normal_z)]))
                else:
                    records.append((int(ux), [(int(uz), -self._normal_z / self._normal_x)]))
        return records
