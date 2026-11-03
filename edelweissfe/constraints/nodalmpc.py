#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
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

"""Node-wise linear constraint between two components of one field, by master-slave elimination.

For every node of the given node sets: ``q_slave = factor * q_master`` (components numbered from 1). With
``factor = 0`` the slave is simply zero (the master only satisfies the condensation, which wants one).

Used for the zero-squeeze-flux condition of the three-station interface element at a boundary of the layer
whose material cannot flow out, ``g . nu = 0`` with ``nu`` the in-plane normal of the boundary: on the X-edges
of a layer tilted by alpha about Y, ``nu = t1 = (cos alpha, 0, -sin alpha)`` gives ``g_x = tan(alpha) g_z``;
on the Y-edges ``g_y = 0``.
"""

from dataclasses import dataclass

from edelweissfe.constraints.base.multipointconstraintbase import (
    MultiPointConstraintBase,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField


@dataclass(frozen=True)
class NodalMPCSchema:
    """The options of the constraint."""

    field: str | None = schemaField(description="Field whose components are tied.", dtype=str, default=None, required=True)
    nSets: str | None = schemaField(
        description="Semicolon-separated node sets.", dtype=str, default=None, required=True
    )
    slave: int | None = schemaField(description="Slave component (1-based).", dtype=int, default=None, required=True)
    master: int | None = schemaField(description="Master component (1-based).", dtype=int, default=None, required=True)
    factor: float = schemaField(description="q_slave = factor * q_master.", dtype=float, default=0.0)
    activeFromTime: float | None = schemaField(
        description="Model time from which the constraint acts (it acts from the start if omitted).", dtype=float, default=None
    )


class Constraint(MultiPointConstraintBase):
    """q_slave = factor * q_master on every node of the node sets."""

    schema = NodalMPCSchema

    def __init__(self, name: str, model: FEModel, *, configuration: NodalMPCSchema = NodalMPCSchema()):
        self.name = name
        self._model = model
        self._activeFromTime = configuration.activeFromTime
        self._field = configuration.field
        self._slave, self._master, self._factor = configuration.slave - 1, configuration.master - 1, configuration.factor
        if self._slave == self._master:
            raise ValueError(f"NodalMPC \x27{name}\x27: slave and master are the same component.")
        nodes = {}
        for setName in [s.strip() for s in configuration.nSets.split(";") if s.strip()]:
            if setName not in model.nodeSets:
                raise KeyError(f"NodalMPC \x27{name}\x27: node set \x27{setName}\x27 not found.")
            for node in model.nodeSets[setName]:
                nodes[node.label] = node
        self._nodes = [nodes[k] for k in sorted(nodes)]
        print(f"NodalMPC \x27{name}\x27: {len(self._nodes)} nodes, {self._field}[{self._slave + 1}] = {self._factor} * {self._field}[{self._master + 1}]")

    @property
    def nodes(self):
        return self._nodes

    @classmethod
    def fromConstraintDefinition(cls, name: str, definition: dict, model: FEModel, journal: "Journal" = None) -> "Constraint":
        return cls(name, model, configuration=buildSchemaFromOptions(cls.schema, definition))

    def getMultiPointConstraints(self, dofManager) -> list[tuple[int, list[tuple[int, float]]]]:
        # like the inclined roller: the condensation acts on the increment, so a late start guides the motion from then on
        if self._activeFromTime is not None and self._model.time < self._activeFromTime - 1e-12:
            return []
        idx = dofManager.idcsOfFieldVariablesInDofVector
        records = []
        for node in self._nodes:
            if self._field not in node.fields or node.fields[self._field] not in idx:
                raise KeyError(f"Constraint \x27{self.name}\x27: node {node.label} has no active \x27{self._field}\x27 DOFs.")
            d = idx[node.fields[self._field]]
            records.append((int(d[self._slave]), [(int(d[self._master]), float(self._factor))]))
        return records
