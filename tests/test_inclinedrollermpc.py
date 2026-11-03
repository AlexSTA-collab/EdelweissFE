#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression test for the ``inclinedrollermpc`` constraint: the elimination form delivers the same
projection as the Lagrange-multiplier roller, with no scalar variables and no zero diagonal."""

import math

import numpy as np
import pytest

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.config import registry
from edelweissfe.constraints.inclinedroller import Constraint as LagrangeRoller
from edelweissfe.constraints.inclinedrollermpc import Constraint
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet

ANGLE = 10.0


def _square(label0, plane):
    pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
    xyz = [(0.0, a, b) for a, b in pts] if plane == "X-" else [(a, 0.0, b) for a, b in pts]
    return [Node(label0 + i, np.array(c, dtype=float)) for i, c in enumerate(xyz)]


def _buildModel():
    model = FEModel(3)
    xm, ym = _square(1, "X-"), _square(11, "Y-")
    model.nodeSets["bottom-X-"] = NodeSet("bottom-X-", xm)
    model.nodeSets["bottom-Y-"] = NodeSet("bottom-Y-", ym)
    return model, xm, ym


def _options(**extra):
    opts = {"field": "displacement", "nSets": "bottom-X-;bottom-Y-", "angle": str(ANGLE)}
    opts.update(extra)
    return opts


class _DofManager:
    """Three consecutive DOFs per node, in the order of the constraint's nodes."""

    def __init__(self, nodes):
        self.idcsOfFieldVariablesInDofVector = {}
        for i, node in enumerate(nodes):
            node.fields = {"displacement": ("u", node.label)}
            self.idcsOfFieldVariablesInDofVector[("u", node.label)] = np.array([3 * i, 3 * i + 1, 3 * i + 2])


def _lagrangeProjection(model, target):
    c = LagrangeRoller.fromConstraintDefinition("lateral-roller", _options(), model)
    n, nu = c.nDof, 3 * len(c.nodes)
    K, P = np.zeros((n, n)), np.zeros(n)
    c.applyConstraint(np.zeros(n), np.zeros(n), P, K, None)
    A = K.copy()
    A[np.arange(nu), np.arange(nu)] += 1.0
    b = np.zeros(n)
    b[:nu] = target
    return np.linalg.solve(A, b)[:nu], [nd.label for nd in c.nodes]


def _eliminationProjection(model, target):
    c = Constraint.fromConstraintDefinition("lateral-roller", _options(), model)
    records = c.getMultiPointConstraints(_DofManager(c.nodes))
    n = 3 * len(c.nodes)
    T = np.eye(n)
    for slave, masters in records:
        T[slave, :] = 0.0
        for master, weight in masters:
            T[slave, master] += weight
    # closest point to target in the range of T: minimise |T w - target| over the free DOFs
    free = [i for i in range(n) if i not in {s for s, _ in records}]
    w = np.zeros(n)
    w[free] = np.linalg.lstsq(T[:, free], target, rcond=None)[0]
    return T @ w, records, [nd.label for nd in c.nodes]


def test_elimination_gives_the_projection_of_the_multiplier_form():
    model, _, _ = _buildModel()
    target = np.random.default_rng(0).normal(size=3 * 8)
    uL, labelsL = _lagrangeProjection(model, target)
    uE, _, labelsE = _eliminationProjection(model, target)
    assert labelsL == labelsE
    assert np.allclose(uL, uE, atol=1e-12)


def test_records_of_a_guided_and_a_symmetry_node():
    model, xm, ym = _buildModel()
    _, records, labels = _eliminationProjection(model, np.zeros(3 * 8))
    byslave = dict(records)
    i = labels.index(xm[0].label)
    assert byslave[3 * i + 1] == [(3 * i, 0.0)]
    (master, weight), = byslave[3 * i + 2]
    assert master == 3 * i
    assert weight == pytest.approx(-math.tan(math.radians(ANGLE)))
    j = labels.index(ym[0].label)
    assert byslave[3 * j + 1] == [(3 * j, 0.0)]
    assert 3 * j not in byslave and 3 * j + 2 not in byslave  # a symmetry node keeps u_x, u_z free


def test_no_scalar_variables_and_registered_as_a_multi_point_constraint():
    from edelweissfe.constraints.base.multipointconstraintbase import MultiPointConstraintBase

    cls, _ = registry.lookup("constraint", "inclinedrollermpc")
    assert cls is Constraint and issubclass(cls, MultiPointConstraintBase)
    assert not hasattr(Constraint, "getNumberOfAdditionalNeededScalarVariables")


def test_a_roller_active_from_a_later_time_has_no_records_before_it():
    model, _, _ = _buildModel()
    c = Constraint.fromConstraintDefinition("lateral-roller", _options(activeFromTime="1.0"), model)
    dofs = _DofManager(c.nodes)
    model.time = 0.0
    assert c.getMultiPointConstraints(dofs) == []
    model.time = 1.0  # the start of the second step: the roller guides the increments from here on
    always = Constraint.fromConstraintDefinition("lateral-roller", _options(), model)
    assert c.getMultiPointConstraints(dofs) == always.getMultiPointConstraints(dofs)
