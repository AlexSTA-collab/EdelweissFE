#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression test for the ``inclinedroller`` Lagrange-multiplier constraint (ported from the
EdelweissFE 26.05 branch): a rollered face keeps ``n.u = 0`` and ``u_y = 0``, a Y-normal
(symmetry) face keeps only ``u_y = 0``."""

import math

import numpy as np
import pytest

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.config import registry
from edelweissfe.constraints.inclinedroller import Constraint
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.inputfileparser import parseInputFile

ANGLE = 10.0


def _square(label0, plane):
    """Four corners of a unit square in the plane x=0 ('X-') or y=0 ('Y-')."""
    pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
    if plane == "X-":
        xyz = [(0.0, a, b) for a, b in pts]
    else:
        xyz = [(a, 0.0, b) for a, b in pts]
    return [Node(label0 + i, np.array(c, dtype=float)) for i, c in enumerate(xyz)]


def _buildModel():
    model = FEModel(3)
    xm = _square(1, "X-")
    ym = _square(11, "Y-")
    model.nodeSets["bottom-X-"] = NodeSet("bottom-X-", xm)
    model.nodeSets["bottom-Y-"] = NodeSet("bottom-Y-", ym)
    return model, xm, ym


def _solveProjection(constraint, target):
    """Closest displacement to ``target`` (identity 'stiffness') subject to the constraint."""
    n = constraint.nDof
    nu = 3 * len(constraint.nodes)
    K = np.zeros((n, n))
    P = np.zeros(n)
    constraint.applyConstraint(np.zeros(n), np.zeros(n), P, K, None)
    A = K.copy()
    A[np.arange(nu), np.arange(nu)] += 1.0
    b = np.zeros(n)
    b[:nu] = target
    return np.linalg.solve(A, b)[:nu].reshape(-1, 3)


def _constraint(model, **extra):
    opts = {"field": "displacement", "nSets": "bottom-X-;bottom-Y-", "angle": str(ANGLE)}
    opts.update(extra)
    return Constraint.fromConstraintDefinition("lateral-roller", opts, model)


def test_rolled_face_keeps_normal_and_uy_zero_symmetry_face_only_uy():
    model, xm, ym = _buildModel()
    c = _constraint(model)
    rng = np.random.default_rng(0)
    target = rng.normal(size=3 * len(c.nodes))
    u = _solveProjection(c, target)

    th = math.radians(ANGLE)
    labels = [nd.label for nd in c.nodes]
    for nd in xm:  # guided face
        ux, uy, uz = u[labels.index(nd.label)]
        assert math.sin(th) * ux + math.cos(th) * uz == pytest.approx(0.0, abs=1e-12)
        assert uy == pytest.approx(0.0, abs=1e-12)
    t = np.array([math.cos(th), 0.0, -math.sin(th)])
    t_target = target.reshape(-1, 3)
    for nd in ym:  # Y-symmetry face: only u_y = 0, ux and uz untouched
        i = labels.index(nd.label)
        assert u[i, 1] == pytest.approx(0.0, abs=1e-12)
        assert u[i, 0] == pytest.approx(t_target[i, 0], abs=1e-12)
        assert u[i, 2] == pytest.approx(t_target[i, 2], abs=1e-12)
        assert abs(math.sin(th) * u[i, 0] + math.cos(th) * u[i, 2]) > 1e-6  # n.u NOT enforced
    # the guided face keeps the tangential component of the target
    i = labels.index(xm[0].label)
    assert u[i] @ t == pytest.approx(t_target[i] @ t, abs=1e-12)


def test_legacy_pins_normal_on_every_face():
    model, xm, ym = _buildModel()
    c = _constraint(model, legacy="True")
    u = _solveProjection(c, np.random.default_rng(1).normal(size=3 * len(c.nodes)))
    th = math.radians(ANGLE)
    assert np.allclose(math.sin(th) * u[:, 0] + math.cos(th) * u[:, 2], 0.0, atol=1e-12)
    assert np.allclose(u[:, 1], 0.0, atol=1e-12)


def test_registered_and_keyword_line_spelling(tmp_path):
    cls, _ = registry.lookup("constraint", "inclinedroller")
    assert cls is Constraint
    inp = tmp_path / "job.inp"
    inp.write_text(
        "*constraint, type=inclinedroller, name=lateral-roller\n"
        "nSets=bottom-X-;bottom-Y-\n"
        "field=displacement\n"
        "angle=10\n"
    )
    parsed = parseInputFile(str(inp))
    assert len(parsed["constraint"]) == 1
