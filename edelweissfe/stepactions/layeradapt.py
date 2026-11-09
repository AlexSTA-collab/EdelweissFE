# -*- coding: utf-8 -*-
"""
Run-time p-adaptivity of the layerwise interface element LWIQUAD4_ADAPT (Marmot, bending-v26.11).

At the start of every increment, from the converged state of the previous one (reviewer's 24th check, 2026-11-09):

* the element's normal-part indicator ``eta_n = sqrt(E_n(last pair) / E_n)`` (elastic energy of the normal part
  (G n) (x) n of the gradient of its highest active pair of modes, relative to that of the whole gradient) decides its
  requested p: ``p + 2`` if ``eta_n > tolerance`` (two, so that both parities are added: the pair indicator cannot tell
  which one is missing), unless the element is *stalled*: two successive raises each reduced eta_n by less than a factor
  0.5 (an algebraic decay at a plastic front, where raising p buys nothing); ``p - 1`` only if the highest mode alone
  carries ``eta_1 < coarsen * tolerance`` of the normal energy, the element never yielded and eta_n did not grow (a dead
  band against oscillating between parities); else ``p``;
* the in-plane indicator ``eta_s`` is only reported (a request for in-plane refinement, which p cannot provide);
* the p of a bottom node is the maximum over the elements touching it (a conforming space: every element sharing the
  node uses the same set of modes there);
* the node values are written into the converged state ``adaptControl`` of every quadrature point of every element.

The through-thickness quadrature is not set here: each element chooses its own rule (Gauss 3/5/7/9/13, the smallest with
n >= p + 2, locked at 13 by its yield predictor), because the rule is local and needs no agreement between neighbours.
The initial p is the element's (``LWI_P0``, default 2). Every p >= 1 is allowed (the Legendre family is hierarchical).
"""

import os
from collections import Counter
from dataclasses import dataclass

import numpy as np

from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.misc import withoutParserBookkeepingKeys
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField


@dataclass(frozen=True)
class LayerAdaptSchema:
    """Options of the ``layeradapt`` keyword."""

    name: str | None = schemaField(
        description="Name of the step action.", dtype=str, default=None, required=True, structuralOnly=True
    )
    elSet: str | None = schemaField(
        description="The LWIQUAD4_ADAPT elements.", dtype=str, default="all", structuralOnly=True
    )
    tolerance: float = schemaField(
        description="Refine where the last pair of modes carries more than this share of the gradient (energy norm).",
        dtype=float,
        default=1e-2,
    )
    coarsen: float = schemaField(
        description="Coarsen (elastic elements only) when the highest mode alone is below coarsen * tolerance.",
        dtype=float,
        default=0.1,
    )
    pMin: int = schemaField(description="Smallest p.", dtype=int, default=1)
    pMax: int = schemaField(description="Largest p (at most 11).", dtype=int, default=11)


class StepAction(StepActionBase):
    """Sets the nodal p of the LWIQUAD4_ADAPT elements from their converged indicators."""

    schema = LayerAdaptSchema
    checkpointedState = {}

    N_QP = 4
    N_CONTROL = 6
    N_STATE = 10

    def __init__(self, name, elementSet, tolerance, coarsen, pMin, pMax, journal):
        self.name = name
        self.elements = list(elementSet)
        self.tolerance, self.coarsen = tolerance, coarsen
        self.pMin, self.pMax = max(1, pMin), min(11, pMax)
        self.journal = journal
        self.pInit = min(11, max(1, int(os.environ.get("LWI_P0", "2"))))
        self.prevEta = {}
        self.raiseLog = {}  # element -> [(p, eta_n at the raise)]
        self.stalled = set()
        self._layout = None

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        elSetName = definition.pop("elSet", "all")
        opts = buildSchemaFromOptions(cls.schema, definition)
        return cls(
            name, model.elementSets[elSetName], opts.tolerance, opts.coarsen, opts.pMin, opts.pMax, journal
        )

    def _offsets(self, el):
        if self._layout is None:
            cOff, _ = el.getStateVarSlice("adaptControl")
            sOff, _ = el.getStateVarSlice("adaptState")
            nPerQp = len(el.getStateVars()) // self.N_QP
            self._layout = (cOff, sOff, nPerQp)
        return self._layout

    def applyAtIncrementStart(self, model, timeStep):
        if not self.elements:
            return
        cOff, sOff, nPerQp = self._offsets(self.elements[0])

        states, wanted, nodeP = {}, {}, {}
        for el in self.elements:
            v = el.getStateVars()
            st = v[sOff : sOff + self.N_STATE]
            pe = int(round(st[5])) if st[5] > 0 else self.pInit
            eta, yielded, energy, eta1 = st[4], st[1] > 0.5, st[6], st[9]
            want, lab = pe, el.elNumber
            if energy > 0.0:
                prev = self.prevEta.get(lab, np.inf)
                log = self.raiseLog.setdefault(lab, [])
                # stalled: the last two raises each left eta_n above half of its value before them
                if len(log) >= 2 and log[-1][0] == pe and lab not in self.stalled:
                    e0, e1 = log[-2][1], log[-1][1]
                    if e1 > 0.5 * e0 and eta > 0.5 * e1:
                        self.stalled.add(lab)
                if eta > self.tolerance and pe < self.pMax and lab not in self.stalled:
                    want = min(self.pMax, pe + 2)
                    log.append((want, eta))
                elif eta1 < self.coarsen * self.tolerance and not yielded and pe > self.pMin and eta <= prev:
                    want = pe - 1
                self.prevEta[lab] = eta
            states[el.elNumber] = (el, v, st.copy())
            wanted[el.elNumber] = want
            for node in el.nodes[:4]:
                nodeP[node.label] = max(nodeP.get(node.label, self.pMin), want)

        changed = 0
        for label, (el, v, st) in states.items():
            ctrl = np.array([0.0] + [float(nodeP[n.label]) for n in el.nodes[:4]] + [0.0])
            old = v[cOff : cOff + self.N_CONTROL]
            if not np.array_equal(old, ctrl):
                for q in range(self.N_QP):
                    v[q * nPerQp + cOff : q * nPerQp + cOff + self.N_CONTROL] = ctrl
                el.setStateVars(v)
                changed += 1

        pHist = Counter(nodeP.values())
        levels = Counter(int(round(st[0])) - 1 for (_, _, st) in states.values())
        nYield = sum(1 for (_, _, st) in states.values() if st[1] > 0.5)
        etaS = np.array([st[8] for (_, _, st) in states.values()])
        n = [3, 5, 7, 9, 13]
        self.journal.message(
            "node p: "
            + " ".join(f"{p}:{pHist[p]}" for p in sorted(pHist))
            + " | element rule: "
            + " ".join(f"G{n[L]}:{levels[L]}" for L in sorted(levels) if L >= 0)
            + f" | yielded {nYield} | stalled {len(self.stalled)}"
            + f" | eta_s > 0.1 in {int((etaS > 0.1).sum())} (max {etaS.max() if len(etaS) else 0:.2g})"
            + f" | control changed in {changed} elements",
            self.name,
        )
