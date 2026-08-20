"""Lagrange-multiplier constraint for an interface-aligned roller."""

import math

import numpy as np

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import caseInsensitiveKwargsChecker, castKwargsValuesAndAddDefaults


module = Module("inclinedroller", "A Lagrange-multiplier roller aligned with an inclined interface.")
inputLanguage = InputLanguage()
if "constraint" in inputLanguage:
    inputLanguage["constraint"].addModule(module)

module.addRequiredArg("field", "Vector field constrained by the roller.", str)
module.addRequiredArg("nSets", "Semicolon-separated node sets forming the guided boundary.", str)
module.addRequiredArg("angle", "Interface angle in degrees, measured by rotating the normal about Y.", float)
module.addOptionalArg(
    "legacy",
    "Apply the layer-normal pin to EVERY node set, including Y-normal faces. "
    "Reproduces the pre-fix behaviour; see the class docstring for why that is wrong.",
    bool,
    False,
)
module.addOptionalArg(
    "faceToleranceDegrees",
    "Angular tolerance for classifying a node set's plane as Y-normal.",
    float,
    15.0,
)

documentation = [module]


class Constraint(ConstraintBase):
    """Guide a boundary so it can only move parallel to an inclined interface.

    Each node set is classified by the plane its nodes lie in, and receives the
    constraint appropriate to THAT face:

    - **In-plane-normal faces** (face normal lying in the X-Z plane, e.g. X+/X-):
      the node is guided onto the interface tangent,

          n . u = sin(theta) * u_x + cos(theta) * u_z = 0
          u_y = 0

      leaving one free direction, t = (cos(theta), 0, -sin(theta)).

    - **Y-normal faces** (face normal ~ +/- e_y, e.g. Y+/Y-): a symmetry plane,

          u_y = 0

      only. u_x and u_z stay free.

    Why the distinction matters
    ---------------------------
    The previous implementation applied BOTH equations to every listed node set.
    On a Y-normal face the ``n . u = 0`` equation is spurious: it pins the two
    lateral strips against any motion normal to the interface, while the rest of
    the model is free to move that way. For a near-incompressible (e.g. von
    Mises, plastically isochoric) interface layer that acts as a volumetric
    constraint imposed on a strip of nodes, and it injects a spurious pressure
    disturbance along the lateral edges. Measured on the 30x30 tilted shear
    model: an edge pressure deviation of +0.33 (resolved Cauchy reference) to
    +0.44 (interface element) against an interior level of ~0.3, decaying over
    roughly six elements, with an element-to-element alternation riding on top
    of it in the interface elements.

    Classification is geometric, so existing input files are corrected without
    edits. Pass ``legacy=True`` to restore the old behaviour exactly.
    """

    @caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
    @castKwargsValuesAndAddDefaults(module)
    def __init__(self, name, model, *args, **kwargs):
        super().__init__(name, model, *args, **kwargs)
        definition = CaseInsensitiveDict(kwargs)

        self._name = name
        self._field = definition["field"]
        self._field_size = getFieldSize(self._field, model.domainSize)
        if self._field_size != 3:
            raise ValueError("InclinedRoller requires a three-component vector field.")

        angle = math.radians(definition["angle"])
        self._normal_x = math.sin(angle)
        self._normal_z = math.cos(angle)

        legacy = bool(definition["legacy"])
        cosTolerance = math.cos(math.radians(float(definition["faceToleranceDegrees"])))

        node_set_names = [name.strip() for name in definition["nSets"].split(";") if name.strip()]
        if not node_set_names:
            raise ValueError("InclinedRoller requires at least one node set in 'nSets'.")

        nodes_by_label = {}
        guided_labels = set()
        report = []
        for node_set_name in node_set_names:
            if node_set_name not in model.nodeSets:
                raise KeyError(f"Unknown node set '{node_set_name}' in InclinedRoller constraint.")
            set_nodes = list(model.nodeSets[node_set_name])
            for node in set_nodes:
                nodes_by_label[node.label] = node

            if legacy:
                kind = "guided (legacy)"
            else:
                face_normal = self._fitPlaneNormal(set_nodes)
                if face_normal is None:
                    # too few nodes, or not planar enough to classify -> stay on the safe side
                    kind = "guided (plane fit inconclusive)"
                elif abs(face_normal[1]) >= cosTolerance:
                    kind = "Y-symmetry (u_y = 0 only)"
                else:
                    kind = "guided (n.u = 0, u_y = 0)"
            if not kind.startswith("Y-symmetry"):
                guided_labels.update(node.label for node in set_nodes)
            report.append(f"    {node_set_name:<26} -> {kind}")

        self._nodes = [nodes_by_label[label] for label in sorted(nodes_by_label)]
        self._n_nodes = len(self._nodes)
        self._fields_on_nodes = [[self._field] for _ in self._nodes]

        # A node shared by a guided face and a Y-symmetry face (a corner) keeps the
        # stronger, guided treatment.
        guided_mask = np.array([node.label in guided_labels for node in self._nodes], dtype=bool)
        self._guided = np.flatnonzero(guided_mask)
        n_guided = int(self._guided.size)

        nodal_dofs = self._field_size * self._n_nodes
        self._ux = np.arange(0, nodal_dofs, self._field_size)
        self._uy = self._ux + 1
        self._uz = self._ux + 2
        self._ux_guided = self._ux[self._guided]
        self._uz_guided = self._uz[self._guided]
        # one normal multiplier per guided node, one u_y multiplier per node
        self._lambda_normal = np.arange(nodal_dofs, nodal_dofs + n_guided)
        self._lambda_y = np.arange(nodal_dofs + n_guided, nodal_dofs + n_guided + self._n_nodes)
        self._n_dof = nodal_dofs + n_guided + self._n_nodes
        self.active = True

        print(
            f"InclinedRoller '{name}': {self._n_nodes} nodes, {n_guided} guided by n.u = 0, "
            f"{self._n_nodes - n_guided} on Y-symmetry planes\n" + "\n".join(report)
        )

    @staticmethod
    def _fitPlaneNormal(nodes, planarityTolerance=1.0e-3):
        """Least-variance direction of the node cloud, i.e. the face normal.

        Returns None when the nodes do not define a plane well enough to classify
        (fewer than three nodes, or a cloud that is not sufficiently flat).
        """
        if len(nodes) < 3:
            return None
        coordinates = np.array([node.coordinates for node in nodes], dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            return None
        centred = coordinates - coordinates.mean(axis=0)
        singularValues = np.linalg.svd(centred, compute_uv=False)
        if singularValues[0] <= 0.0:
            return None
        # a planar node set has a vanishing third singular value relative to the first
        if singularValues[2] / singularValues[0] > planarityTolerance:
            return None
        _, _, rightVectors = np.linalg.svd(centred, full_matrices=True)
        return rightVectors[2]

    @property
    def nodes(self):
        return self._nodes

    @property
    def fieldsOnNodes(self):
        return self._fields_on_nodes

    @property
    def nDof(self):
        return self._n_dof

    def getNumberOfAdditionalNeededScalarVariables(self):
        return int(self._guided.size) + self._n_nodes

    def applyConstraint(self, U_np, dU, PExt, K, timeStep):
        if not self.active:
            return

        lambda_normal = U_np[self._lambda_normal]
        lambda_y = U_np[self._lambda_y]
        normal_displacement = self._normal_x * U_np[self._ux_guided] + self._normal_z * U_np[self._uz_guided]
        y_displacement = U_np[self._uy]

        PExt[self._ux_guided] -= self._normal_x * lambda_normal
        PExt[self._uz_guided] -= self._normal_z * lambda_normal
        PExt[self._uy] -= lambda_y
        PExt[self._lambda_normal] -= normal_displacement
        PExt[self._lambda_y] -= y_displacement

        K[self._ux_guided, self._lambda_normal] += self._normal_x
        K[self._uz_guided, self._lambda_normal] += self._normal_z
        K[self._uy, self._lambda_y] += 1.0
        K[self._lambda_normal, self._ux_guided] += self._normal_x
        K[self._lambda_normal, self._uz_guided] += self._normal_z
        K[self._lambda_y, self._uy] += 1.0
