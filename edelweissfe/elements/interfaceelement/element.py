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
#  Alexandros Stathas alexandros.stathas@boku.ac.at
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

import copy
from pathlib import Path

import basix
import numpy as np
import numpy.linalg as lin

from edelweissfe.elements.base.baseelement import BaseElement
from edelweissfe.elements.interfaceelement.elementmatrices import (
    assign_K_grad_s_u_grad_s_v,
    assign_K_grad_s_u_jump_v,
    assign_K_jump_u_grad_s_v,
    assign_K_jumpu_jumpv,
    assign_P_grad_s_v,
    assign_P_jumpv,
    calculate_B_surface_grad,
    calculate_N_jump,
    compute_grad,
    compute_surface_div_grad,
    compute_surface_grad,
    computeJacobian,
    computeNOperator,
    interface_geometry,
)
from edelweissfe.points.node import Node
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict


LOCAL_STIFFNESS_EXPORT_MODE = "implicit"  # "fd", "implicit", or "none"
LOCAL_STIFFNESS_EXPORT_DIR = Path("stiffness_exports")
LOCAL_STIFFNESS_EXPORT_BASENAME = "interface_check"
LOCAL_FD_DEBUG_LOG = True

LOCAL_OPERATOR_DEBUG = True
LOCAL_STIFFNESS_BLOCK_DEBUG = True


elLibrary = CaseInsensitiveDict(
    ILine2=dict(
        nNodes=4,
        nDof=8,
        dofIndices=np.arange(0, 8),
        ensightType="line2",
        nSpatialDimensions=2,
        nInt=2,
        element=basix.create_element(basix.ElementFamily.P, basix.CellType.interval, 1),
        qpoints=basix.make_quadrature(basix.CellType.interval, 2)[0],
        w=basix.make_quadrature(basix.CellType.interval, 2)[1],
        matSize=3,
        index=np.array([0, 1, 3]),
        plStrain=True,
        reorder_nodes_list=[0, 1, 2, 3],
        hasMaterial=False,
    ),
    ILine2R=dict(
        nNodes=4,
        nDof=8,
        dofIndices=np.arange(0, 8),
        ensightType="line2",
        nSpatialDimensions=2,
        nInt=1,
        element=basix.create_element(basix.ElementFamily.P, basix.CellType.interval, 1),
        qpoints=basix.make_quadrature(basix.CellType.interval, 1)[0],
        w=basix.make_quadrature(basix.CellType.interval, 1)[1],
        matSize=3,
        index=np.array([0, 1, 3]),
        plStrain=True,
        reorder_nodes_list=[0, 1, 2, 3],
        hasMaterial=False,
    ),
    ILine3=dict(
        nNodes=6,
        nDof=12,
        dofIndices=np.arange(0, 12),
        ensightType="line3",
        nSpatialDimensions=2,
        nInt=3,
        element=basix.create_element(
            basix.ElementFamily.P,
            basix.CellType.interval,
            2,
            basix.LagrangeVariant.equispaced,
        ),
        qpoints=basix.make_quadrature(basix.CellType.interval, 4)[0],
        w=basix.make_quadrature(basix.CellType.interval, 4)[1],
        matSize=3,
        index=np.array([0, 1, 3]),
        plStrain=True,
        reorder_nodes_list=[0, 1, 2, 3, 4, 5],
        hasMaterial=False,
    ),
    ILine3R=dict(
        nNodes=6,
        nDof=12,
        dofIndices=np.arange(0, 12),
        ensightType="line3",
        nSpatialDimensions=2,
        nInt=2,
        element=basix.create_element(basix.ElementFamily.P, basix.CellType.interval, 2),
        qpoints=basix.make_quadrature(basix.CellType.interval, 3)[0],
        w=basix.make_quadrature(basix.CellType.interval, 2)[1],
        matSize=3,
        index=np.array([0, 1, 3]),
        plStrain=True,
        reorder_nodes_list=[0, 1, 2, 3, 4, 5],
        hasMaterial=False,
    ),
    IQuad4=dict(
        nNodes=8,
        nDof=24,
        dofIndices=np.arange(0, 24),
        ensightType="hexa8",
        nSpatialDimensions=3,
        nInt=4,
        element=basix.create_element(basix.ElementFamily.P, basix.CellType.quadrilateral, 1),
        qpoints=basix.make_quadrature(basix.CellType.quadrilateral, 2)[0],
        w=basix.make_quadrature(basix.CellType.quadrilateral, 2)[1],
        matSize=3,
        index=np.array([0, 1, 3]),
        plStrain=True,
        reorder_nodes_list=[0, 1, 2, 3, 4, 5, 6, 7],
        hasMaterial=False,
    ),
    IQuad8=dict(
        nNodes=16,
        nDof=48,
        dofIndices=np.arange(0, 48),
        ensightType="quad8",
        nSpatialDimensions=3,
        nInt=9,
        element=basix.create_element(basix.ElementFamily.serendipity, basix.CellType.quadrilateral, 2),
        qpoints=basix.make_quadrature(basix.CellType.quadrilateral, 4)[0],
        w=basix.make_quadrature(basix.CellType.quadrilateral, 4)[1],
        matSize=3,
        index=np.array([0, 1, 3]),
        plStrain=True,
        reorder_nodes_list=list(range(16)),
        hasMaterial=False,
    ),
    IQuad8R=dict(
        nNodes=16,
        nDof=48,
        dofIndices=np.arange(0, 48),
        ensightType="quad8",
        nSpatialDimensions=3,
        nInt=8,
        element=basix.create_element(basix.ElementFamily.serendipity, basix.CellType.quadrilateral, 2),
        qpoints=np.delete(basix.make_quadrature(basix.CellType.quadrilateral, 4)[0], 4, axis=0),
        w=np.delete(basix.make_quadrature(basix.CellType.quadrilateral, 4)[1], 4),
        matSize=3,
        index=np.array([0, 1, 3]),
        plStrain=True,
        reorder_nodes_list=list(range(16)),
        hasMaterial=False,
    ),
    IQuad9R=dict(
        nNodes=18,
        nDof=54,
        dofIndices=np.arange(0, 54),
        ensightType="quad9",
        nSpatialDimensions=3,
        nInt=8,
        element=basix.create_element(basix.ElementFamily.P, basix.CellType.quadrilateral, 2),
        qpoints=np.delete(basix.make_quadrature(basix.CellType.quadrilateral, 4)[0], 4, axis=0),
        w=np.delete(basix.make_quadrature(basix.CellType.quadrilateral, 4)[1], 4),
        matSize=3,
        index=np.array([0, 1, 3]),
        plStrain=True,
        reorder_nodes_list=list(range(18)),
        hasMaterial=False,
    ),
)

elLibrary.update(
    {
        "ILine2": elLibrary["ILine2"],
        "ILine2R": elLibrary["ILine2R"],
        "ILine3": elLibrary["ILine3"],
        "ILine3R": elLibrary["ILine3R"],
        "IQuad4": elLibrary["IQuad4"],
        "IQuad8": elLibrary["IQuad8"],
        "IQuad8R": elLibrary["IQuad8R"],
        "IQuad9R": elLibrary["IQuad9R"],
    }
)


class InterfaceElement(BaseElement):
    @property
    def elNumber(self) -> int:
        return self._elNumber

    @property
    def nNodes(self) -> int:
        return self._nNodes

    @property
    def nodes(self) -> list[Node]:
        return self._nodes

    @property
    def nDof(self) -> int:
        return self._nDof

    @property
    def fields(self) -> list[list[str]]:
        return self._fields

    @property
    def dofIndicesPermutation(self) -> np.ndarray:
        return self._dofIndices

    @property
    def ensightType(self) -> str:
        return self._ensightType

    @property
    def visualizationNodes(self) -> list[Node]:
        return self._nodes

    @property
    def hasMaterial(self) -> str:
        return self._hasMaterial  # type: ignore[return-value]

    @property
    def reorder_nodes_list(self):
        return self._reorder_nodes_list

    def __init__(self, elementType: str, elNumber: int):
        self.elementtype = elementType[0].upper() + elementType[1:5].lower() + elementType[5:].upper()
        self._elNumber = elNumber

        try:
            if len(self.elementtype) > 5 and self.elementtype[5].lower() == "n":
                self.elementtype = self.elementtype.replace("N", "").replace("n", "")

            properties = elLibrary[self.elementtype]

        except KeyError:
            raise Exception("This element type doesn't exist.")

        self._nNodes = properties["nNodes"]
        self._nDof = properties["nDof"]
        self._dofIndices = properties["dofIndices"]
        self._ensightType = properties["ensightType"]
        self.nSpatialDimensions = properties["nSpatialDimensions"]
        self._nInt = properties["nInt"]

        self._element = properties["element"]
        self._qpoints = properties["qpoints"]
        self._weight = properties["w"]

        self._matrixSize = properties["matSize"]
        self._activeVoigtIndices = properties["index"]
        self.planeStrain = properties["plStrain"]
        self._hasMaterial = properties.get("hasMaterial", False)

        if self.nSpatialDimensions > 1:
            self._t = 1.0

        self._fields = [["displacement"] for _ in range(self._nNodes)]

        self._dStrain = np.zeros([self._nInt, 9])

        self.number_of_dofs = int(self._nDof * self.nSpatialDimensions)
        self.number_of_strain_comp = int(self._nDof * self.nSpatialDimensions * self.nSpatialDimensions)

        self._dU_GPs = np.zeros((self._nInt, self.nSpatialDimensions * 2))

        self._dSurface_strain_GPs = np.zeros(
            (self._nInt, self.nSpatialDimensions * self.nSpatialDimensions * 2)
        )

        self._reorder_nodes_list = properties.get("reorder_nodes_list", list(range(self._nNodes)))

        self._J_jumpv = np.zeros((self.nDof, self.nDof))
        self._J_grad_s_v = np.zeros((self.nDof, self.nDof))

        self.count = 0

        self._stiffness_export_mode = str(LOCAL_STIFFNESS_EXPORT_MODE).lower()
        self._stiffness_export_dir = Path(LOCAL_STIFFNESS_EXPORT_DIR).expanduser()
        self._stiffness_export_basename = str(LOCAL_STIFFNESS_EXPORT_BASENAME)
        self._fd_debug_log_enabled = bool(LOCAL_FD_DEBUG_LOG)

        self._has_exported_stiffness = False
        self._has_logged_fd_jump_column_debug = False
        self._has_reset_fd_debug_log = False

        self.inverse_order = np.empty_like(np.asarray(self.reorder_nodes_list))
        self.inverse_order[self.reorder_nodes_list] = np.arange(len(self.reorder_nodes_list))

    def setNodes(self, nodes: list[Node]):
        self._nodes = nodes

        _nodesCoordinates = np.array([n.coordinates for n in nodes])
        self._nodesCoordinates = (_nodesCoordinates.transpose())[:, self.reorder_nodes_list]

    def setProperties(self, elementProperties: np.ndarray):
        if self.elementtype[0] == "I":
            self._t = elementProperties[0]

    def initializeElement(self):
        self.basis_function = computeNOperator(
            self._nodesCoordinates,
            self._element,
            self._qpoints,
            self.nSpatialDimensions,
        )

        self.jacobians, self.gradients = computeJacobian(
            self._nodesCoordinates,
            self._element,
            self._qpoints,
            self.nSpatialDimensions,
        )

        self.grad, self.sqrt_detG = compute_grad(
            self._nodesCoordinates,
            self._element,
            self._qpoints,
            self._nInt,
            self.nSpatialDimensions,
        )

        self.surface_grad = compute_surface_grad(
            self._nodesCoordinates,
            self._element,
            self._qpoints,
            self._nInt,
            self.nSpatialDimensions,
        )

        self.surface_div_grad = compute_surface_div_grad(
            self._nodesCoordinates,
            self._element,
            self._qpoints,
            self._nInt,
            self.nSpatialDimensions,
        )

        self.n, self.N, self.T = interface_geometry(
            self._nodesCoordinates,
            self._element,
            self._qpoints,
            self._nInt,
            self.nSpatialDimensions,
        )

        self.N_matrix = calculate_N_jump(self.basis_function[:, :, :, 0])
        self.B_matrix = calculate_B_surface_grad(self.grad, self.surface_grad)

        if LOCAL_OPERATOR_DEBUG and self.elementtype.lower() == "iquad4":
            np.set_printoptions(precision=16, suppress=False, linewidth=240)

            print("\n===== PY Interface ALL GP geometry/operator debug =====")
            print("elementtype =", self.elementtype)
            print("nInt =", self._nInt)
            print("N_matrix shape =", np.asarray(self.N_matrix).shape)
            print("B_matrix shape =", np.asarray(self.B_matrix).shape)
            print("grad shape =", np.asarray(self.grad).shape)
            print("surface_grad shape =", np.asarray(self.surface_grad).shape)
            print("surface_div_grad shape =", np.asarray(self.surface_div_grad).shape)

            for i in range(self._nInt):
                xi_py = np.asarray(self._qpoints[i], dtype=float)
                xi_cxx = 2.0 * xi_py - 1.0

                print(f"\n--- PY GP {i} ---")
                print("qp.index =", i)
                print("qp.xi_python =", xi_py)
                print("qp.xi_cxx_equivalent =", xi_cxx)
                print("qp.weight_python =", self._weight[i])
                print("qp.weight_cxx_equivalent =", 4.0 * self._weight[i])
                print("qp.normal =", self.n[i])
                print("qp.normal.norm =", np.linalg.norm(self.n[i]))
                print("qp.sqrtDetG_python =", self.sqrt_detG[i])
                print("qp.J0xW =", self.sqrt_detG[i] * self._t * self._weight[i])

                print("\nN_matrix[i].shape =", self.N_matrix[i].shape)
                print("N_matrix[i] =")
                print(
                    np.array2string(
                        self.N_matrix[i],
                        precision=16,
                        suppress_small=False,
                        max_line_width=240,
                    )
                )

                print("\nB_matrix[i].shape =", self.B_matrix[i].shape)
                print("B_matrix[i] =")
                print(
                    np.array2string(
                        self.B_matrix[i],
                        precision=16,
                        suppress_small=False,
                        max_line_width=240,
                    )
                )

                surface_grad_arr = np.asarray(self.surface_grad)
                if surface_grad_arr.ndim == 4:
                    surface_grad_i = surface_grad_arr[i]
                elif surface_grad_arr.ndim == 5:
                    surface_grad_i = surface_grad_arr[:, :, :, i, 0]
                else:
                    surface_grad_i = surface_grad_arr

                print("\nsurface_grad at qp shape =", np.asarray(surface_grad_i).shape)
                print("surface_grad at qp =")
                print(
                    np.array2string(
                        surface_grad_i,
                        precision=16,
                        suppress_small=False,
                        max_line_width=240,
                    )
                )

            print("\n===== END PY Interface ALL GP geometry/operator debug =====")

    def setMaterial(self, material: type):
        self.material = material

        stateVarsSize = (3 + 9) + 2 * (3 + 9) + self.material.getNumberOfRequiredStateVars()
        self._matrixSize = 21

        self._Q_ij = np.zeros([self._nInt, 3, 3])
        self._Z_ijkl = np.zeros([self._nInt, 3, 3, 3, 3])
        self._H_ijk = np.zeros([self._nInt, 3, 3, 3])
        self._Y_ijkl = np.zeros([self._nInt, 3, 3, 3, 3])

        self._hasMaterial = True
        self._stateVarsRef = np.zeros([self._nInt, stateVarsSize])

        self._stateVars = [
            CaseInsensitiveDict(
                {
                    "force": self._stateVarsRef[i][0:3],
                    "surface stress": self._stateVarsRef[i][3:12],
                    "displacement": self._stateVarsRef[i][12:18],
                    "surface strain": self._stateVarsRef[i][18:36],
                    "materialstate": self._stateVarsRef[i][36:],
                }
            )
            for i in range(self._nInt)
        ]

        self._stateVarsTemp = np.zeros([self._nInt, stateVarsSize])

    def setInitialCondition(self, stateType: str, values: np.ndarray):
        raise Exception(
            "Setting an initial condition is not possible "
            "with this element provider."
        )

    def computeDistributedLoad(
        self,
        loadType: str,
        P: np.ndarray,
        K: np.ndarray,
        faceID: int,
        load: np.ndarray,
        U: np.ndarray,
        time: np.ndarray,
        dT: float,
    ):
        raise Exception(
            "Applying a distributed load is currently not possible "
            "with this element provider."
        )

    def computeYourself(
        self,
        K: np.ndarray,
        P: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: np.ndarray,
        dTime: float,
    ):
        dU = dU.reshape((self._nNodes, -1))

        self._stateVarsTemp = [self._stateVarsRef[i].copy() for i in range(self._nInt)].copy()

        self.number_of_element_nodes = int(self._nNodes / 2)
        self.number_of_top_dofs = int(self._nDof / 2)
        self.number_of_top_strain_comp = int(self.nSpatialDimensions * self.nSpatialDimensions)

        dU_GPs_bottom = np.einsum(
            "qcm,m->qc",
            self.N_matrix,
            dU[: self.number_of_element_nodes].flatten(),
        )
        dU_GPs_top = np.einsum(
            "qcm,m->qc",
            self.N_matrix,
            dU[self.number_of_element_nodes :].flatten(),
        )

        self._dU_GPs = np.ascontiguousarray(np.hstack((dU_GPs_top, dU_GPs_bottom)))

        dSurface_strain_GPs_bottom = np.einsum(
            "qcm,m->qc",
            self.B_matrix,
            dU[: self.number_of_element_nodes].flatten(),
        )
        dSurface_strain_GPs_top = np.einsum(
            "qcm,m->qc",
            self.B_matrix,
            dU[self.number_of_element_nodes :].flatten(),
        )

        self._dSurface_strain_GPs = np.ascontiguousarray(
            np.hstack((dSurface_strain_GPs_top, dSurface_strain_GPs_bottom))
        )

        K_before = K.copy()

        fd_mode = self._stiffness_export_mode == "fd"
        if fd_mode:
            self._J_jumpv.fill(0.0)
            self._J_grad_s_v.fill(0.0)

        for i in range(self._nInt):
            self._force_at_Gauss = self._stateVarsTemp[i][0 : self.nSpatialDimensions]
            force_state_old = np.array(
                self._stateVarsTemp[i][0 : self.nSpatialDimensions],
                dtype=np.float64,
            )
            self._force_at_Gauss_X = force_state_old.copy()

            self._surface_stress_at_Gauss = self._stateVarsTemp[
                i
            ][3 : int(3 + self.nSpatialDimensions**2)].reshape(
                (self.nSpatialDimensions, self.nSpatialDimensions)
            )

            surface_stress_state_old = np.array(
                self._stateVarsTemp[i][3 : int(3 + self.nSpatialDimensions**2)].reshape(
                    (self.nSpatialDimensions, self.nSpatialDimensions)
                ),
                dtype=np.float64,
            )
            self._surface_stress_at_Gauss_X = surface_stress_state_old.copy()

            self.material.assignStateVars(self._stateVarsTemp[i][36:])

            if not self.planeStrain and self.nSpatialDimensions == 2:
                raise Exception("Plain stress is not yet implemented in this element provider.")

            detJ = self.sqrt_detG[i]

            self.material.computeStress(
                self._force_at_Gauss,
                self._surface_stress_at_Gauss,
                self._Q_ij[i],
                self._Z_ijkl[i],
                self._H_ijk[i],
                self._Y_ijkl[i],
                self._dU_GPs[i],
                self._dSurface_strain_GPs[i],
                self.n[i],
                time[-1],
                dTime,
            )

            if not fd_mode:
                Z_ijkl = self._Z_ijkl[i][
                    : self.nSpatialDimensions,
                    : self.nSpatialDimensions,
                    : self.nSpatialDimensions,
                    : self.nSpatialDimensions,
                ]

                Q_ij = self._Q_ij[i][
                    : self.nSpatialDimensions,
                    : self.nSpatialDimensions,
                ]

                H_ijk = self._H_ijk[i][
                    : self.nSpatialDimensions,
                    : self.nSpatialDimensions,
                    : self.nSpatialDimensions,
                ]

                Y_ijkl = self._Y_ijkl[i][
                    : self.nSpatialDimensions,
                    : self.nSpatialDimensions,
                    : self.nSpatialDimensions,
                    : self.nSpatialDimensions,
                ]

                wfac = detJ * self._t * self._weight[i]

                K_jumpu_jumpv = assign_K_jumpu_jumpv(self.N_matrix[i], Q_ij)

                K_grad_s_u_grad_s_v_Z = assign_K_grad_s_u_grad_s_v(
                    self.B_matrix[i],
                    Z_ijkl,
                )

                K_grad_s_u_grad_s_v_Y = assign_K_grad_s_u_grad_s_v(
                    self.B_matrix[i],
                    Y_ijkl,
                )

                K_jump_u_grad_s_v = assign_K_jump_u_grad_s_v(
                    self.N_matrix[i],
                    self.B_matrix[i],
                    H_ijk,
                )

                K_grad_s_u_jump_v = assign_K_grad_s_u_jump_v(
                    self.N_matrix[i],
                    self.B_matrix[i],
                    H_ijk,
                )

                K_total_qp_unweighted = (
                    K_jumpu_jumpv
                    + K_grad_s_u_grad_s_v_Z
                    + K_grad_s_u_grad_s_v_Y
                    + K_grad_s_u_jump_v
                    + K_jump_u_grad_s_v
                )

                K_total_qp_weighted = K_total_qp_unweighted * wfac

                if LOCAL_STIFFNESS_BLOCK_DEBUG and self.elementtype.lower() == "iquad4":
                    np.set_printoptions(precision=16, suppress=False, linewidth=240)

                    print(f"\n--- PY GP {i} ---")
                    print("qp.xi_python =", np.asarray(self._qpoints[i], dtype=float))
                    print("qp.xi_cxx_equivalent =", 2.0 * np.asarray(self._qpoints[i], dtype=float) - 1.0)
                    print("qp.weight_python =", self._weight[i])
                    print("qp.sqrtDetG =", detJ)
                    print("qp.J0xW =", wfac)

                    print("\nQ_ij shape =", Q_ij.shape)
                    print("Q_ij =")
                    print(
                        np.array2string(
                            Q_ij,
                            precision=16,
                            suppress_small=False,
                            max_line_width=240,
                        )
                    )

                    print("\nZ_ijkl shape =", Z_ijkl.reshape(9, 9).shape)
                    print("Z_ijkl =")
                    print(
                        np.array2string(
                            Z_ijkl.reshape(9, 9),
                            precision=16,
                            suppress_small=False,
                            max_line_width=240,
                        )
                    )

                    print("\nY_ijkl shape =", Y_ijkl.reshape(9, 9).shape)
                    print("Y_ijkl =")
                    print(
                        np.array2string(
                            Y_ijkl.reshape(9, 9),
                            precision=16,
                            suppress_small=False,
                            max_line_width=240,
                        )
                    )

                    print("\nH_ijk shape =", H_ijk.reshape(3, 9).shape)
                    print("H_ijk =")
                    print(
                        np.array2string(
                            H_ijk.reshape(3, 9),
                            precision=16,
                            suppress_small=False,
                            max_line_width=240,
                        )
                    )

                    print("\nK_jumpu_jumpv shape =", K_jumpu_jumpv.shape)
                    print("K_jumpu_jumpv =")
                    print(
                        np.array2string(
                            K_jumpu_jumpv,
                            precision=16,
                            suppress_small=False,
                            max_line_width=240,
                        )
                    )

                    print("\nK_grad_s_u_grad_s_v_Z shape =", K_grad_s_u_grad_s_v_Z.shape)
                    print("K_grad_s_u_grad_s_v_Z =")
                    print(
                        np.array2string(
                            K_grad_s_u_grad_s_v_Z,
                            precision=16,
                            suppress_small=False,
                            max_line_width=240,
                        )
                    )

                    print("\nK_grad_s_u_grad_s_v_Y shape =", K_grad_s_u_grad_s_v_Y.shape)
                    print("K_grad_s_u_grad_s_v_Y =")
                    print(
                        np.array2string(
                            K_grad_s_u_grad_s_v_Y,
                            precision=16,
                            suppress_small=False,
                            max_line_width=240,
                        )
                    )

                    print("\nK_grad_s_u_jump_v shape =", K_grad_s_u_jump_v.shape)
                    print("K_grad_s_u_jump_v =")
                    print(
                        np.array2string(
                            K_grad_s_u_jump_v,
                            precision=16,
                            suppress_small=False,
                            max_line_width=240,
                        )
                    )

                    print("\nK_jump_u_grad_s_v shape =", K_jump_u_grad_s_v.shape)
                    print("K_jump_u_grad_s_v =")
                    print(
                        np.array2string(
                            K_jump_u_grad_s_v,
                            precision=16,
                            suppress_small=False,
                            max_line_width=240,
                        )
                    )

                    print("\nK_total_qp_unweighted shape =", K_total_qp_unweighted.shape)
                    print("K_total_qp_unweighted =")
                    print(
                        np.array2string(
                            K_total_qp_unweighted,
                            precision=16,
                            suppress_small=False,
                            max_line_width=240,
                        )
                    )

                    print("\nK_total_qp_weighted shape =", K_total_qp_weighted.shape)
                    print("K_total_qp_weighted =")
                    print(
                        np.array2string(
                            K_total_qp_weighted,
                            precision=16,
                            suppress_small=False,
                            max_line_width=240,
                        )
                    )

                K += K_total_qp_weighted.flatten()

            P_jumpv = assign_P_jumpv(self.N_matrix[i], self._force_at_Gauss)

            P -= P_jumpv.flatten() * detJ * self._t * self._weight[i]

            P_grad_s_v = assign_P_grad_s_v(
                self.B_matrix[i],
                self._surface_stress_at_Gauss,
            )

            P -= P_grad_s_v.flatten() * detJ * self._t * self._weight[i]

            self._stateVarsTemp[i][0 : self.nSpatialDimensions] = self._force_at_Gauss

            self._stateVarsTemp[i][3 : int(3 + self.nSpatialDimensions**2)] = (
                self._surface_stress_at_Gauss.reshape(-1)
            )

            self._stateVarsTemp[i][12 : int(12 + self._dU_GPs[i].shape[0])] += self._dU_GPs[i]

            self._stateVarsTemp[i][18 : int(18 + self._dSurface_strain_GPs[i].shape[0])] += (
                self._dSurface_strain_GPs[i]
            )

            if fd_mode:
                stateVars_old = np.array(self._stateVarsRef[i][36:], dtype=np.float64)
                stateVars_base = np.array(self._stateVarsTemp[i][36:], dtype=np.float64)

                J_jumpv_temp, J_grad_s_v_temp = self.calculate_forward_gradient_X_right(
                    self.N_matrix[i],
                    self.B_matrix[i],
                    time,
                    dTime,
                    dU,
                    i,
                    P_jumpv[:, 0],
                    P_grad_s_v[:, 0],
                    force_state_old,
                    surface_stress_state_old,
                    stateVars_old,
                    stateVars_base,
                )

                detJ = self.sqrt_detG[i]

                self._fd_debug_log(
                    f"[FD GP={i}] ||J_jumpv_temp||_F = "
                    f"{np.linalg.norm(J_jumpv_temp * detJ * self._t * self._weight[i]):.18e}"
                )

                self._fd_debug_log(
                    f"[FD GP={i}] ||J_grad_s_v_temp||_F = "
                    f"{np.linalg.norm(J_grad_s_v_temp * detJ * self._t * self._weight[i]):.18e}"
                )

                K += (J_jumpv_temp + J_grad_s_v_temp).flatten() * detJ * self._t * self._weight[i]

                self._J_jumpv += J_jumpv_temp * detJ * self._t * self._weight[i]
                self._J_grad_s_v += J_grad_s_v_temp * detJ * self._t * self._weight[i]

        if self._stiffness_export_mode != "none" and not self._has_exported_stiffness:
            export_dir = self._stiffness_export_dir
            export_dir.mkdir(parents=True, exist_ok=True)

            implicit_matrix = (K - K_before).reshape((self.nDof, self.nDof))
            export_mode = self._stiffness_export_mode

            if export_mode == "implicit":
                export_matrix = implicit_matrix
            elif export_mode == "fd":
                J_jumpv_matrix = self._J_jumpv.reshape((self.nDof, self.nDof))
                J_grad_s_v_matrix = self._J_grad_s_v.reshape((self.nDof, self.nDof))
                export_matrix = J_jumpv_matrix + J_grad_s_v_matrix
            else:
                export_matrix = None

            if export_matrix is not None:
                export_path = export_dir / f"{self._stiffness_export_basename}_el{self._elNumber}_{export_mode}.txt"
                np.savetxt(export_path, export_matrix, fmt="%.18e")
                self._has_exported_stiffness = True

    def _fd_debug_log(self, message: str):
        if not self._fd_debug_log_enabled:
            return

        log_dir = self._stiffness_export_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / f"{self._stiffness_export_basename}_el{self._elNumber}_fd_debug.txt"

        mode = "w" if not self._has_reset_fd_debug_log else "a"

        with log_path.open(mode, encoding="utf-8") as log_file:
            log_file.write(message)
            if not message.endswith("\n"):
                log_file.write("\n")

        self._has_reset_fd_debug_log = True

    def calculate_forward_gradient_X_right(
        self,
        N_matrix,
        B_matrix,
        time,
        dTime,
        dU,
        i,
        P_jumpv_X,
        P_grad_s_v_X,
        force_state_old,
        surface_stress_state_old,
        stateVars_old,
        stateVars_base,
    ):
        J_jumpv = np.zeros((self.nDof, self.nDof))
        J_grad_s_v = np.zeros((self.nDof, self.nDof))

        dU_array = np.asarray(dU)

        force_base_state = np.array(force_state_old, dtype=np.float64)
        surface_stress_base_state = np.array(surface_stress_state_old, dtype=np.float64)

        stateVars_buf_base = np.array(stateVars_old, dtype=np.float64)
        self.material.assignStateVars(stateVars_buf_base)

        if i == 0:
            self._fd_debug_log(
                f"[FD base GP={i}] ||dU_GP||_2={np.linalg.norm(self._dU_GPs[i]):.6e} "
                f"||dSurf_GP||_2={np.linalg.norm(self._dSurface_strain_GPs[i]):.6e}"
            )

        self.material.computeStress(
            force_base_state,
            surface_stress_base_state,
            self._Q_ij[i].copy(),
            self._Z_ijkl[i].copy(),
            self._H_ijk[i].copy(),
            self._Y_ijkl[i].copy(),
            self._dU_GPs[i],
            self._dSurface_strain_GPs[i],
            self.n[i],
            time[-1],
            dTime,
        )

        P_jumpv_0 = assign_P_jumpv(N_matrix, force_base_state)[:, 0]
        P_grad_s_v_0 = assign_P_grad_s_v(B_matrix, surface_stress_base_state)[:, 0]

        for p in range(dU_array.size):
            stateVars_buf = np.array(stateVars_old, dtype=np.float64)
            self.material.assignStateVars(stateVars_buf)

            force_at_Gauss_right = force_state_old.copy()
            surface_stress_at_Gauss_right = surface_stress_state_old.copy()

            dU_right = dU_array.reshape(-1).copy()
            epsilon = max(1.0, np.abs(dU_right[p])) * 1e-10

            dU_right[p] += epsilon
            dU_right = dU_right.reshape((-1, 3))

            dU_GPs_bottom_right = np.einsum(
                "cm,m->c",
                N_matrix,
                dU_right[: self.number_of_element_nodes].flatten(),
            )

            dU_GPs_top_right = np.einsum(
                "cm,m->c",
                N_matrix,
                dU_right[self.number_of_element_nodes :].flatten(),
            )

            dU_GPs_right = np.ascontiguousarray(
                np.hstack((dU_GPs_top_right, dU_GPs_bottom_right))
            )

            dSurface_strain_GPs_bottom_right = np.einsum(
                "cm,m->c",
                B_matrix,
                dU_right[: self.number_of_element_nodes].flatten(),
            )

            dSurface_strain_GPs_top_right = np.einsum(
                "cm,m->c",
                B_matrix,
                dU_right[self.number_of_element_nodes :].flatten(),
            )

            dSurface_strain_GPs_right = np.ascontiguousarray(
                np.hstack((dSurface_strain_GPs_top_right, dSurface_strain_GPs_bottom_right))
            )

            dU_GPs_delta = np.ascontiguousarray(dU_GPs_right - self._dU_GPs[i])

            dSurface_strain_GPs_delta = np.ascontiguousarray(
                dSurface_strain_GPs_right - self._dSurface_strain_GPs[i]
            )

            if i == 0 and (p < 3 or p == dU_array.size - 1):
                self._fd_debug_log(
                    f"[FD col GP={i} p={p}] eps={epsilon:.6e} "
                    f"||dU_delta||_2={np.linalg.norm(dU_GPs_delta):.6e} "
                    f"||dSurf_delta||_2={np.linalg.norm(dSurface_strain_GPs_delta):.6e}"
                )

            self.material.computeStress(
                force_at_Gauss_right,
                surface_stress_at_Gauss_right,
                self._Q_ij[i].copy(),
                self._Z_ijkl[i].copy(),
                self._H_ijk[i].copy(),
                self._Y_ijkl[i].copy(),
                dU_GPs_right,
                dSurface_strain_GPs_right,
                self.n[i],
                time[-1],
                dTime,
            )

            P_jumpv_X_right = assign_P_jumpv(N_matrix, force_at_Gauss_right)[:, 0]
            P_grad_s_v_X_right = assign_P_grad_s_v(B_matrix, surface_stress_at_Gauss_right)[:, 0]

            J_jumpv[:, p] = (P_jumpv_X_right - P_jumpv_0) / epsilon
            J_grad_s_v[:, p] = (P_grad_s_v_X_right - P_grad_s_v_0) / epsilon

        self.material.assignStateVars(self._stateVarsTemp[i][36:])

        if i == 0:
            self._fd_debug_log(f"[FD] J_jumpv[0,0]    = {J_jumpv[0, 0]:.6g}")
            self._fd_debug_log(f"[FD] J_grad_s_v[0,0] = {J_grad_s_v[0, 0]:.6g}")

        return J_jumpv, J_grad_s_v

    def calculate_central_gradient_X_right(self, N_matrix, B_matrix, time, dTime, dU, i):
        J_jumpv = np.zeros((self.nDof, self.nDof))
        J_grad_s_v = np.zeros((self.nDof, self.nDof))

        dStressdStrain = np.zeros([self._nInt, self._matrixSize, self._matrixSize])

        for p in range(dU.flatten().shape[0]):
            force_at_Gauss_right = copy.deepcopy(self._force_at_Gauss_X)
            surface_stress_at_Gauss_right = copy.deepcopy(self._surface_stress_at_Gauss_X)

            dU_right = dU.copy().flatten()
            epsilon = max(1.0, np.abs(dU_right.flatten()[p])) * 1e-4

            dU_right[p] += epsilon
            dU_right = dU_right.reshape((-1, 3))

            dU_GPs_top_right = np.einsum(
                "cm,m->c",
                N_matrix,
                dU_right[: self.number_of_element_nodes].flatten(),
            )

            dU_GPs_bottom_right = np.einsum(
                "cm,m->c",
                N_matrix,
                dU_right[self.number_of_element_nodes :].flatten(),
            )

            dU_GPs_right = np.ascontiguousarray(
                np.hstack((dU_GPs_top_right, dU_GPs_bottom_right))
            )

            dSurface_strain_GPs_top_right = np.einsum(
                "cm,m->c",
                B_matrix,
                dU_right[: self.number_of_element_nodes].flatten(),
            )

            dSurface_strain_GPs_bottom_right = np.einsum(
                "cm,m->c",
                B_matrix,
                dU_right[self.number_of_element_nodes :].flatten(),
            )

            dSurface_strain_GPs_right = np.ascontiguousarray(
                np.hstack((dSurface_strain_GPs_top_right, dSurface_strain_GPs_bottom_right))
            )

            self.material.computeStress(
                force_at_Gauss_right,
                surface_stress_at_Gauss_right,
                self._Q_ij[i],
                self._Z_ijkl[i],
                self._H_ijk[i],
                self._Y_ijkl[i],
                dU_GPs_right[i],
                dSurface_strain_GPs_right[i],
                self.n[i],
                time[-1],
                dTime,
            )

            P_jumpv_X_right = assign_P_jumpv(N_matrix, force_at_Gauss_right)
            P_grad_s_v_X_right = assign_P_grad_s_v(B_matrix, surface_stress_at_Gauss_right)

            force_at_Gauss_left = copy.deepcopy(self._force_at_Gauss_X)
            surface_stress_at_Gauss_left = copy.deepcopy(self._surface_stress_at_Gauss_X)

            dU_left = dU.copy().flatten()
            dU_left[p] -= epsilon
            dU_left = dU_left.reshape((-1, 3))

            dU_GPs_top_left = np.einsum(
                "cm,m->c",
                self.N_matrix,
                dU_left[: self.number_of_element_nodes].flatten(),
            )

            dU_GPs_bottom_left = np.einsum(
                "cm,m->c",
                self.N_matrix,
                dU_left[self.number_of_element_nodes :].flatten(),
            )

            dU_GPs_left = np.ascontiguousarray(
                np.hstack((dU_GPs_top_left, dU_GPs_bottom_left))
            )

            dSurface_strain_GPs_top_left = np.einsum(
                "cm,m->c",
                self.B_matrix,
                dU_left[: self.number_of_element_nodes].flatten(),
            )

            dSurface_strain_GPs_bottom_left = np.einsum(
                "cm,m->c",
                self.B_matrix,
                dU_left[self.number_of_element_nodes :].flatten(),
            )

            dSurface_strain_GPs_left = np.ascontiguousarray(
                np.hstack((dSurface_strain_GPs_top_left, dSurface_strain_GPs_bottom_left))
            )

            self.material.computeStress(
                force_at_Gauss_left,
                surface_stress_at_Gauss_left,
                dStressdStrain[i],
                dU_GPs_left[i],
                dSurface_strain_GPs_left[i],
                self.n[i],
                time[-1],
                dTime,
            )

            P_jumpv_X_left = assign_P_jumpv(N_matrix, force_at_Gauss_left)
            P_grad_s_v_X_left = assign_P_grad_s_v(B_matrix, surface_stress_at_Gauss_left)

            J_jumpv[:, p] = (P_jumpv_X_right[:, 0] - P_jumpv_X_left[:, 0]) / (2.0 * epsilon)
            J_grad_s_v[:, p] = (P_grad_s_v_X_right[:, 0] - P_grad_s_v_X_left[:, 0]) / (2.0 * epsilon)

        return J_jumpv, J_grad_s_v

    def computeBodyForce(
        self,
        P: np.ndarray,
        K: np.ndarray,
        load: np.ndarray,
        U: np.ndarray,
        time: np.ndarray,
        dTime: float,
    ):
        Nbasis = computeNOperator(
            self._nodesCoordinates,
            self._element,
            self._qpoints,
            self.nSpatialDimensions,
        )

        for i in range(self._nInt):
            J, _ = self.grad[:, :, :, i]
            P += np.outer(Nbasis[:, :, i], load).flatten() * lin.det(J[:, :, i]) * self._t * self._weight[i]

    def acceptLastState(self):
        self._stateVarsRef[:] = [self._stateVarsTemp[i][:] for i in range(self._nInt)]

    def resetToLastValidState(self):
        pass

    def getResultArray(
        self,
        result: str,
        quadraturePoint: int,
        getPersistentView: bool = True,
    ) -> np.ndarray:
        return self._stateVars[quadraturePoint][result]

    def getCoordinatesAtCenter(self) -> np.ndarray:
        x = self._nodesCoordinates
        return np.average(x, axis=1)

    def getNumberOfQuadraturePoints(self) -> int:
        return self._nInt

    def getCoordinatesAtQuadraturePoints(self) -> np.ndarray:
        N = computeNOperator(
            self._nodesCoordinates,
            self._element,
            self._qpoints,
            self.nSpatialDimensions,
        )
        return self._nodesCoordinates @ N