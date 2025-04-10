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
import warnings

from numpy._core.fromnumeric import shape
#import importlib

warnings.filterwarnings(
    "ignore",
    message=".*Conversion of an array with ndim > 0 to a scalar is deprecated.*",
    category=DeprecationWarning
)
import basix
import numpy as np
import numpy.linalg as lin

warnings.simplefilter("ignore", DeprecationWarning)
#importlib.reload(np)

import sys
sys.path.append('/home/alexsta1993/alexandros/EdelweissFE')
from edelweissfe.elements.base.baseelement import BaseElement
from edelweissfe.materials.marmotinterfacematerial.marmotinterfacematerialwrapper import MarmotInterfaceMaterialWrapper

from edelweissfe.elements.interfaceelement.elementmatrices import (
    compute_grad,
    compute_surface_grad,
    compute_surface_div_grad,
    interface_geometry,
    assign_K_jumpu_jumpv,
    assign_K_grad_s_u_grad_s_v,
    assign_K_grad_s_u_jump_v,
    assign_K_jump_u_grad_s_v,
    assign_P_jumpv,
    assign_P_grad_s_v,
    computeJacobian,
    computeNOperator,
)
from edelweissfe.points.node import Node
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict

# The interface element is either a 1D or a 2D element

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
    ),
    ILine3=dict(
        nNodes=6,
        nDof=12,
        dofIndices=np.arange(0, 12),
        ensightType="line3",
        nSpatialDimensions=2,
        nInt=3,
        element=basix.create_element(basix.ElementFamily.P, basix.CellType.interval, 2, basix.LagrangeVariant.equispaced),
        qpoints=basix.make_quadrature(basix.CellType.interval, 4)[0],
        w=basix.make_quadrature(basix.CellType.interval, 4)[1],
        matSize=3,
        index=np.array([0, 1, 3]),
        plStrain=True,
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
    ),
    IQuad4=dict(
        nNodes=8,
        nDof=24,
        dofIndices=np.arange(0, 24),
        ensightType="quad4",
        nSpatialDimensions=3,
        nInt=4,
        element=basix.create_element(
            basix.ElementFamily.P, basix.CellType.quadrilateral, 1
        ),
        qpoints=basix.make_quadrature(basix.CellType.quadrilateral, 2)[0],
        w=basix.make_quadrature(basix.CellType.quadrilateral, 2)[1],
        matSize=3,
        index=np.array([0, 1, 3]),
        plStrain=True,
        reorder_nodes_list=[0,1,3,2,4,5,7,6],
    ),
    IQuad8=dict(
        nNodes=16,
        nDof=48,
        dofIndices=np.arange(0, 48),
        ensightType="quad8",
        nSpatialDimensions=3,
        nInt=9,
        element=basix.create_element(
            basix.ElementFamily.serendipity, basix.CellType.quadrilateral, 2
        ),
        qpoints=basix.make_quadrature(basix.CellType.quadrilateral, 4)[0],
        w=basix.make_quadrature(basix.CellType.quadrilateral, 4)[1],
        matSize=3,
        index=np.array([0, 1, 3]),
        plStrain=True,
    ),
    IQuad8R=dict(
        nNodes=16,
        nDof=48,
        dofIndices=np.arange(0, 48),
        ensightType="quad8",
        nSpatialDimensions=3,
        nInt=8,
        element=basix.create_element(
            basix.ElementFamily.serendipity, basix.CellType.quadrilateral, 2
        ),
        # remove the last quadrature point at t\he center
        qpoints=np.delete(basix.make_quadrature(basix.CellType.quadrilateral, 4)[0], 4, axis = 0),
        w=np.delete(basix.make_quadrature(basix.CellType.quadrilateral, 4)[1], 4),
        matSize=3,
        index=np.array([0, 1, 3]),
        plStrain=True,
    ),
    IQuad9R=dict(
        nNodes=18,
        nDof=54,
        dofIndices=np.arange(0, 54),
        ensightType="quad9",
        nSpatialDimensions=3,
        nInt=8,
        element=basix.create_element(
            basix.ElementFamily.P, basix.CellType.quadrilateral, 2
        ),
        qpoints=np.delete(basix.make_quadrature(basix.CellType.quadrilateral, 4)[0], 4, axis = 0),
        w=np.delete(basix.make_quadrature(basix.CellType.quadrilateral, 4)[1], 4),
        matSize=3,
        index=np.array([0, 1, 3]),
        plStrain=True,
    ),
)

# add variations
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
    """This element can be used for EdelweissFE.
    The element currently only allows calculations with node forces
    and given displacements.

    Parameters
    ----------
    elementType
        A string identifying the requested element formulation as shown below.
    elNumber
        A unique integer label used for all kinds of purposes.

    The following types of elements and attributes are currently possible
    (elementType):

    Elements
    --------
        ILine2
            line element with 4 nodes.
        ILine3
            line element with 6 nodes.
        IQuad4
            quadrilateral element with 8 nodes.
        IQud8
            quadrilateral element with 16 nodes.
        IQuad9
            quadrilateral element with 18 nodes.

    optional Parameters
    -------------------
    The following attributes are also included in the elementtype definition:

        R
            reduced integration for element, in elementtype[5].

    If R is not given by the user, we assume regular increment.
    """

    @property
    def elNumber(self) -> int:
        """The unique number of this element"""

        return self._elNumber  # return number

    @property
    def nNodes(self) -> int:
        """The number of nodes this element requires"""

        return self._nNodes

    @property
    def nodes(self) -> int:
        """The list of nodes this element holds"""

        return self._nodes

    @property
    def nDof(self) -> int:
        """The total number of degrees of freedom this element has"""

        return self._nDof

    @property
    def fields(self) -> list[list[str]]:
        """The list of fields per nodes."""

        return self._fields

    @property
    def dofIndicesPermutation(self) -> np.ndarray:
        """The permutation pattern for the residual vector and the stiffness
        matrix to aggregate all entries in order to resemble the defined
        fields nodewise. In this case it stays the same because we use the
        nodes exactly like they are."""

        return self._dofIndices

    @property
    def ensightType(self) -> str:
        """The shape of the element in Ensight Gold notation."""

        return self._ensightType

    @property
    def visualizationNodes(self) -> str:
        """The nodes for visualization."""

        return self._nodes

    @property
    def hasMaterial(self) -> str:
        """Flag to check if a material was assigned to this element."""

        return self._hasMaterial
    
    @property
    def reorder_nodes_list(self):
        return self._reorder_nodes_list

    def __init__(self, elementType: str, elNumber: int):
        self.elementtype = (
            elementType[0].upper() + elementType[1:5].lower() + elementType[5:].upper()
        )
        self._elNumber = elNumber
        try:
            if (
                len(self.elementtype) > 5 and self.elementtype[5].lower() == "n"
            ):  # normal integration
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

        print('QPOINTS',self._qpoints)
        print('W', self._weight)

        self._matrixSize = properties["matSize"]
        self._activeVoigtIndices = properties["index"]
        self.planeStrain = properties["plStrain"]
        
        if self.nSpatialDimensions >1:
            self._t = 1  # "thickness" for 3D elements
        self._fields = [["displacement"] for i in range(self._nNodes)]
        self._dStrain = np.zeros([self._nInt, 9]) # We don't assume symmetry of the stresses (The material is symmetric instead)

        self.number_of_dofs =int(self._nDof*self.nSpatialDimensions)
        self.number_of_strain_comp = int(self._nDof*self.nSpatialDimensions*self.nSpatialDimensions)
        
        self._dU_GPs = np.zeros((self._nInt, self.nSpatialDimensions*2)) # GPs times the spatial dimension for vector elements top interpolation and bottom interpolation 
        self._dSurface_strain_GPs = np.zeros((self._nInt,self.nSpatialDimensions*self.nSpatialDimensions*2)) #See the definition of the einsum below q,i*j*d
        self._reorder_nodes_list =properties["reorder_nodes_list"]
 
    def setNodes(self, nodes: list[Node]):
        """Assign the nodes to the element.

        Parameters
        ----------
        nodes
            A list of nodes.
        """

        self._nodes = nodes
        _nodesCoordinates = np.array(
            [n.coordinates for n in nodes]
        )  # get node coordinates
        self._nodesCoordinates = (
            _nodesCoordinates.transpose()
        )[:,self.reorder_nodes_list]
        print('self._nodesCoordinates ',self._nodesCoordinates.shape )
        #[self.reorder_nodes_list,:]  # nodes given column-wise: x-coordinate - y-coordinate

    def setProperties(self, elementProperties: np.ndarray):
        """Assign a set of properties to the element.

        Parameters
        ----------
        elementProperties
            A numpy array containing the element properties.

        Attributes
        ----------
        thickness
            Thickness of 2D elements.
        """

        if self.elementtype[0] == "I":
            self._t = elementProperties[0]  # thickness

    def initializeElement(
        self,
    ):
        """Initalize the element to be ready for computing."""

        # initialize the matrices
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
            self.nSpatialDimensions
        )

    def setMaterial(self, material: type):
        """Assign a material
        Parameters
        ----------
        material
            An initialized instance of a material
        """
        self.material = material
        stateVarsSize = (3+9) + 2*(3+ 9) +self.material.getNumberOfRequiredStateVars()
        self._matrixSize = 21

        self._dStressdStrain = np.zeros(
            [self._nInt, self._matrixSize, self._matrixSize]
        )
        self._hasMaterial = True
        self._stateVarsRef = np.zeros([self._nInt, stateVarsSize])
        self._stateVars =[ 
            CaseInsensitiveDict(
                {
                    "force": self._stateVarsRef[i][0:3],
                    "surface stress": self._stateVarsRef[i][3:12],
                    "displacement":  self._stateVarsRef[i][12:18], 
                    "surface strain": self._stateVarsRef[i][18:36],
                    "materialstate": self._stateVarsRef[i][36:],
                }
            )
            for i in range(self._nInt)
        ]
        self._stateVarsTemp = np.zeros([self._nInt, stateVarsSize])
        print('material Initialization was successfull')

    def setInitialCondition(self, stateType: str, values: np.ndarray):
        """Assign initial conditions.

        Parameters
        ----------
        stateType
            The type of initial state.
        values
            The numpy array describing the initial state.
        """

        raise Exception(
            "Setting an initial condition is not possible\
                        with this element provider."
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
        """Evaluate residual and stiffness for given time, field, and field
        increment due to a surface load.

        Parameters
        ----------
        loadType
            The type of load.
        P
            The external load vector to be defined.
        K
            The stiffness matrix to be defined.
        faceID
            The number of the elements face this load acts on.
        load
            The magnitude (or vector) describing the load.
        U
            The current solution vector.
        time
            Array of step time and total time.
        dTime
            The time increment.
        """

        raise Exception(
            "Applying a distributed load is currently not possible\
                        with this element provider."
        )

    def computeYourself(self, K: np.ndarray, P: np.ndarray, U: np.ndarray, dU: np.ndarray, time: np.ndarray, dTime: float,):
        """Evaluate the residual and stiffness matrix for given time, field,
        and field increment due to a displacement or load.

        Parameters
        ----------
        P
            The external load vector gets calculated.
        K
            The stiffness matrix gets calculated.
        U
            The current solution vector.
        dU
            The current solution vector increment.
        time
            Array of step time and total time.
        dTime
            The time increment.
        """

        # assume it's plain strain if it's not given by user

        dU = dU.reshape((self._nNodes,-1)) 
        # copy all elements
        self._stateVarsTemp = [
            self._stateVarsRef[i].copy() for i in range(self._nInt)
        ].copy()

        # strain increment
        # displacement increment top and bottom
        self.number_of_element_nodes = int(self._nNodes/2) # The element has double the number of spatial dofs we keep only the nodes necessary for the geometric decription 
        self.number_of_top_dofs =int(self._nDof/2)
        self.number_of_top_strain_comp = int(self.nSpatialDimensions*self.nSpatialDimensions)
        print('dU at nodes shape:', dU.shape) 
        print('dU at nodes:\n', dU)
        dU_GPs_top = np.einsum('aiq,ai->iq',self.basis_function[:,:,:,0],dU[:self.number_of_element_nodes]).transpose((1,0))
        dU_GPs_bottom = np.einsum('aiq,ai->iq',self.basis_function[:,:,:,0],dU[self.number_of_element_nodes:]).transpose((1,0))

        self._dU_GPs =np.ascontiguousarray( np.hstack((dU_GPs_top, dU_GPs_bottom)))
      
        for i in range(self._nInt):
            # get stress and strain
            self._force_at_Gauss = self._stateVarsTemp[i][0:self.nSpatialDimensions] # 3
            self._surface_stress_at_Gauss = self._stateVarsTemp[i][3:int(3+self.nSpatialDimensions**2)].reshape((self.nSpatialDimensions,self.nSpatialDimensions)) #9

            self.material.assignStateVars(self._stateVarsTemp[i][24:]) #Not necessary for now... (elasticity)

            # use 3D for 2D planeStrain
            # Calculate material properties for the interface
            # These are computed at the begining of the analysis. Then we store them to the dsde matrix.
            # Their evolution is described by a return mapping algorithm (see Steimann and Mosler).

            if not self.planeStrain and self.nSpatialDimensions == 2:
                raise Exception("Plain stress is not yet implemented in this element provider.")
                # self.material.computePlaneStress(stress, self._dStressdStrain[i], self._dStrain[i], time, dTime)
            else:
                #print('self._force_at_Gauss',self._force_at_Gauss.shape)
                #print('self._surface_stress_at_Gauss',self._surface_stress_at_Gauss.shape)
                #print('self._dStressdStrain',self._dStressdStrain[i].shape)
                #print('self._dU_GPs',self._dU_GPs[i].shape)
                #print('self._dSurface_strain_GPs',self._dSurface_strain_GPs[i].shape)
                #print('self.n',self.n[i])
                #print('time',time),
                #print('dTime',dTime)


                self.material.computeStress( self._force_at_Gauss,
                                            self._surface_stress_at_Gauss,
                                            self._dStressdStrain[i],
                                            self._dU_GPs[i],
                                            self._dSurface_strain_GPs[i],
                                            self.n[i],
                                            time[-1],
                                            dTime
                                            )

            print('self._force_at_Gauss:\n',self._force_at_Gauss)
            print('self._surface_stress_at_Gauss:\n',self._surface_stress_at_Gauss)
            #print('self._dStressdStrain',self._dStressdStrain[i])
            #print('self._dU_GPs',self._dU_GPs[i])
            #print('self._dSurface_strain_GPs',self._dSurface_strain_GPs[i])
            #print('self.n',self.n[i])
            print('time:\n',time)
            print('dTime:\n',dTime)

            Z_ijkl = self._dStressdStrain[i][:9,:9].reshape((3,3,3,3))[:self.nSpatialDimensions,:self.nSpatialDimensions,:self.nSpatialDimensions,:self.nSpatialDimensions] #Pick only the appropriate spatial dimensions
            H_inv_ij = self._dStressdStrain[i][9:12, 9:12].reshape((3,3))[:self.nSpatialDimensions,:self.nSpatialDimensions]
            H_inv_nF_ijk = self._dStressdStrain[i][:9,9:12].reshape((3,3,3))[:self.nSpatialDimensions,:self.nSpatialDimensions,:self.nSpatialDimensions]
            Yn_H_inv_Fn_ijkl = self._dStressdStrain[i][12:21,12:21].reshape((3,3,3,3))[:self.nSpatialDimensions,:self.nSpatialDimensions,:self.nSpatialDimensions,:self.nSpatialDimensions]
            

            Nbasis = self.basis_function[
                :, :, i
            ]  # Interponation operator between the nodes of one side
            grad_s = self.surface_grad[
                :, :, :, i
            ]  # surface gradient operator between the nodes of one side
            
            detJ = self.sqrt_detG[i]
        
            K_jumpu_jumpv = assign_K_jumpu_jumpv(self.grad, Nbasis, H_inv_ij, i)

            K += K_jumpu_jumpv.flatten() * detJ * self._t * self._weight[i]

            # Additional energy due to surface stiffness terms with Z_ijkl
            # get stiffness matrix for element j in point i
            K_grad_s_u_grad_s_v = assign_K_grad_s_u_grad_s_v(
                self.grad, grad_s, Z_ijkl, i
            )

            K += K_grad_s_u_grad_s_v.flatten() * detJ * self._t * self._weight[i]

            # Additional energy due to surface stiffness terms with Yn_H_inv_Fn_ijkl
            # get stiffness matrix for element j in point i
            K_grad_s_u_grad_s_v = assign_K_grad_s_u_grad_s_v(
                self.grad, grad_s, Yn_H_inv_Fn_ijkl, i
            )

            K += K_grad_s_u_grad_s_v.flatten() * detJ * self._t * self._weight[i]

            # Additional energy due to coupling between surface stiffness and jump terms with H_inv_nF_ijk
            # get stiffness matrix for element j in point i
            K_jump_u_grad_s_v = assign_K_jump_u_grad_s_v(
                self.grad, grad_s, Nbasis, H_inv_nF_ijk, i
            )

            K += K_jump_u_grad_s_v.flatten() * detJ * self._t * self._weight[i]

            # Additional energy due to coupling between jump and surface stiffness terms with H_inv_nF_ijk
            # get stiffness matrix for element j in point i
            K_grad_s_u_jump_v = assign_K_grad_s_u_jump_v(
                self.grad, grad_s, Nbasis, H_inv_nF_ijk, i
            )

            K += K_grad_s_u_jump_v.flatten() * detJ * self._t * self._weight[i]

            # Because we do not separate between coupled forces from jumps and surface elasticity only two terms are present instead of five
            # If we want more control we need to assign the extra forces and stress parts in the state variables vector increasing memory requirements
            #print('P shape:', P.shape)
            # calculate P (jump contribution)
            P_jumpv = assign_P_jumpv(self.grad, Nbasis, self._force_at_Gauss, i)
            P -= (P_jumpv.flatten() * detJ * self._t * self._weight[i])[0]

            # calculate P (surface elasticity contribution)
            P_grad_s_v = assign_P_grad_s_v(self.grad, grad_s, self._surface_stress_at_Gauss, i)
            P -= (P_grad_s_v.flatten() * detJ * self._t * self._weight[i])[0]
            

            #return the new force, surface_stress, dU_top, dU_bottom, dSurface_strain_top, dSurface_strain_bottom 
            self._stateVarsTemp[i][0:self.nSpatialDimensions] = self._force_at_Gauss
            self._stateVarsTemp[i][3:int(3+self.nSpatialDimensions**2)] = self._surface_stress_at_Gauss.reshape(-1)
            self._stateVarsTemp[i][12:int(12+self._dU_GPs[i].shape[0])] = self._dU_GPs[i]
            self._stateVarsTemp[i][18:int(18+self._dSurface_strain_GPs[i].shape[0])]  = self._dSurface_strain_GPs[i]

    def computeBodyForce(
        self,
        P: np.ndarray,
        K: np.ndarray,
        load: np.ndarray,
        U: np.ndarray,
        time: np.ndarray,
        dTime: float,
    ):
        """Evaluate residual and stiffness for given time, field, and field increment due to a body force load.

        Parameters
        ----------
        P
            The external load vector to be defined.
        K
            The stiffness matrix to be defined.
        load
            The magnitude (or vector) describing the load.
        U
            The current solution vector.
        time
            Array of step time and total time.
        dTime
            The time increment.
        """

        Nbasis = computeNOperator(
            self._nodesCoordinates,
            self._element,
            self._qpoints,
            self.nSpatialDimensions,
        )

        for i in range(self._nInt):
            J, _ = self.grad[:,:,:,i] 
            P += (
                np.outer(Nbasis[:, :, i], load).flatten()
                * lin.det(J[:, :, i])
                * self._t
                * self._weight[i]
            )

    def acceptLastState(
        self,
    ):
        """Accept the computed state (in nonlinear iteration schemes)."""

        # copy every array in array (complete copying)
        self._stateVarsRef[:] = [self._stateVarsTemp[i][:] for i in range(self._nInt)]

    def resetToLastValidState(
        self,
    ):
        """Reset to the last valid state."""

    def getResultArray(
        self, result: str, quadraturePoint: int, getPersistentView: bool = True
    ) -> np.ndarray:
        """Get the array of a result, possibly as a persistent view which is continiously
        updated by the element.

        Parameters
        ----------
        result
            The name of the result.
        quadraturePoint
            The number of the quadrature point.
        getPersistentView
            If true, the returned array should be continiously updated by the element.

        Returns
        -------
        np.ndarray
            The result.
        """

        return self._stateVars[quadraturePoint][result]

    def getCoordinatesAtCenter(self) -> np.ndarray:
        """Compute the underlying MarmotElement centroid coordinates.

        Returns
        -------
        np.ndarray
            The element's central coordinates.
        """

        x = self._nodesCoordinates
        return np.average(x, axis=1)

    def getNumberOfQuadraturePoints(self) -> int:
        """Get the number of Quadrature points the element has.

        Returns
        -------
        nInt
            The number of Quadrature points.
        """

        return self._nInt

    def getCoordinatesAtQuadraturePoints(self) -> np.ndarray:
        """Compute the underlying MarmotElement qp coordinates.

        Returns
        -------
        np.ndarray
            The element's qp coordinates.
        """

        N = computeNOperator(
            self._nodesCoordinates,
            self._element,
            self._qpoints,
            self.nSpatialDimensions,
        )
        return self._nodesCoordinates @ N
        
def main():
    #interface_element = InterfaceElement('ILine3',0)
    interface_element = InterfaceElement('IQuad4',0)
    
    #nodes = np.array([[0.0, 0.0],
    #                [1.0, 1.0]])

    #nodes = np.array([[0.0, 0.0],
    #                  [2.0, 0.0],
    #                  [1.0,0.0]])
   

    #Plane parallel to z axis

    nodes = np.array([[0, 1, 1, 0, 0, 1, 1, 0],
                      [0, 0, 1, 1, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0]])
    #nodes = np.array([[0.0, 0.0, 0.0],
    #                 [1.0, 0.0, 0.0],
    #                 [0.0, 1.0, 0.0],
    #                 [1.0, 1.0, 0.0]])
    
    #plane oriented along a normal (-0.7,0,0.7)
    #nodes = np.array([[0.0, 0.0, 0.0],
    #                 [2.0, 0.0, 2.0],
    #                 [0.0, 2.0, 0.0],
    #                 [2.0, 2.0, 2.0]])
    

    #plane normal to ()
    #nodes = np.array([
    #                 [0.0, 0.0, 0.0],
    #                 [0.0, 1.0, 0.0],
    #                 [1.0, 1.0, 0.0],
    #                 [1.0, 0.0, 0.0],
    #                 [0.5, 0.0, 0.0],
    #                 [0.0, 0.5, 0.0],
    #                 [0.5, 1.0, 0.0],
    #                 [1.0, 0.5, 0.0],
    #                 [0.5, 0.5, 0.0]
    #                 ])


    interface_element._nodesCoordinates = nodes
    interface_element.initializeElement()
    #interface_element._nodesCoordinates = interface_element.setNodes(nodes)[:int(nodes.shape[1]/2)]#nodes.transpose(1,0)[:int(nodes.shape[1]/2)]
    
    E_M= 200.0; nu_M = 0.3; E_I = 200.0; nu_I = 0.3; E_0 = 200.0*0.1; nu_0 = 0.3\

    materialProperties = np.array([E_M, nu_M, E_I, nu_I, E_0, nu_0])
    material = interface_element.setMaterial(MarmotInterfaceMaterialWrapper(materialProperties,0))
    #material = interface_element.setMaterial('1', materialProperties)
 
    force_GPs = np.repeat(np.array([100., 200., 300.]),interface_element._nInt)
    surface_stress_GPs = np.repeat(np.ones((3,3)), interface_element._nInt); dStress_dStrain_GPs = np.repeat(np.ones((21,21)), interface_element._nInt) * 2\
    
    P_nodes = np.zeros((interface_element.nDof,1))#.reshape((interface_element._nNodes,-1))
    U_nodes = np.zeros((interface_element.nDof,1))#.reshape((interface_element._nNodes,-1))
    dU_nodes = np.ones(interface_element.nDof) * 3 # the element contains 6 nodes 3 top 3 bottom with dofs per node eq to the spatialdimension i.e 12 in total
    
    print('dU_nodes shape:', dU_nodes.shape[0])
    dU_nodes[int(dU_nodes.shape[0]//2):] = -1*dU_nodes[int(dU_nodes.shape[0]//2):]
    
    #print('P_nodes',P_nodes.shape) 
    #print('U_nodes', U_nodes.shape)
    #print('dU_nodes', dU_nodes.shape)

    K = np.random.rand(interface_element._nDof,interface_element._nDof)
    K = np.einsum('jk,ik->ij',K ,K.transpose((1,0))).flatten() # create a positive definite matrix to avoid singularities
    
    #print('Alex  K:', K.shape)
    time= np.array([0.0, 0.0])
    dTime = 0.1

    interface_element.computeYourself(K, P_nodes, U_nodes, dU_nodes, time, dTime)


if __name__=="__main__":
    main()
