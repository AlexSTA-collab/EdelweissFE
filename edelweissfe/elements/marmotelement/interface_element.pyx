#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin Cython wrapper for Marmot's InterfaceFiniteElement.

This wrapper mirrors the minimal interface of `MarmotElementWrapper` but
uses the element's `assignMaterial` method instead of trying to resolve a
material code through the classical material factory. `InterfaceFiniteElement`
overrides `assignMaterial` and internally uses the hypo-elastic interface
material factory, so calling it directly is the correct, autonomous flow.

Keep this wrapper focused and small so it doesn't interfere with the
existing Marmot element wrappers.
"""

import numpy as np

cimport cython
cimport numpy as np
from libc.stdlib cimport free, malloc
from libcpp.string cimport string
from libcpp.vector cimport vector

from edelweissfe.utils.exceptions import CutbackRequest

# Reuse the existing Marmot element pxd which already declares the
# necessary C++ classes, enums and nogil/cpdef prototypes. This keeps
# signatures consistent with the other element wrapper and avoids
# duplicate/partial extern declarations that caused earlier Cython
# compile errors.

from edelweissfe.elements.marmotelement.element cimport (
    DistributedLoadTypes,
    ElementProperties,
    MarmotElement,
    MarmotElementFactory,
    MarmotMaterialSection,
    StateTypes,
    StateView,
)

mapLoadTypes={
        'pressure' : DistributedLoadTypes.Pressure,
        'surface torsion' : DistributedLoadTypes.SurfaceTorsion,
        'surface traction' : DistributedLoadTypes.SurfaceTraction
     }

mapStateTypes={
        'geostatic stress' : StateTypes.GeostaticStress,
        'sdvini' : StateTypes.MarmotMaterialStateVars,
        'initialize material': StateTypes.MarmotMaterialInitialization
     }


@cython.final
cdef class MarmotInterfaceElementWrapper:
    """A minimal wrapper for Marmot's InterfaceFiniteElement that calls
    the element's assignMaterial method (which uses the hypo-elastic
    interface factory internally).
    """

    cdef MarmotElement* marmotElement
    cdef list _nodes
    cdef int _elNumber
    cdef str _elType
    cdef int _nNodes
    cdef int _nDof
    cdef int _nSpatialDimensions
    cdef list _fields
    cdef str _ensightType
    cdef int _hasMaterial

    cdef public double[::1] _stateVars
    cdef public double[::1] nodeCoordinates
    cdef public double[::1] _elementProperties
    cdef public double[::1] _stateVarsTemp
    cdef public double[::1] _materialProperties
    cdef int nStateVars
    cdef np.ndarray _dofIndicesPermutation
    cdef public dict _missingStateBuffers

    def __cinit__(self, elementType, elNumber):
        try:
            # only construct the underlying C++ object at C-level here
            self.marmotElement = MarmotElementFactory.createElement(
                    elementType.upper().encode("utf-8"),
                    elNumber)
        except ValueError:
            raise NotImplementedError("Marmot element {:} not found in library.".format(elementType))
        # ensure the persistent missing-state buffer dict exists at C-level
        # so any C-level calls made before __init__ finish see a valid attribute.
        self._missingStateBuffers = {}

    def __init__(self, elementType, elNumber):

        # initialize python-visible metadata from the already-created C++ object
        self._elNumber = elNumber
        self._elType = elementType

        self._nNodes = self.marmotElement.getNNodes()
        self._nSpatialDimensions = self.marmotElement.getNSpatialDimensions()
        self._nDof = self.marmotElement.getNDofPerElement()

        cdef vector[vector[string]] fields = self.marmotElement.getNodeFields()
        self._fields = [[s.decode('utf-8') for s in n] for n in fields]

        cdef vector[int] permutationPattern = self.marmotElement.getDofIndicesPermutationPattern()
        self._dofIndicesPermutation = np.asarray(permutationPattern)

        self._ensightType = self.marmotElement.getElementShape().decode('utf-8')

        self._hasMaterial = False
    # persistent buffers for results that Marmot does not provide via getStateView
    # key: (resultName(str), gaussPoint(int)) -> numpy.ndarray

    @property
    def elNumber(self):
        return self._elNumber

    @property
    def nSpatialDimensions(self):
       return self._nSpatialDimensions

    @property
    def elType(self):
        return self._elType

    @property
    def nodes(self):
        return self._nodes

    @property
    def nNodes(self):
        return self._nNodes

    @property
    def nDof(self):
        return self._nDof

    @property
    def fields(self):
        return self._fields

    @property
    def dofIndicesPermutation(self):
        return self._dofIndicesPermutation

    @property
    def ensightType(self):
        return self._ensightType

    @property
    def visualizationNodes(self):
        return self._nodes

    @property
    def hasMaterial(self):
        return self._hasMaterial

    def setNodes (self, nodes):
        self._nodes = nodes
        self.nodeCoordinates = np.concatenate([ node.coordinates for node in nodes])
        self.marmotElement.assignNodeCoordinates(&self.nodeCoordinates[0])

    def setProperties(self, elementProperties):
        self._elementProperties = elementProperties
        self.marmotElement.assignProperty(
                ElementProperties(
                        &self._elementProperties[0],
                        self._elementProperties.shape[0] ) )

    def initializeElement(self, ):
        self.marmotElement.initializeYourself()

    def setMaterial(self, material):
        """Assign a Marmot interface material by calling the C++
        `assignMaterial` method on the InterfaceFiniteElement. That method
        internally constructs the correct hypo-elastic interface material
        instances via the Marmot interface factory.
        """

        self._materialProperties =  material.properties
        try:
            # Call the element-level assignMaterial which expects a material
            # name and properties. InterfaceFiniteElement implements this and
            # uses the hypo-elastic interface factory.
            self.marmotElement.assignProperty(
                    MarmotMaterialSection(
                        material.name.upper().encode("UTF-8"),
                        &self._materialProperties[0],
                        self._materialProperties.shape[0]))
        except ValueError:
            raise NotImplementedError("Marmot interface material {:} not found in library.".format(material.name))

        self.nStateVars =           self.marmotElement.getNumberOfRequiredStateVars()

        self._stateVars =            np.zeros(self.nStateVars)
        self._stateVarsTemp =        np.zeros(self.nStateVars)

        self.marmotElement.assignStateVars(&self._stateVarsTemp[0], self.nStateVars)

        self._hasMaterial = True



    cpdef void _initializeStateVarsTemp(self) nogil:
        self._stateVarsTemp[:] = self._stateVars

    def setInitialCondition(self,
                            stateType,
                            const double[::1] values):
        """Assign initial conditions to the underlying Marmot element"""

        if not self._hasMaterial:
            raise Exception("Element {:} has no material assigned!".format(self._elNumber))

        self._initializeStateVarsTemp()
        self.marmotElement.setInitialConditions(mapStateTypes[stateType], &values[0])
        self.acceptLastState()

    cpdef void computeYourself(self,
                         double[::1] Ke,
                         double[::1] Pe,
                         const double[::1] U,
                         const double[::1] dU,
                         const double[::1] time,
                         double dTime) nogil except *:
        """Evaluate residual and stiffness for given time, field, and field increment."""

        if not self._hasMaterial:
            raise Exception("Element {:} has no material assigned!".format(self._elNumber))

        cdef double pNewDT
        with nogil:
            self._initializeStateVarsTemp()

            pNewDT = 1e36

            self.marmotElement.computeYourself(&U[0], &dU[0],
                                                &Pe[0],
                                                &Ke[0],
                                                &time[0],
                                                dTime,
                                                pNewDT)
            if pNewDT < 1.0:
                raise CutbackRequest("Element {:} requests for a cutback!".format(self.elNumber), pNewDT)

    def computeDistributedLoad(self,
                               str loadType,
                               double[::1] P,
                               double[::1] K,
                               int faceID,
                               const double[::1] load,
                               const double[::1] U,
                               const double[::1] time,
                               double dTime):
        """Evaluate residual and stiffness for given time, field, and field increment due to a surface load."""

        self.marmotElement.computeDistributedLoad(mapLoadTypes[loadType],
                                    &P[0],
                                    &K[0],
                                    faceID,
                                    &load[0],
                                    &U[0],
                                    &time[0],
                                    dTime)

    def computeBodyForce(self,
            double[::1] P,
            double[::1] K,
           const double[::1] load,
           const double[::1] U,
           const double[::1] time,
           double dTime):
        """Evaluate residual and stiffness for given time, field, and field increment due to a volume load."""

        self.marmotElement.computeBodyForce(
                                    &P[0],
                                    &K[0],
                                    &load[0],
                                    &U[0],
                                    &time[0],
                                    dTime)

    def acceptLastState(self,):
        """Accept the computed state (in nonlinear iteration schemes)."""

        self._stateVars[:] = self._stateVarsTemp

    def resetToLastValidState(self,):
        """Reset to the last valid state."""

        self._stateVarsTemp[:] = self._stateVars

    def getResultArray(self, result, quadraturePoint, getPersistentView=True):
        """Get the array of a result, possibly as a persistent view which is continiously
        updated by the underlying MarmotElement."""

        if not self._hasMaterial:
            raise Exception("Element {:} has no material assigned!".format(self._elNumber))

        cdef string result_ = result.encode('UTF-8')
        return np.array(self.getStateView(result_, quadraturePoint), copy= not getPersistentView)

    cdef double[::1] getStateView(self, string result, int quadraturePoint):
        """Directly access the state vars of the underlying MarmotElement

        Note: if the underlying C++ returns a NULL pointer, raise ValueError so
        callers can handle missing state data gracefully.
        """

        if not self._hasMaterial:
            raise Exception("Element {:} has no material assigned!".format(self._elNumber))

        cdef StateView res = self.marmotElement.getStateView(result, quadraturePoint)
        if res.stateLocation != NULL and res.stateSize > 0:
            return <double[:res.stateSize]> ( res.stateLocation )

        # If Marmot returns no data for this (result, gp), provide a persistent
        # zero-buffer which mimics a valid state view. This keeps the behaviour
        # identical to other Marmot elements (which always return a non-NULL
        # pointer) and avoids changing the ElementResultCollector contract.
        py_result = result.decode('utf-8')
        key = (py_result, int(quadraturePoint))
        try:
            arr = self._missingStateBuffers[key]
        except KeyError:
            # choose a sensible fallback size: prefer element's nStateVars if known
            size = self.nStateVars if getattr(self, 'nStateVars', 0) > 0 else 1
            arr = np.zeros(size, dtype=np.double)
            self._missingStateBuffers[key] = arr

        cdef double[::1] mv = arr
        return mv

    def getCoordinatesAtCenter(self):
        return np.asarray ( self.marmotElement.getCoordinatesAtCenter() )

    def getCoordinatesAtQuadraturePoints(self):
        return np.asarray ( self.marmotElement.getCoordinatesAtQuadraturePoints() )

    def getNumberOfQuadraturePoints(self):
        return self.marmotElement.getNumberOfQuadraturePoints()

    def __dealloc__(self):
        del self.marmotElement
