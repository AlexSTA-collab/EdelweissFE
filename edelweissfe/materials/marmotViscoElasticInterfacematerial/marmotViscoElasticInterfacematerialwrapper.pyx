import numpy as np

cimport cython
cimport libcpp.cast
cimport numpy as np

from edelweissfe.utils.exceptions import CutbackRequest

from libc.stdlib cimport free, malloc
from libcpp.memory cimport allocator, make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef class MarmotViscoElasticInterfaceMaterialWrapper:

    cdef LinearViscoElasticInterface* _theMarmotInterfaceMaterialInstance
    cdef double[::1] _stateVars
    cdef double[::1] _materialProperties
    cdef int _materialID

    def __init__(self, double[::1] materialProperties, int materialID):
        pass


    def __cinit__(self, double[::1] materialProperties, int materialID):
        """This C-level method is responsible for actually instancing the Marmot material.
        In contrast to the __init__ method, it is guaranteed that this method is called only once.

        Parameters
        ----------
        materialProperties : np.ndarray
            The material properties of the material.
        materialID : int
            The material ID of the material (for debugging purposes).
        """


        self._materialProperties = materialProperties
        self._materialID = materialID
        cdef int nMaterialProperties = len(materialProperties)
        self._theMarmotInterfaceMaterialInstance = new LinearViscoElasticInterface(&self._materialProperties[0], nMaterialProperties, self._materialID)

    def computeStress(self,
                      double[::1]  force,
                      double[:,::1] surface_stress, # assume RowMajor order = C order, i.e. the last index is the fastest changing index
                      double[:,::1] dStress_dStrain, # assume RowMajor order = C order, i.e. the last index is the fastest changing index
                      double[::1] dU,
                      double[::1] dSurface_strain,
                      double[::1] normal,
                      double  timeOld,
                      double  dT):

        cdef double pNewDT
        pNewDT = 1e36

        self._theMarmotInterfaceMaterialInstance.computeStress(
            &force[0],
            &surface_stress[0,0],
            &dStress_dStrain[0,0],
            &dU[0],
            &dSurface_strain[0],
            &normal[0],
            &timeOld,
            dT,
            pNewDT) 
        if pNewDT < 1.0:
            raise CutbackRequest("Material requests for a cutback!", pNewDT)

    def getNumberOfRequiredStateVars(self,):

        cdef int numberOfRequiredStateVarsMaterial = self._theMarmotInterfaceMaterialInstance.getNumberOfRequiredStateVars()
        return numberOfRequiredStateVarsMaterial

    def assignStateVars(self, double[::1] stateVars):

        self._stateVars = stateVars

        self._theMarmotInterfaceMaterialInstance.assignStateVars(&self._stateVars[0], len(self._stateVars))

    def getResultArray(self, result, getPersistentView=True):

        cdef string result_ =  result.encode('UTF-8')

        cdef StateView res = self._theMarmotInterfaceMaterialInstance.getStateView(result_)

        cdef double[::1] theView = <double[:res.stateSize]> ( res.stateLocation )

        return np.array(  theView, copy= not getPersistentView)

    def __dealloc__(self):
        # this is the destructor
        del self._theMarmotInterfaceMaterialInstance

def test_MarmotViscoElasticInterfaceMaterialWrapper():

    print("Testing MarmotInterfaceMaterialWrapper...") 
    marmotMaterialViscoElasticInterfaceWrapper = MarmotViscoElasticInterfaceMaterialWrapper(np.array([1.0,  0.3, 1.0,  0.3, 1e8,  0.3,  1e-5, 1, 2, 1, 0.01, 1, 2, 1, 0.01, 86400]), 2)

    force = np.array([100., 200., 300.])
    surface_stress = np.ones((3,3))
    dStress_dStrain = np.ones((21,21)) * 2 
    dU = np.ones(6) * 3
    dSurface_strain = np.ones(18) * 4
    normal = np.ones(3) * 5
    timeOld = 0.0
    dT = 0.1

    marmotMaterialViscoElasticInterfaceWrapper.computeStress(force, surface_stress, dStress_dStrain, dU, dSurface_strain, normal, timeOld, dT)

    


