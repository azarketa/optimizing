"""
@author: azarketa
This file includes the classes and methods employed for Wells turbine simulations and optimization techniques.
"""

##########################################################################################################################################
##########################################################################################################################################
###################################################################TODO###################################################################
##########################################################################################################################################
##########################################################################################################################################

# Deal with the warning included in the documentation of the 'twist_optimization()' method.
# Make the 'get_localParam_gradDir()' and 'searchParamMax_uponGradParam()' methods generic, so that they may be applied, independently, to the 'pitch_optimization()' and the 'twist_optimization()' methods. In the current version, they have been developed for their inclusion in the 'pitch_optimization()' method; extending the functionality to the 'twist_optimization()' method may require using callback functions in the input parameters.
# Fit the K constant for pitch-dependent dimensionaless turbine curves. The collapse of the dimensionless turbine with respect to the operational parmaeters 8i.e. rotational speed) is valid for unpitched configurations; it does not occur the same hwen the pitch is varied. Currently, it is not known whether it is possible to derive a functional relation between pitch-varying Wells curves and a slope of the psi-phi dimensionless curve.
# The twist optimization method, when launched in the 'hubtip' mode, does not discriminate between twist distributions that are monotonously increasing or decreasing, and those which are not. This sould be made explicit in an updated version.

##########################################################################################################################################
##########################################################################################################################################
############################################################IMPORTING PACKAGES############################################################
##########################################################################################################################################
##########################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, simpson
import scipy.special as scsp
import copy
from typing import Union
import itertools
import re as regex
import pickle
import abc

# Check if running in Jupyter Notebook
if 'ipykernel' in sys.modules:
    import tqdm.notebook as tqdm
else:
    import tqdm

import mathTools as mt
from turboTools import polars_postproc_load as pload
from GAOptTools import pygad
from airfoilTools import geomConstants as geCnt
from expTestTools import expGeomDataClasses as expGeCl
from expTestTools import expWindTunnelDataClasses as expWTCl
from expTestTools import expWindTunnelConstants as expWTCnt
from cfdTools import cfdDataClasses as cfdCl
from cfdTools import cfdConstants as cfdCnt
from turboTools import turbineConstants as turbCnt

##########################################################################################################################################
##########################################################################################################################################
##############################################################LOADING POLARS##############################################################
##########################################################################################################################################
##########################################################################################################################################

filepath = __file__.split("turbineClasses.py")[0].replace("\\", "/") + "/"
polars = pload.polar_database_load(filepath=filepath, pick=False)
cpto_pstd = pload.cpto_pstd_database_load(filepath=filepath, pick=False)
cpto_qstd = pload.cpto_pstd_database_load(filename='Cpto-qstd/Cpto-qstd', filepath=filepath, pick=False)

##########################################################################################################################################
##########################################################################################################################################
#################################################TURBINE-CLASS-RELATED CLASS DEFINITIONS##################################################
##########################################################################################################################################
##########################################################################################################################################

#--------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------CLASS anlyDataset------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------#
class anlyDataset(abc.ABC):
    '''Class inheriting from abc.ABC for empty class creation, intending to store analytically-derived datasets.'''
    pass

#--------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------CLASS anlyData--------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------#
class anlyData(abc.ABC):
    '''Class inheriting from abc.ABC for empty class creation, intending to store analytically-derived data.'''
    pass

#--------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------CLASS basePPClass------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------#
class basePPClass(abc.ABC):
    '''Class inheriting from abc.ABC for empty class creation.'''
    pass

#--------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------CLASS geom------------------------------------------------#             
#--------------------------------------------------------------------------------------------------------------------#
class geom:
    '''Geometrical parameters of a turbine stage.'''
    
    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------DEFINING METHODS-------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    
    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------__init__() METHOD------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    def __init__(self,
                 ttype: turbCnt.TYPES=turbCnt.TYPES.Wells,
                 N: int=50,
                 rcas: float=0.25,
                 hub_to_tip_ratio: float=0.75,
                 chord: Union[list, np.ndarray, float]=0.117,
                 angpitch: Union[list, np.ndarray, float]=0,
                 airfoil: Union[geCnt.PROFILES, np.ndarray]=geCnt.PROFILES.NACA0015,
                 tip_percent: float=0.5,
                 Z: int=7) -> None:
        '''Initializes the turbine with the provided input values.
        
        **parameters**:
        :param ttype: a turbCnt.TYPES enum constant value, specifying if the system is either a 'Wells' or an 'Impulse' turbine. Default is 'Wells'.
        :param N: number of blade elements (discretization). Default is 50.
        :param rcas: casing radius. Default is 0.25.
        :param hub_to_tip_ratio: hub to tip ratio value. Default is 0.75.
        :param chord: chord value. It is possible to provide an array of length N establishing the chord value at each blade element. Default is 0.117, which means that the chord is constant along the radius.
        :param angpitch: angular pitch value. It is possible to provide an array of length N establishing the angular pitch value at each blade element. Default is 0, which means that no pitch is given to the elements.
        :param airfoil: airfoil geometry that corresponds to chord. It is possible to provide an array of length N establishing the airfoil geometry at each blade element. Default is NACA0015, which means that the whole blade is considered to own a single geometry, namely the one of the NACA0015.
        :param tip_percent: tip gap value, in terms of chord percentage. Default is 0.5%.
        :param Z: number of blades. Default is 7.        
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------DECLARING ATTRIBUTES----------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------# 

        # Conditional for determining which turbine type is instantiated.
        # If a 'Wells' turbine is instantiated:
        if ttype == turbCnt.TYPES.Wells:
            #####
            ##### Setting provided values to attributes.
            #####            
            # Type of turbine.
            self._ttype = turbCnt.TYPES.Wells
            # Casing radius of turbine.
            self._rcas = rcas
            # Hub-to-tip ratio.
            self._hub_to_tip_ratio = hub_to_tip_ratio
            # Tip percentage.
            self._tip_percent = tip_percent
            # Blade number.
            self._Z = Z
            #####
            ##### Setting calculated values to attributes.
            #####                                 
            # Setting tip radius.
            if type(chord) in [float, np.float64, np.float32]:
                self._rtip = self.rcas - chord*tip_percent/100
            else:
                self._rtip = self.rcas - chord[-1]*tip_percent/100
            # Setting hub radius.
            self._rhub = self.rtip*hub_to_tip_ratio
            # Setting differential radial element.
            self._dr = (self._rtip - self._rhub)/N            
            # Setting radial array.
            self._r = np.array([self.rhub + i*self._dr for i in range(N + 1)])  
            # Chord.
            # The conditional structure is intended to cast a single float input value to an array
            # of length equal to the number of blade elements with entries coinciding with the single
            # float value, or to set the 'chord' property of the object equal to the input array or list
            # instead.
            if type(chord) in [float, np.float64, np.float32]:
                self._chord = np.array([chord for _ in self.r])
            elif type(chord) == list:
                self._chord = np.array(chord)
            else:
                self._chord = chord
            # Pitch.
            # The conditional structure is intended to cast a single float input value to an array
            # of length equal to the number of blade elements with entries coinciding with the single
            # float value, or to set the 'angpitch' property of the object equal to the input array or list
            # instead.
            if type(angpitch) in [float, np.float64, np.float32]:
                self._angpitch = np.array([angpitch for _ in self.r])
            elif type(angpitch) == list:
                self._angpitch = np.array(angpitch)
            else:
                self._angpitch = angpitch                            
            # Airfoil.
            # The conditional structure is intended to cast a single string input value to a
            # list of length equal to the number of blade elements coinciding with the single string
            # value, or to set the 'airfoil' property of the object equal to the input array or list
            # instead.            
            if isinstance(airfoil, geCnt.PROFILES):
                self._airfoil = np.array([airfoil for _ in self.r])
            else:        
                self._airfoil = [_ for _ in airfoil]            
            # Airfoil coords and differential, element-wise areas.
            if isinstance(airfoil, geCnt.PROFILES):
                # Instantiating airfoil coordinates file name.
                filestr = airfoil.name + ".txt"                
                # Checking for existence of airfoil file.
                if filestr in os.listdir("/".join([filepath, "COORDS/"])):
                    # If the file exists, then load coordinates.
                    self._airfoil_coords = [np.loadtxt(open("/".join([filepath, "COORDS", filestr])).readlines(), usecols=(0, 1), unpack=True) for _ in self.r]
                else:
                    # Otherwise, set coordinates to [[0], [0]] point.
                    self._airfoil_coords = [[[0], [0]] for _ in self.r]                    
            else:
                # Instantiating list for storing airfoils' coordinates.                
                self._airfoil_coords = list()
                for _ in airfoil:
                    # Instantiating airfoil coordinates file name.
                    filestr = _.name + ".txt"                    
                    # Checking for existence of airfoil file.
                    if filestr in os.listdir("/".join([filepath, "COORDS/"])):   
                        # If the file exists, then load coordinates.
                        self._airfoil_coords.append(np.loadtxt(open("/".join([filepath, "COORDS", filestr])).readlines(), usecols=(0, 1), unpack=True))
                    else:
                        # Otherwise, set coordinates to [[0], [0]] point.
                        self._airfoil_coords.append([[0], [0]])
            # Rescaling airfoils according to elements' chord values.
            if type(self._chord) == np.ndarray:
                for e, _ in enumerate(self._airfoil_coords):
                    _[0] *= self._chord[e]
                    _[1] *= self._chord[e]
            else:
                for e, _ in enumerate(self._airfoil_coords):
                    _[0] *= self._chord
                    _[1] *= self._chord
            # Computing differential, element-wise areas by calculating the perimeter of the airfoils from the arclength and multiplying them with dr.
            self._dA = np.array([sum(np.sqrt(_[0]**2 + _[1]**2))*self._dr for _ in self._airfoil_coords])
            # Setting tip percent value.
            self._tip_c = self.tip_percent*self.chord[-1]*0.01
            # Setting pitch following 1995Raghunatan.
            self._pitch = 2*np.pi*self.rtip/self.Z - self.chord[-1]
            # Setting aspect ratio.
            self._AR = (self.rtip - self.rhub)/self.chord[-1]              
            # Setting solidity array.
            self._sigma = self._Z*self._chord/(2*np.pi*self._r)
        # If an 'Impulse' turbine is instantiated.
        elif ttype == turbCnt.TYPES.Impulse:
            # TODO: Not implemented yet. Raising 'NotImplementedError'.
            raise NotImplementedError()

    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------DECLARING PROPERTIES-----------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    
    # Declaration of get property rcas for casing radius.
    @property
    def rcas(self) -> float:
        return self._rcas
    # Declaration of get property hub_to_tip_ratio for hub-to-tip ratio.
    @property
    def hub_to_tip_ratio(self) -> float:
        return self._hub_to_tip_ratio
    # Declaration of get property chord for chord.
    @property
    def chord(self) -> Union[np.ndarray, float]:
        return self._chord
    # Declaration of get property angpitch for angular pitch.
    @property
    def angpitch(self) -> Union[np.ndarray, float]:
        return self._angpitch    
    # Declaration of get property airfoil for airfoil geometry.    
    @property
    def airfoil(self) -> Union[np.ndarray, str]:
        return self._airfoil    
    # Declaration of get property airfoil_coords for airfoil coordinates.
    def airfoil_coords(self) -> np.ndarray:
        return self._airfoil_coords
    # Declaration of get property dA for differential, element-wise areas.
    @property
    def dA(self) -> np.ndarray:
        return self._dA
    # Declaration of get property tip_percent for tip percent.
    @property
    def tip_percent(self) -> float:
        return self._tip_percent
    # Declaration of get property Z for number of blades.
    @property
    def Z(self) -> int:
        return self._Z
    # Declaration of get property tip_c for tip clearance.
    @property
    def tip_c(self) -> float:
        return self._tip_c
    # Declaration of get property rtip for tip radius.
    @property
    def rtip(self) -> float:
        return self._rtip
    # Declaration of get property rhub for hub radius.
    @property
    def rhub(self) -> float:
        return self._rhub
    # Declaration of get property pitch for pitch.
    @property
    def pitch(self) -> float:
        return self._pitch
    # Declaration of get property AR for aspect ratio.
    @property
    def AR(self) -> float:
        return self._AR
    # Declaration of get property r for radial array.
    @property
    def r(self) -> np.ndarray:
        return self._r
    # Declaration of get property sigma for solidity.
    @property
    def sigma(self) -> float:
        return self._sigma
    # Declaration of get property dr for radial inter-element spacing.
    @property
    def dr(self) -> float:
        return self._dr
       
    # Declaration of set property rcas for casing radius.
    @rcas.setter 
    def rcas(self, value: float) -> None:
        self._rcas = value
        self.__init__(ttype=self._ttype, rcas=self._rcas, hub_to_tip_ratio=self._hub_to_tip_ratio,
                     chord=self._chord, angpitch=self._angpitch, airfoil=self._airfoil, tip_percent=self._tip_percent, Z=self._Z)
        return
    # Declaration of set property hub_to_tip_ratio for hub-to-tip ratio.
    @hub_to_tip_ratio.setter
    def hub_to_tip_ratio(self, value :float) -> None:
        self._hub_to_tip_ratio = value
        self.__init__(ttype=self._ttype, rcas=self._rcas, hub_to_tip_ratio=self._hub_to_tip_ratio,
                     chord=self._chord, angpitch=self._angpitch, airfoil=self._airfoil, tip_percent=self._tip_percent, Z=self._Z)
        return
    # Declaration of set property chord for chord.
    @chord.setter
    def chord(self, value: Union[list, np.ndarray, float]) -> None:
        self._chord = value
        self.__init__(ttype=self._ttype, rcas=self._rcas, hub_to_tip_ratio=self._hub_to_tip_ratio,
                     chord=self._chord, angpitch=self._angpitch, airfoil=self._airfoil, tip_percent=self._tip_percent, Z=self._Z)
        return
    # Declaration of set property chord for angular pitch.
    @angpitch.setter
    def angpitch(self, value: Union[list, np.ndarray, float]) -> None:
        self._angpitch = value
        self.__init__(ttype=self._ttype, rcas=self._rcas, hub_to_tip_ratio=self._hub_to_tip_ratio,
                     chord=self._chord, angpitch=self._angpitch, airfoil=self._airfoil, tip_percent=self._tip_percent, Z=self._Z)
        return    
    # Declaration of set property airfoil for airfoil geometry.
    @airfoil.setter
    def airfoil(self, value: Union[list, np.ndarray, geCnt.PROFILES]) -> None:        
        self._airfoil = value        
        self.__init__(ttype=self._ttype, rcas=self._rcas, hub_to_tip_ratio=self._hub_to_tip_ratio,
                     chord=self._chord, angpitch=self._angpitch, airfoil=self._airfoil, tip_percent=self._tip_percent, Z=self._Z)
        return
    # Declaration of set property tip_percent for tip percent.
    @tip_percent.setter
    def tip_percent(self, value: float) -> None:
        self._tip_percent = value
        self.__init__(ttype=self._ttype, rcas=self._rcas, hub_to_tip_ratio=self._hub_to_tip_ratio,
                     chord=self._chord, angpitch=self._angpitch, airfoil=self._airfoil, tip_percent=self._tip_percent, Z=self._Z)
        return
    # Declaration of set property Z for number of blades.
    @Z.setter
    def Z(self, value: int) -> None:
        self._Z = value
        self.__init__(ttype=self._ttype, rcas=self._rcas, hub_to_tip_ratio=self._hub_to_tip_ratio,
                     chord=self._chord, angpitch=self._angpitch, airfoil=self._airfoil, tip_percent=self._tip_percent, Z=self._Z)
        return
    
#--------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------CLASS flow------------------------------------------------#             
#--------------------------------------------------------------------------------------------------------------------#     
class flow:
    '''Flow parameters entering a turbine stage.'''
    
    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------DEFINING METHODS-------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    
    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------__init__() METHOD------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    def __init__(self,
                 p: float=101325,
                 T: float=288.15,
                 R: float=287.058,
                 nu: float=1.81e-5) -> None:
        '''Initializes a given instance of the input parameters entering a turbine.
        
        **parameters**:
        :param p: atmospheric pressure in [Pa]. Default value is 101325 [Pa].
        :param T: ambient temperature in [K]. Default value is 288.15 [K].
        :param R: gas constant in [J/(kg·K)]. Default value is 287.058 [J/(kg·K)].
        :param nu: gas viscosity in [kg/(m·s)]. Default value is 1.81e-5 [kg/(m·s)].       
        '''        
        
        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------DECLARING ATTRIBUTES-----------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
    
        # Atmospheric pressure.
        self._p = p
        # Ambient temperature.
        self._T = T
        # Gas constant.
        self._R = R
        # Fluid viscosity.
        self._nu = nu
        # Calculating density assuming ideal gas. Setting it to 'rho' attribute.
        self._rho = p/(T*R)

    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------DECLARING PROPERTIES-----------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#                
        
    # Declaration of get property p for atmospheric pressure.
    @property
    def p(self) -> float:
        return self._p
    # Declaration of get property T for ambient temperature.
    @property
    def T(self) -> float:
        return self._T
    # Declaration of get property R for gas (air) constant.
    @property
    def R(self) -> float:
        return self._R
    # Declaration of get property nu for gas (air) viscosity.
    @property
    def nu(self) -> float:
        return self._nu
    # Declaration of get property rho for gas (air) density.
    @property
    def rho(self) -> float:
        return self._rho

    # Declaration of set property p for atmospheric pressure.
    @p.setter 
    def p(self, value: float) -> None:
        self.__init__(p=value, T=self._T, R=self._R, nu=self._nu)
        return
    # Declaration of set property T for ambient temperature.
    @T.setter
    def T(self, value: float) -> None:
        self.__init__(p=self._P, T=value, R=self._R, nu=self._nu)
        return
    # Declaration of set property R for gas (air) constant.
    @R.setter
    def R(self, value: float) -> None:
        self.__init__(p=self._P, T=self._T, R=value, nu=self._nu)
        return
    # Declaration of set property nu for gas (air) viscosity.
    @nu.setter
    def nu(self, value: float) -> None:
        self.__init__(p=self._P, T=self._T, R=self._R, nu=value)
        return
    
#--------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------CLASS velocity_triangle---------------------------------------------#             
#--------------------------------------------------------------------------------------------------------------------#
class velocity_triangle:
    '''Spanwise velocity triangle of a turbine stage.'''
    
    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------DEFINING METHODS-------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    
    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------__init__() METHOD------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    def __init__(self, N: int=50) -> None:
        '''Initializes a given instance of a velocity triangle.
        
        **parameters**, **return**, **return type**:
        :param N: number of blade elements (discretization). Default is 50.      
        '''        
        
        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------DECLARING ATTRIBUTES-----------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        # Absolute angle.
        self._alpha = np.zeros(N + 1)
        # Axial component of absolute velocity.
        self._vx = np.zeros(N + 1)
        # Tangential component of absolute velocity.
        self._vtheta = np.zeros(N + 1)
        # Absolute velocity magnitude.
        self._v = np.zeros(N + 1)
        # Tangential velocity.
        self._U = np.zeros(N + 1)
        # Axial component of relative velocity.
        self._wx = np.zeros(N + 1)
        # Tangential component of relative velocity.
        self._wtheta = np.zeros(N + 1)
        # Relative velocity magnitude.
        self._w = np.zeros(N + 1)
        # Relative angle.
        self._beta = np.zeros(N + 1)
        # Combination of relative angle and pitch.
        self._gamma = np.zeros(N + 1)
        # Flow coefficient.
        self._phi = np.zeros(N + 1)
        # Differential flow-rate.
        self._dq = np.zeros(N + 1)
        # Flow-rate.
        self._q = 0
        # Non-dimensional flow-rate.
        self._Phi = 0

    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------DECLARING PROPERTIES-----------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#                
        
    # Declaration of get property alpha for absolute velocity angle.
    @property
    def alpha(self) -> np.ndarray:
        return self._alpha
    # Declaration of get property vx for axial component of absolute velocity.
    @property
    def vx(self) -> np.ndarray:
        return self._vx
    # Declaration of get property vtheta for tangential component of absolute velocity.
    @property
    def vtheta(self) -> np.ndarray:
        return self._vtheta
    # Declaration of get property v for magnitude of absolute velocity.
    @property
    def v(self) -> np.ndarray:
        return self._v
    # Declaration of get property U for tangential velocity.
    @property
    def U(self) -> np.ndarray:
        return self._U
    # Declaration of get property wx for axial component of relative velocity.
    @property
    def wx(self):
        return self._wx
    # Declaration of get property wtheta for tangential component of relative velocity.
    @property
    def wtheta(self) -> np.ndarray:
        return self._wtheta
    # Declaration of get property w for magnitude of relative velocity.
    @property
    def w(self) -> np.ndarray:
        return self._w
    # Declaration of get property beta for relative velocity angle.
    @property
    def beta(self) -> np.ndarray:
        return self._beta
    # Declaration of get property gamma for relative velocity angle.
    @property
    def gamma(self) -> np.ndarray:
        return self._gamma        
    # Declaration of get property phi for flow coefficient.
    @property
    def phi(self) -> np.ndarray:
        return self._phi
    # Declaration of get property dq for differential flow-rate value.
    @property
    def dq(self) -> np.ndarray:
        return self._dq
    # Declaration of get property q for flow-rate value.
    @property
    def q(self) -> float:
        return self._q
    # Declaration of get property Phi for dimensionless flow-rate value.
    @property
    def Phi(self) -> float:
        return self._Phi
    
    # Declaration of set property alpha for absolute velocity angle.
    @alpha.setter 
    def alpha(self, value: np.ndarray) -> None:
        assert len(value) == len(self._alpha), 'Provide an array of size ' + len(self._alpha) + '; current array has size ' + str(len(value))
        self._alpha = value
        return
    # Declaration of set property vx for axial component of absolute velocity.
    @vx.setter
    def vx(self, value: np.ndarray) -> None:
        assert len(value) == len(self._vx), 'Provide an array of size ' + len(self._vx) + '; current array has size ' + str(len(value))
        self._vx = value
        return
    # Declaration of set property vtheta for tangential component of absolute velocity.
    @vtheta.setter
    def vtheta(self, value: np.ndarray) -> None:
        assert len(value) == len(self._vtheta), 'Provide an array of size ' + len(self._vtheta) + '; current array has size ' + str(len(value))
        self._vtheta = value
        return
    # Declaration of set property v for magnitude of absolute velocity.
    @v.setter
    def v(self, value: np.ndarray) -> None:
        assert len(value) == len(self._v), 'Provide an array of size ' + len(self._v) + '; current array has size ' + str(len(value))
        self._v = value
        return
    # Declaration of set property U for tangential velocity.
    @U.setter 
    def U(self, value: np.ndarray) -> None:
        assert len(value) == len(self._U), 'Provide an array of size ' + len(self._U) + '; current array has size ' + str(len(value))
        self._U = value
        return
    # Declaration of set property wx for axial component of relative velocity.
    @wx.setter
    def wx(self, value: np.ndarray) -> None:
        assert len(value) == len(self._wx), 'Provide an array of size ' + len(self._wx) + '; current array has size ' + str(len(value))
        self._wx = value
        return
    # Declaration of set property wtheta for tangential component of relative velocity.
    @wtheta.setter
    def wtheta(self, value: np.ndarray) -> None:
        assert len(value) == len(self._wtheta), 'Provide an array of size ' + len(self._wtheta) + '; current array has size ' + str(len(value))
        self._wtheta = value
        return
    # Declaration of set property w for magnitude of relative velocity.
    @w.setter
    def w(self, value: np.ndarray) -> None:
        assert len(value) == len(self._w), 'Provide an array of size ' + len(self._w) + '; current array has size ' + str(len(value))
        self._w = value
        return        
    # Declaration of set property beta for relative velocity angle.
    @beta.setter
    def beta(self, value: np.ndarray) -> None:
        assert len(value) == len(self._beta), 'Provide an array of size ' + len(self._beta) + '; current array has size ' + str(len(value))
        self._beta = value
        return
    # Declaration of set property gamma for combination of relative velocity angle and pitch.
    @gamma.setter
    def gamma(self, value: np.ndarray) -> None:
        assert len(value) == len(self._gamma), 'Provide an array of size ' + len(self._gamma) + '; current array has size ' + str(len(value))
        self._gamma = value
        return        
    # Declaration of set property phi for flow coefficient.
    @phi.setter
    def phi(self, value: np.ndarray) -> None:
        assert len(value) == len(self._phi), 'Provide an array of size ' + len(self._phi) + '; current array has size ' + str(len(value))
        self._phi = value
        return
    # Declaration of set property dq for differential flow-rate value.
    @dq.setter
    def dq(self, value: np.ndarray) -> None:
        assert len(value) == len(self._dq), 'Provide an array of size ' + len(self._dq) + '; current array has size ' + str(len(value))
        self._dq = value
        return
    # Declaration of set property q for flow-rate value.
    @q.setter
    def q(self, value: float) -> None:        
        self._q = value
        return
    # Declaration of set property Phi for dimensionless flow-rate value.
    @Phi.setter
    def Phi(self, value: float) -> None:
        self._Phi = value
        return
    
#--------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------CLASS loads-------------------------------------------------#             
#--------------------------------------------------------------------------------------------------------------------#
class loads:
    '''Spanwise load coefficients on a turbine stage.'''
    
    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------DEFINING METHODS-------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    
    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------__init__() METHOD------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    def __init__(self, N: int=50):
        '''Initializes a given instance of a class containing loads and coefficients.
        
        **parameters**, **return**, **return type**:
        :param N: number of blade elements (discretization). Default is 50.
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------DECLARING ATTRIBUTES-----------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        # Reynolds number.
        self._Re = np.zeros(N + 1)
        # Lift coefficient.
        self._cl = np.zeros(N + 1)
        # Differential lift force.
        self._dl = np.zeros(N + 1)
        # Drag coefficient.
        self._cd = np.zeros(N + 1)
        # Differential drag force.
        self._dd = np.zeros(N + 1)
        # Induced drag coefficient.
        self._cdtc = np.zeros(N + 1)
        # Differential induced drag force.
        self._ddtc = np.zeros(N + 1)        
        # Axial force coefficient.
        self._cx = np.zeros(N + 1)
        # Differential axial force.
        self._dx = np.zeros(N + 1)        
        # Tangential force coefficient.
        self._ctheta = np.zeros(N + 1)
        # Differential tangential force.
        self._dtheta = np.zeros(N + 1)        
        # Interference-corrected axial force coefficient.
        self._cxCIF = np.zeros(N + 1)
        # Differential interference-corrected axial force.
        self._dxCIF = np.zeros(N + 1)        
        # Interference-corrected tangential force coefficient.
        self._cthetaCIF = np.zeros(N + 1)
        # Differential interference-corrected tangential force.
        self._dthetaCIF = np.zeros(N + 1)        
        # Input coefficient.
        self._cinput = 0
        # Torque coefficient.
        self._ctau = 0

    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------DECLARING PROPERTIES-----------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#                
        
    # Declaration of get property Re for Reynolds numbers.
    @property
    def Re(self) -> np.ndarray:
        return self._Re
    # Declaration of get property cl for lift coefficients.
    @property
    def cl(self) -> np.ndarray:
        return self._cl
    # Declaration of get property dl for differential lift forces.
    @property
    def dl(self) -> np.ndarray:
        return self._dl      
    # Declaration of get property cd for drag coefficients.
    @property
    def cd(self) -> np.ndarray:
        return self._cd
    # Declaration of get property dd for differential drag forces.
    @property
    def dd(self) -> np.ndarray:
        return self._dd    
    # Declaration of get property cdtc for induced drag coefficients.
    @property
    def cdtc(self) -> np.ndarray:
        return self._cdtc
    # Declaration of get property ddtc for differential induced drag forces.
    @property
    def ddtc(self) -> np.ndarray:
        return self._ddtc    
    # Declaration of get property cx for axial coefficients.
    @property
    def cx(self) -> np.ndarray:
        return self._cx
    # Declaration of get property dx for differential axial forces.
    @property
    def dx(self) -> np.ndarray:
        return self._dx    
    # Declaration of get property ctheta for tangential coefficients.
    @property
    def ctheta(self) -> np.ndarray:
        return self._ctheta
    # Declaration of get property dtheta for differential tangential forces.
    @property
    def dtheta(self) -> np.ndarray:
        return self._dtheta    
    # Declaration of get property cxCIF for corrected axial coefficients.
    @property
    def cxCIF(self) -> np.ndarray:
        return self._cxCIF
    # Declaration of get property dxCIF for corrected differential axial forces.
    @property
    def dxCIF(self) -> np.ndarray:
        return self._dxCIF
    # Declaration of get property cthetaCIF for corrected tangential coefficients.    
    @property
    def cthetaCIF(self) -> np.ndarray:
        return self._cthetaCIF
    # Declaration of get property dthetaCIF for corrected differential tangential coefficients.    
    @property
    def dthetaCIF(self) -> np.ndarray:
        return self._dthetaCIF    
    # Declaration of get property cinput for input coefficients.
    @property
    def cinput(self) -> np.ndarray:
        return self._cinput
    # Declaration of get property ctau for torque coefficients.
    @property
    def ctau(self) -> np.ndarray:
        return self._ctau
    
    # Declaration of set property Re for Reynolds number.
    @Re.setter 
    def Re(self, value: np.ndarray) -> None:
        assert len(value) == len(self._Re), 'Provide an array of size ' + len(self._Re) + '; current array has size ' + str(len(value))
        self._Re = value
        return
    # Declaration of set property cl for lift coefficients.
    @cl.setter
    def cl(self, value: np.ndarray) -> None:
        assert len(value) == len(self._cl), 'Provide an array of size ' + len(self._cl) + '; current array has size ' + str(len(value))
        self._cl = value
        return
    # Declaration of set property dl for lift forces.
    @dl.setter
    def dl(self, value: np.ndarray) -> None:
        assert len(value) == len(self._dl), 'Provide an array of size ' + len(self._dl) + '; current array has size ' + str(len(value))
        self._dl = value
        return    
    # Declaration of set property cd for drag coefficients.
    @cd.setter
    def cd(self, value: np.ndarray) -> None:
        assert len(value) == len(self._cd), 'Provide an array of size ' + len(self._cd) + '; current array has size ' + str(len(value))
        self._cd = value
        return
    # Declaration of set property dd for drag forces.
    @dd.setter
    def dd(self, value: np.ndarray) -> None:
        assert len(value) == len(self._dd), 'Provide an array of size ' + len(self._dd) + '; current array has size ' + str(len(value))
        self._dd = value
        return    
    # Declaration of set property cdtc for induced drag coefficients.
    @cdtc.setter
    def cdtc(self, value: np.ndarray) -> None:
        assert len(value) == len(self._cdtc), 'Provide an array of size ' + len(self._cdtc) + '; current array has size ' + str(len(value))
        self._cdtc = value
        return
    # Declaration of set property ddtc for induced drag loads.
    @ddtc.setter
    def ddtc(self, value: np.ndarray) -> None:
        assert len(value) == len(self._ddtc), 'Provide an array of size ' + len(self._ddtc) + '; current array has size ' + str(len(value))
        self._ddtc = value
        return    
    # Declaration of set property cx for axial coefficients.
    @cx.setter
    def cx(self, value: np.ndarray) -> None:
        assert len(value) == len(self._cx), 'Provide an array of size ' + len(self._cx) + '; current array has size ' + str(len(value))
        self._cx = value
        return
    # Declaration of set property dx for axial loads.
    @dx.setter
    def dx(self, value: np.ndarray) -> None:
        assert len(value) == len(self._dx), 'Provide an array of size ' + len(self._dx) + '; current array has size ' + str(len(value))
        self._dx = value
        return    
    # Declaration of set property ctheta for tangential coefficients.
    @ctheta.setter 
    def ctheta(self, value: np.ndarray) -> None:
        assert len(value) == len(self._ctheta), 'Provide an array of size ' + len(self._ctheta) + '; current array has size ' + str(len(value))
        self._ctheta = value
        return
    # Declaration of set property dtheta for tangential coefficients.
    @dtheta.setter 
    def dtheta(self, value: np.ndarray) -> None:
        assert len(value) == len(self._dtheta), 'Provide an array of size ' + len(self._dtheta) + '; current array has size ' + str(len(value))
        self._dtheta = value
        return    
    # Declaration of set property cxCIF for corrected axial coefficients.
    @cxCIF.setter
    def cxCIF(self, value: np.ndarray) -> None:
        assert len(value) == len(self._cxCIF), 'Provide an array of size ' + len(self._cxCIF) + '; current array has size ' + str(len(value))
        self._cxCIF = value
        return
    # Declaration of set property dxCIF for corrected axial coefficients.
    @dxCIF.setter
    def dxCIF(self, value: np.ndarray) -> None:
        assert len(value) == len(self._dxCIF), 'Provide an array of size ' + len(self._dxCIF) + '; current array has size ' + str(len(value))
        self._dxCIF = value
        return    
    # Declaration of set property cthetaCIF for corrected tangential coefficients.
    @cthetaCIF.setter
    def cthetaCIF(self, value: np.ndarray) -> None:
        assert len(value) == len(self._cthetaCIF), 'Provide an array of size ' + len(self._cthetaCIF) + '; current array has size ' + str(len(value))
        self._cthetaCIF = value
        return
    # Declaration of set property dthetaCIF for corrected tangential coefficients.
    @dthetaCIF.setter
    def dthetaCIF(self, value: np.ndarray) -> None:
        assert len(value) == len(self._dthetaCIF), 'Provide an array of size ' + len(self._dthetaCIF) + '; current array has size ' + str(len(value))
        self._dthetaCIF = value
        return    
    # Declaration of set property cinput for input coefficients.
    @cinput.setter
    def cinput(self, value: np.ndarray) -> None:
        self._cinput = value
        return  
    # Declaration of set property ctau for torque coefficients.
    @ctau.setter
    def ctau(self, value: np.ndarray) -> None:
        self._ctau = value
        return
    
#--------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------CLASS eout-----------------------------------------------#             
#--------------------------------------------------------------------------------------------------------------------#
class eout:
    '''Energetic outpus of a turbine stage.'''
    
    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------DEFINING METHODS-------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    
    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------__init__() METHOD------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    def __init__(self, N: int=50):
        '''Initializes a given instance of a class containing the energetic outputs of a turbine stage.
        
        **parameters**:
        :param N: number of blade elements (discretization). Default is 50.
        
        **return**:
        :return: instance of 'eout' class.
        
        **rtype**:
        :rtype: type(eout), obj.        
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------DECLARING ATTRIBUTES-----------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        # Radially-varying differential torque.
        self._difftau = np.zeros(N + 1)
        # Radially-varying differential power.
        self._diffpower = np.zeros(N + 1)
        # Radially-varying static-to-static pressure drop.
        self._dp_st_to_st = np.zeros(N + 1)
        # Radially-varying total-to-static pressure drop.
        self._dp_tot_to_st = np.zeros(N + 1)
        # Radially-varying total-to-total pressure drop.
        self._dp_tot_to_tot = np.zeros(N + 1)
        # Radially-varying viscous losses.
        self._dpvisc = np.zeros(N + 1)
        # Radially-varying kinetic losses.
        self._dpk = np.zeros(N + 1)
        # Radially-varying efficiency.
        self._deff = np.zeros(N + 1)
        # Integrated torque value.
        self._tau = 0
        # Integrated power.
        self._power = 0
        # Non-dimensional power.
        self._Pi = 0
        # Integrated static-to-static pressure drop.
        self._p_st_to_st = 0
        # Integrated total-to-static pressure drop.
        self._p_tot_to_st = 0
        # Non-dimensional integrated total-to-static pressure drop.
        self._Psi = 0
        # Integrated total-to-total pressure drop.
        self._p_tot_to_tot = 0
        # Integrated viscous losses.
        self._pvisc = 0
        # Integrated kinetic losses.
        self._pk = 0
        # Integrated efficiency.
        self._eff = 0
        # Efficiency coming from non-dimensional quantities.
        self._Eta = 0

    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------DECLARING PROPERTIES-----------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#                
        
    # Declaration of get property difftau for differential torque.
    @property
    def difftau(self) -> np.ndarray:
        return self._difftau
    # Declaration of get property diffpower for differential power.
    @property
    def diffpower(self) -> np.ndarray:
        return self._diffpower
    # Declaration of get property dp_st_to_st for differential static-to-static pressure-drop.
    @property
    def dp_st_to_st(self) -> np.ndarray:
        return self._dp_st_to_st
    # Declaration of get property dp_tot_to_st for differential total-to-static pressure-drop.
    @property
    def dp_tot_to_st(self) -> np.ndarray:
        return self._dp_tot_to_st
    # Declaration of get property dp_tot_to_tot for differential total-to-total pressure-drop.
    @property
    def dp_tot_to_tot(self) -> np.ndarray:
        return self._dp_tot_to_tot
    # Declaration of get property dpvisc for differential pressure-drop due to viscous losses.
    @property
    def dpvisc(self) -> np.ndarray:
        return self._dpvisc
    # Declaration of get property dpk for differential pressure-drop due to kinetic losses.
    @property
    def dpk(self) -> np.ndarray:
        return self._dpk
    # Declaration of get property deff for differential efficiency.
    @property
    def deff(self) -> np.ndarray:
        return self._deff
    # Declaration of get property tau for torque.
    @property
    def tau(self) -> float:
        return self._tau
    # Declaration of get property power for power.
    @property
    def power(self) -> float:
        return self._power
    # Declaration of get property Pi for dimensionless power.
    @property
    def Pi(self) -> float:
        return self._Pi
    # Declaration of get property p_st_to_st for static-to-static pressure-drop.
    @property
    def p_st_to_st(self) -> float:
        return self._p_st_to_st
    # Declaration of get property p_tot_to_st for total-to-static pressure-drop.
    @property
    def p_tot_to_st(self) -> float:
        return self._p_tot_to_st
    # Declaration of get property p_tot_to_tot for total-to-total pressure-drop.
    @property
    def p_tot_to_tot(self) -> float:
        return self._p_tot_to_tot
    # Declaration of get property Psi for dimensionless total-to-total pressure-drop.
    @property
    def Psi(self) -> float:
        return self._Psi
    # Declaration of get property pvisc for pressure-drop due to viscous losses.
    @property
    def pvisc(self) -> float:
        return self._pvisc
    # Declaration of get property pk for pressure-drop due to kinetic losses.
    @property
    def pk(self) -> float:
        return self._pk
    # Declaration of get property eff for efficiency.
    @property
    def eff(self) -> float:
        return self._eff
    # Declaration of get property Eta for efficiency coming from dimensionless variables.
    @property
    def Eta(self) -> float:
        return self._Eta
    
    # Declaration of set property difftau for differential torque.    
    @difftau.setter 
    def difftau(self, value: np.ndarray) -> None:
        assert len(value) == len(self._difftau), 'Provide an array of size ' + len(self._difftau) + '; current array has size ' + str(len(value))
        self._difftau = value
        return
    # Declaration of set property diffpower for differential power.
    @diffpower.setter
    def diffpower(self, value: np.ndarray) -> None:
        assert len(value) == len(self._diffpower), 'Provide an array of size ' + len(self._diffpower) + '; current array has size ' + str(len(value))
        self._diffpower = value
        return
    # Declaration of set property dp_st_to_st for differential static-to-static pressure-drop.
    @dp_st_to_st.setter
    def dp_st_to_st(self, value: np.ndarray) -> None:
        assert len(value) == len(self._dp_st_to_st), 'Provide an array of size ' + len(self._dp_st_to_st) + '; current array has size ' + str(len(value))
        self._dp_st_to_st = value
        return
    # Declaration of set property dp_tot_to_st for differential total-to-static pressure-drop.
    @dp_tot_to_st.setter
    def dp_tot_to_st(self, value: np.ndarray) -> None:
        assert len(value) ==len(self._dp_tot_to_st), 'Provide an array of size ' + len(self._dp_tot_to_st) + '; current array has size ' + str(len(value))
        self._dp_tot_to_st = value
        return
    # Declaration of set property dp_tot_to_tot for differential total-to-total pressure-drop.
    @dp_tot_to_tot.setter 
    def dp_tot_to_tot(self, value: np.ndarray) -> None:
        assert len(value) == len(self._dp_tot_to_tot), 'Provide an array of size ' + len(self._dp_tot_to_tot) + '; current array has size ' + str(len(value))
        self._dp_tot_to_tot = value
        return
    # Declaration of set property dpvisc for differential pressure-drop due to viscous losses.
    @dpvisc.setter
    def dpvisc(self, value: np.ndarray) -> None:
        assert len(value) == len(self._dpvisc), 'Provide an array of size ' + len(self._dpvisc) + '; current array has size ' + str(len(value))
        self._dpvisc = value
        return
    # Declaration of set property dpk for differential pressure-drop due to kinetic losses.    
    @dpk.setter
    def dpk(self, value: np.ndarray) -> None:
        assert len(value) == len(self._dpk), 'Provide an array of size ' + len(self._dpk) + '; current array has size ' + str(len(value))
        self._dpk = value
        return   
    # Declaration of set property deff for differential efficiency.    
    @deff.setter
    def deff(self, value: np.ndarray) -> None:
        assert len(value) == len(self._deff), 'Provide an array of size ' + len(self._deff) + '; current array has size ' + str(len(value))
        self._deff = value
        return
    # Declaration of set property tau for torque.
    @tau.setter 
    def tau(self, value: float) -> None:        
        self._tau = value
        return
    # Declaration of set property power for power.    
    @power.setter
    def power(self, value: float) -> None:        
        self._power = value
        return
    # Declaration of set property Pi for dimensionless power.
    @Pi.setter
    def Pi(self, value: float) -> None:
        self._Pi = value
        return
    # Declaration of set property p_st_to_st for static-to-static pressure-drop.
    @p_st_to_st.setter
    def p_st_to_st(self, value: float) -> None:
        self._p_st_to_st = value
        return
    # Declaration of set property p_tot_to_st for total-to-static pressure-drop.
    @p_tot_to_st.setter
    def p_tot_to_st(self, value: float) -> None:
        self._p_tot_to_st = value
        return
    # Declaration of set property p_tot_to_tot for total-to-total pressure-drop.
    @p_tot_to_tot.setter 
    def p_tot_to_tot(self, value: float) -> None:
        self._p_tot_to_tot = value
        return
    # Declaration of set property Psi for dimensionless total-to-total pressure-drop.    
    @Psi.setter
    def Psi(self, value: float) -> None:
        self._Psi = value
        return
    # Declaration of set property pvisc for pressure-drop due to viscous losses.    
    @pvisc.setter
    def pvisc(self, value: float) -> None:
        self._pvisc = value
        return
    # Declaration of set property pk for pressure-drop due to kinetic losses.    
    @pk.setter
    def pk(self, value: float) -> None:
        self._pk = value
        return       
    # Declaration of set property eff for efficiency.        
    @eff.setter
    def eff(self, value: float) -> None:
        self._eff = value
        return
    # Declaration of set property Eta for efficiency coming from dimensionless variables.
    @Eta.setter
    def Eta(self, value: float) -> None:
        self._Eta = value
        return

#--------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------CLASS turbstage-----------------------------------------------#             
#--------------------------------------------------------------------------------------------------------------------#
class turbstage:
    '''Sets a turbine stage.'''
    
    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------DEFINING METHODS-------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    
    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------__init__() METHOD------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    def __init__(self,
                 ttype: turbCnt.TYPES=turbCnt.TYPES.Wells,
                 omega: float=3600,                 
                 N: int=50,
                 rcas: float=0.25,
                 hub_to_tip_ratio: float=0.75,
                 chord: Union[list, np.ndarray, float]=0.117,
                 angpitch: Union[list, np.ndarray, float]=0,
                 airfoil: Union[geCnt.PROFILES, np.ndarray]=geCnt.PROFILES.NACA0015,
                 tip_percent: float=0.5,
                 Z: int=7,
                 p: float=101325,
                 T: float=288.15,
                 R: float=287.058,
                 nu: float=1.81e-5) -> None:
        '''Initializes a given instance of a class containing the a turbine stage.
        
        **parameters**:
        :param ttype: instance of the turbCnt.TYPES enum specifying turbine type; either 'Wells' or 'Impulse'. Default is 'Wells'.
        :param omega: rotational speed of turbine stage, in [RPM]. Default is 3600 [RPM].
        :param N: number of blade elements (discretization). Default is 50.
        :param rcas: casing radius. Default is 0.25.
        :param hub_to_tip_ratio: hub to tip ratio value. Default is 0.75.
        :param chord: chord value. Default is 0.117. It can be an element-wise array.
        :param angpitch: angular pitch value. Default is 0. It can be an element-wise array.
        :param airfoil: geometry of the airfoils of the blade. Default is NACA0015, meaning that the airfoil geometry is constant throughout the blade. It can be an element-wise array.
        :param tip_percent: tip gap value, in terms of chord percentage. Default is 0.5%.
        :param Z: number of blades. Default is 7.
        :param p: atmospheric pressure in [Pa]. Default value is 101325 [Pa].
        :param T: ambient temperature in [K]. Default value is 288.15 [K].
        :param R: gas constant in [J/(kg·K)]. Default value is 287.058 [J/(kg·K)]
        :param nu: gas viscosity in [kg/(m·s)]. Default value is 1.81e-5 [kg/(m·s)].
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------DECLARING ATTRIBUTES-----------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        # Type of turbine.
        self._ttype = ttype
        # Rotational speed.
        self._omega = omega*2*np.pi/60
        # Geometrical parameters.
        self._geom = geom(ttype=ttype,
                               N=N,
                               rcas=rcas,
                               hub_to_tip_ratio=hub_to_tip_ratio,
                               chord=chord,
                               angpitch=angpitch,
                               airfoil=airfoil,
                               tip_percent=tip_percent,
                               Z=Z
                              )
        # Flow parameters.
        self._flow = flow(p=p,
                               T=T,
                               R=R,
                               nu=nu
                              )
        # Inlet velocity triangle.
        self._it = velocity_triangle(N=N)
        # Outlet velocity triangle.
        self._ot = velocity_triangle(N=N)
        # Coefficients.
        self._lcoefs = loads(N=N)
        # Energetic output.
        self._eout = eout(N=N)

    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------DECLARING PROPERTIES-----------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#        
        
    # Declaration of get property ttype for turbine stage type.
    @property
    def ttype(self) -> str:
        return self._ttype
    # Declaration of get property omega for rotational speed.
    @property
    def omega(self) -> float:
        return self._omega
    # Declaration of get property gp for turbine stage geometrical parameters.
    @property
    def geom(self) -> Union[float, np.ndarray]:
        return self._geom
    # Declaration of get property fp for turbine stage flow parameters.
    @property
    def flow(self) -> float:
        return self._flow
    # Declaration of get property it for turbine stage inlet velocity triangle.
    @property
    def it(self) -> velocity_triangle:
        return self._it
    # Declaration of get property ot for turbine stage outlet velocity triangle.
    @property
    def ot(self) -> velocity_triangle:
        return self._ot
    # Declaration of get property coefs for turbine stage load coefficients.
    @property
    def lcoefs(self) -> loads:
        return self._lcoefs
    # Declaration of get property eout for turbine energetic output.
    @property
    def eout(self) -> eout:
        return self._eout
    
    # Declaration of set property omega for rotational speed.
    @omega.setter
    def omega(self, value) -> None:
        self._omega = value
        return
    # Declaration of set property it for input velocity triangle.
    @it.setter
    def it(self, value) -> None:
        self._it = value
        return
    
    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------DEFINING METHODS-------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#

    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------it_triangle_solve() METHOD-------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#    
    def it_triangle_solve(self,
                          phi_q_v: Union[list, np.ndarray, float],
                          omega: float,
                          inputparam: str="vx",
                          refvel: str='it',
                          ret=True) -> Union[tuple, None]:        
        '''Function for subsidiarizing the calculation of the inlet velocity triangle.

        **parameters**:
        :param phi_q_v: input parameter value for which the BEM method is applied. It can be either a flow coefficient value (phi), a flow-rate value (q) or an axial velocity value (vx).
        :param omega: rotational speed of the turbine [rad/s].
        :param inputparam: string determining the type of input parameter. It can be either 'phi' (stipulating flow coefficient), 'flowrate' (flow-rate) or 'vx' (axial velocity). Default is 'vx'.
        :param refvel: string representing the referential velocity employed for calculating loads and velocity-dependent outputs; it is "it" for setting the referential velocity equal to the relative velocity magnitude of the inlet velocity triangle, or 'ave' for setting it equal to the relative velocity magnitude based on the actuator disk value (the axial component being the average of downstream and upstream velocities).
        :param ret: boolean flag determining whether a tuple is to be returned, or None. Default is True, meaning that a tuple is returned.
        
        **return**:
        :return: a tuple containing; (tangential velocity at midspan, flow coefficient at midspan, auxiliary differential flow-rates per blade element, referential axial velocities, referential total velocities, referential relative velocities) (ret=True); otherwise None (ret=False).
        
        **rtype**:
        :rtype: tuple, None.
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------------------BODY-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------# 
                              
        # Conditional for checking whether "inputparam" is "it"; in such a case, a pre-established velocity triangle is provided as an input, and it is not necessary to begin the calculations from determining the inlet velocity triangle.
        if inputparam == "it":
            if np.sign(self.it.U[0])!=-np.sign(omega):
                self.it.U *= -1
                self.it.phi *= -1
                self.it.alpha = np.arctan(self.it.vx/self.it.vtheta)
                self.it.wtheta = self.it.vtheta - self.it.U
                self.it.w = np.sqrt(self.it.wx**2 + self.it.wtheta**2)
                self.it.beta = np.abs(np.arctan(self.it.wx/self.it.wtheta)) + self.geom.angpitch*np.pi/180
            Um = self.it.U[len(self.it.U)//2]
            phi = self.it.phi[len(self.it.phi)//2]
            dq_aux = np.abs(self.it.dq)
        else:
            # Tangential velocity calculation.
            self.it.U = -np.array([omega*_ for _ in self.geom.r])
            self.ot.U = copy.deepcopy(self.it.U)
            # Midspan tangential velocity instantiation.
            Um = self.it.U[len(self.it.U)//2]
            # Axial velocity, midspan flow parameter and flow coefficient calculations depending on the provided input parameter.
            if inputparam=="phi":
                # If the input parameter is a flow coefficient, then:
                # check, first, if the input variable is an array or not.        
                if isinstance(phi_q_v, np.ndarray):
                    # If it is, then set the axial velocity to the product of the input parameter and the local (element-wise) tangential velocity.
                    self.it.vx = np.array([_[0]*_[1] for _ in zip(phi_q_v, self.it.U)])
                else:
                    # If it isn't, then the axial velocity of the inlet velocity triangle is obtained by multiplying phi and Um, assuming a constant axial velocity radially.
                    self.it.vx = np.array([phi_q_v*Um for _ in self.geom.r])                    
                # the midspan flow parameter coincides with the input value.
                phi = phi_q_v
                # The differential flow-rate is calculated from its definition.
                self.it.dq = 2*np.pi*self.geom.r*self.it.vx
                # The flow-rate is calculated by integration.
                self.it.q = simpson(self.it.dq, dx=self.geom.dr)
            elif inputparam=="q":
                # If the input parameter is a flow rate value, then:
                # check, first, if the input variable is an array or not.
                if isinstance(phi_q_v, np.ndarray):
                    # If it is, then set the axial velocity from its definition of differential flow-rate.
                    self.it.vx = phi_q_v/(2*np.pi*self.geom.r)
                    # Set the differential flow-rate to the input variable.
                    self.it.dq = phi_q_v
                    # Set the flow-rate to the integral value of the differential flow-rate.
                    self.it.q = simpson(self.it.dq, dx=self.geom.dr)
                else:
                    # If it isn't, then set the axial velocity from its definition of integral flow-rate, assuming constant axial velocity of the inlet velocity triangle.
                    self.it.vx = np.array([phi_q_v/(np.pi*(self.geom.rtip**2 - self.geom.rhub**2)) for _ in self.geom.r])
                    # Set the integral flow-rate to the input variable.
                    self.it.q = phi_q_v
                    # Set the differential flow-rate from its definition.
                    self.it.dq = 2*np.pi*self.geom.r*self.it.vx
                # The midspan flow parameter is obtained by dividing the axial velocity with the midspan tangential velocity.
                phi = self.it.vx[len(self.it.vx)//2]/Um
            else:
                # Check, first, if the input variable is an array or not.
                if isinstance(phi_q_v, np.ndarray):
                    # If it is, then set the axial velocity component of the inlet velocity triangle to the input variable.
                    self.it.vx = phi_q_v
                else:
                    # If it isn't, then set the axial velocity component of the inlet velocity triangle to a constant value equal to the input parameter.
                    self.it.vx = np.array([phi_q_v for _ in self.geom.r])
                # The midspan flow paramter is obtained by dividing the axial velocity with the midspan tangential velocity.
                phi = self.it.vx[len(self.it.vx)//2]/Um    
                # The differential flow-rate is calculated from its definition.
                self.it.dq = 2*np.pi*self.geom.r*self.it.vx
                # The flow-rate is calculated by integration.
                self.it.q = simpson(self.it.dq, dx=self.geom.dr)
            # Setting the non-dimensional flow-coefficient at the input triangle.
            self.it.Phi = self.it.q/(np.abs(omega)*(2*self.geom.rtip)**3)            
            # The alpha angle (between the absolute and tangential velocities) is set to pi/2 by assumption.
            self.it.alpha = np.array([np.pi/2 for i in range(len(self.it.alpha))])
            # The magnitude of the absolute velocity is obtained by pitagorean application.
            self.it.v = np.sqrt(self.it.vx**2 + self.it.vtheta**2)
            # The axial component of the relative velocity is set to the axial component of the absolute velocity.
            self.it.wx = np.array([_ for _ in self.it.vx])
            # The tangential component of the relative velocity is set to (-) the tangential velocity, from the vectorial identity.
            self.it.wtheta = np.array([-_ for _ in self.it.U])
            # The magnitude of the relative velocity is obtained by pitagorean application.
            self.it.w = np.sqrt(self.it.wx**2 + self.it.wtheta**2)
            # The beta angle (between the relative and tangential velocities) is obtained from trigonometric relations.            
            self.it.beta = np.arctan(self.it.wx/self.it.wtheta)
            # The gamma angle is the sum of the beta and the pitch angle.
            self.it.gamma = np.arctan(self.it.wx/self.it.wtheta) + self.geom.angpitch*np.pi/180
            # The radially-varying flow coefficient is obtained from its definition.
            self.it.phi = self.it.vx/self.it.U
            # Expressing an auxiliar differential flow-rate as the absolute value of the flow-rate (necessary for loss calculations).
            dq_aux = np.abs(self.it.dq)

        # Declaring 'vx', 'v' and w' as referential velocity magnitudes at the disc and setting them to 0.
        vx = 0
        v = 0
        w = 0
        # Conditional for setting the referential velocity magnitudes according to the value of 'refvel'.
        if refvel == "it":
            # If 'refvel' is 'it', then the referential magnitude is set equal to the inlet velocity triangle's relative magnitude.
            vx = self.it.vx
            v = self.it.v
            w = self.it.w
        else:
            # If 'refvel' is 'ave', then the referential velocity magnitudes are set equal to the velocity magnitudes at the disk.
            vx = 0.5*(self.ot.vx + self.it.vx)
            v = np.sqrt(vx**2 + self.it.vtheta**2)
            w = np.sqrt(vx**2 + self.it.wtheta**2)        
        
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------RETURN------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#

        # Conditional tree for determining the return statement.
        if ret:
            # Return statement providing it-related variables.
            return Um, phi, dq_aux, vx, v, w
        else:
            # Return statement.
            return

    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------reynolds_calculation() METHOD------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    def reynolds_calculation(self, vel: np.ndarray) -> None:
        '''Function for subsidiarizing the calculation of the blade-element-wise Reynolds numbers.

        **parameters**:
        :param vel: referential velocity for calculating the Reynolds number.
        '''

        #--------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------------------BODY-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#              
        
        # The Reynolds number is obtained from its definition, Re=p·W·c/u, where p is the density and nu the viscosity.        
        self.lcoefs.Re = self.flow.rho*self.geom.chord*vel/self.flow.nu

        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------RETURN------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        # Return statement.
        return

    #--------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------interp_data_calculation() METHOD----------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#    
    def interp_data_calculation(self, gamma: float, blade_index: int, mode: str='whole') -> tuple:
        '''Function for subsidiarizing the calculation of the data to be used in the bilinear interpolation.

        **parameters**:
        :param gamma: input relative angle-of-attack.
        :param blade_index: index determining the blade element for which the interpolation data is calculated.
        :param mode: string determining the mode of calculation and return type.
        
        **return**:
        :return: tuple containing; (list of angles-of-attack from polars, index of 0 angle-of-attack, list of differences between absolute and relative angles-of-attack, flooring index, ceiling index, list of lifts from polars, list of drags from polars, lower bound of Reynolds, upper bound of Reynolds) (mode=whole). Otherwise; (list of angles-of-atatck from polars, list of lifts from polars, lower bound of Reynolds, upper bound of Reynolds) (mode=min).
        
        **rtype**:
        :rtype: tuple.
        '''

        #--------------------------------------------------------------------------------------------------------------------#
        #------------------------------------------------------ASSERTIONS----------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#        
        
        # Asserts whether the 'mode' input variable is valid. If not, it raises an error message.
        assert mode in ["whole", "min"], "Provide a valid mode for the calculations, either 'whole' or 'min''; current value is " + str(mode)        

        #--------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------------------BODY-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#   
        
        # Getting the lift and drag coefficient polars corresponding to current blade element's airfoil geometry.
        polar_cl = polars[self.geom.airfoil[blade_index].name + "CL"]
        # If mode=='whole', then retrieve the cd data from the corresponding polar.
        if mode == 'whole':
            polar_cd = polars[self.geom.airfoil[blade_index].name + "CD"]
        # Getting the Reynolds values for which the lift and drag coefficients have been obtained.
        Re_values = list(polar_cl.keys())[1:]
        # Getting the index that determines between which tested Reynolds values falls the current Reynolds, and retrieving those extreme Reynolds values for interpolation.
        if all([self.lcoefs.Re[blade_index] - _ > 0 for _ in Re_values]):
            minindex = len(Re_values) - 2
        else:
            minindex = [self.lcoefs.Re[blade_index] - _ < 0 for _ in Re_values].index(True) - 1
        Remin = Re_values[minindex]
        Remax = Re_values[minindex + 1]
        # Getting the lift and drag datasets that correspond to the minimum and maximum Reynolds values.
        lifts = [polar_cl[Remin], polar_cl[Remax]]
        # If mode=='whole', then set the maximum and minimum data for the 'drags' variable.
        if mode == 'whole':
            drags = [polar_cd[Remin], polar_cd[Remax]]

        # Retrieving the angles-of-attack for which the coefficients have been obtained.
        alpha = polar_cl['angle'] 
        # Getting the index for which the angle-of-attack is 0.
        alpha_0ind = list(alpha).index(0)
        # Calculating the effective angle observed by the blade element, difference between the absolute and relative angles-of-attack.
        diffalpha = list(alpha - gamma)
        # Calculating the floor and ceiling indexes of the effective angle for computing the bilinear interpolation, in case it is required.
        floorind = [_ > 0 for _ in diffalpha].index(True) - 1        
        ceilind = [_ > 0 for _ in diffalpha].index(True)             

        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------RETURN------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#

        # Conditional tree for determining the 'return' data structure.
        if mode == 'whole':
            # If mode=='whole', then return the angles, lifts, drags and maximum and minimum Reynolds numbers.
            return alpha, alpha_0ind, diffalpha, floorind, ceilind, lifts, drags, Remin, Remax
        else:
            # If mode=='min' (minimal), then return the lifts and maximum and minimum Reynolds numbers.
            return alpha, lifts, Remin, Remax

    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------differential_load_calculation() METHOD-------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    def differential_load_calculation(self, vel: float) -> None:
        '''Function for subsidiarizing the calculation of differential loads.

        **parameters**:
        :param vel: velocity reference.
        '''

        #--------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------------------BODY-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------# 
                                          
        # Differential lift computation.
        self.lcoefs.dl = self.lcoefs.cl*0.5*self.flow.rho*(vel**2)*self.geom.dA
        # Differential drag computation.
        self.lcoefs.dd = self.lcoefs.cd*0.5*self.flow.rho*(vel**2)*self.geom.dA
        # Induced drag coefficients from its formulation.
        self.lcoefs.cdtc = self.lcoefs.cd + 0.7*np.abs(self.lcoefs.cl)*self.geom.tip_c/(self.geom.AR*self.geom.pitch)
        # Differential induced drag load computation.
        self.lcoefs.ddtc = self.lcoefs.cdtc*0.5*self.flow.rho*(vel**2)*self.geom.dA
        # Projection for obtaining axial coefficient.
        self.lcoefs.cx = self.lcoefs.cl*np.cos(self.it.beta) + self.lcoefs.cdtc*np.sin(self.it.beta)
        # Differential axial load computation.
        self.lcoefs.dx = self.lcoefs.cx*0.5*self.flow.rho*(vel**2)*self.geom.dA
        # Projection for obtaining tangential coefficient.
        self.lcoefs.ctheta = self.lcoefs.cl*np.sin(self.it.beta) - self.lcoefs.cdtc*np.cos(self.it.beta)
        # Differential tangential load computation.
        self.lcoefs.dtheta = self.lcoefs.ctheta*0.5*self.flow.rho*(vel**2)*self.geom.dA

        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------RETURN------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#

        # return statement
        return

    #--------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------cif_correction() METHOD---------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#    
    def cif_correction(self) -> None:
        '''Function for subsidiarizing the calculation of corrected load coefficients.'''

        #--------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------------------BODY-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------# 
                                          
        # Correction of axial coefficient.
        self.lcoefs.cxCIF = self.lcoefs.cx/(1 - self.geom.sigma**2)
        # Correction of axial load.
        self.lcoefs.dxCIF = self.lcoefs.dx/(1 - self.geom.sigma**2)
        # Correction of tangential coefficient.
        self.lcoefs.cthetaCIF = self.lcoefs.ctheta/(1 - self.geom.sigma**2)       
        # Correction of tangential load.
        self.lcoefs.dthetaCIF = self.lcoefs.dtheta/(1 - self.geom.sigma**2)

        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------RETURN------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#

        # return statement
        return

    #--------------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------bessel_imposition() METHOD--------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#    
    def bessel_imposition(self, x_idx: int=0) -> None:
        '''Function for subsidiarizing the imposition of equilibrium via the Bessel formulation.'''

        #--------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------------------BODY-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#          

        # Parameters
        R = 1  # Example boundary radius (adjust as needed)
        U = 1  # Axial velocity (adjust as needed)
        V = 1  # Radial velocity (adjust as needed)
        rho = 1  # Density (adjust as needed)
        p0 = 1  # Pressure reference (adjust as needed)

        # Define Bessel functions of the first kind (J_0) and second kind (Y_0)
        def bessel_functions(n, r):
            return scsp.jn(n, r), scsp.yn(n, r)

        # Calculate eigenvalues (lambda_n) from boundary conditions (example, use actual method)
        def get_eigenvalues(n_max, r):
            # Here we use a simple approximation to get eigenvalues
            eigenvalues = np.linspace(1, n_max, n_max)  # Placeholder, adjust based on problem specifics
            return eigenvalues

        # Axial velocity function using boundary conditions
        def axial_velocity(r, x, n_max, i):
            A_n = np.ones(n_max)  # Placeholder for coefficients
            eigenvalues = get_eigenvalues(n_max, r)
            v_x = self.it.vx[i]                
            for n in range(n_max):
                Jn, Yn = bessel_functions(n, r)
                v_x += A_n[n] * np.exp(-eigenvalues[n] * x) * Jn  # Simplified for this example
            return v_x

        # Implement the iterative solution (simplified)
        def solve_velocity_field(r_values, x_values, n_max):
            velocity_field = np.zeros((len(r_values), len(x_values)))
            for i, r in enumerate(r_values):
                for j, x in enumerate(x_values):
                    velocity_field[i, j] = axial_velocity(r, x, n_max, i)
            return velocity_field

        # Plot the results
        r_values = self.geom.r
        x_values = np.linspace(0, self.geom.r[-1]*10, 100)   # Axial positions
        n_max = 10  # Maximum number of Bessel functions to include

        velocity_field = solve_velocity_field(r_values, x_values, n_max)
        
        self.ot.vx = np.array(velocity_field[:,x_idx]*np.sign(self.it.vx[0]))
        self.ot.alpha = np.arctan(self.ot.vx/self.ot.vtheta)

        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------RETURN------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#

        # return statement
        return

    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------radEq_imposition() METHOD--------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#    
    def radEq_imposition(self) -> None:
        '''Function for subsidiarizing the calculation of equilibrium via radial-equilibrium imposition.'''

        #--------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------------------BODY-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#        
        # Instantiating the 'vx2' variable as the square of the axial component of the outlet velocity triangle.
        vx2 = np.array([float(_) for _ in self.it.vx])**2
        # Instantiating the 'gamma' variable, which represents the element-wise circulation along the blade.
        gamma = 2*np.pi*self.geom.r*self.ot.vtheta       
        # Instantiating the 'dvx2_dr' variable, which represents the derivative of the square of the axial velocity component with respect to the radial parameter.
        dvx2_dr = (-1/(2*np.pi*self.geom.r)**2)*(gamma**2)        
        # Computing the radial values of the square of the axial velocity component of the outlet velocity triangle from the obtained differences. The loop starts from the second blade element, meaning that the velocity value of the first element is kept unchanged on a first guess/approximation.
        for e, _ in enumerate(self.geom.r[1:]):
            e += 1
            vx2[e] = vx2[e-1] + dvx2_dr[e]*self.geom.dr                   
        # Instantiating the 'diff_q' variable and setting its value to 1, representing the threshold for fulfilling the continuity condition.
        diff_q = 1
        # Instantiating the 'diff_prev' variable and setting its value to 2 (so that, in the first step, 'diff_q_prev > diff_q'). 
        diff_q_prev = 2
        # Instantiating the 'sign' variable and setting its value to 1.
        sign = 1
        # Instantiating the 'reverse' variable and setting its value to 'False'.
        reverse = False
        # Continuity-equation enforcement.
        while diff_q > self.it.q*0.1:
            # If there are no negative square elements, then the axial velocity component of the outlet velocity triangle is set equal to the square-root of the obtained distribution of the square velocity.
            if min(vx2) >= 0:
                self.ot.vx = np.sqrt(vx2)
            # If there are negative square elements, those are computed by making the negative value absolute and making the overall velocity component change its sign.
            else:
                break
            # The 'diff' and 'diffaux' variables compute the difference between a current-step flow-rate and the flow-rate provided by the inlet velocity triangle, and the previous-step flow-rate and idem, respecively (check the while loop below).                
            diff_q = np.abs(np.abs(self.it.q) - np.abs(simpson(2*np.pi*self.geom.r*self.ot.vx, dx=self.geom.dr)))
            # Condition for checking whether the current step's continuity condition fulfillment is further from the previous step's one; if it is, then continuity divergence is taking place. The sign is reversed, and the 'reverse' variable set to 'True'.
            if diff_q_prev - diff_q < 0 and not reverse:
                sign *= -1
                reverse = True
            # Setting 'diff_q_prev' to 'diff_q' for the next step's continuity condition checking.
            diff_q_prev = diff_q
            # Varying the square of the axial velocity component differentially.            
            vx2 += sign*1e-2
        # Casting the outlet velocity triangle's axial velocity components to match their original signs.
        self.ot.vx *= np.sign(self.it.vx[0])
        # Recomputing the outlet velocity triangle's absolute angle.
        self.ot.alpha = np.arctan(self.ot.vx/self.ot.vtheta)

        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------RETURN------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#

        # return statement
        return
    
    #--------------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------radEq_imposition_2() METHOD-------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#    
    def radEq_imposition_2(self) -> None:
        '''Function for calculating equilibrium via radial-equilibrium imposition using numerical integration.'''

        # Square of the axial component of the outlet velocity triangle
        vx2_0 = self.it.vx[0]**2  # Initial condition for vx2

        # Element-wise circulation along the blade
        gamma = 2 * np.pi * self.geom.r * self.ot.vtheta

        # Function representing the differential equation
        def dvx2_dr(r, vx2):
            idx = np.searchsorted(self.geom.r, r)  # Find the nearest index for r
            if idx >= len(self.geom.r):
                idx = len(self.geom.r) - 1
            return - (gamma[idx] / (2 * np.pi * r))**2

        # Radial range for integration
        r_span = (self.geom.r[0], self.geom.r[-1])

        # Solve the differential equation using solve_ivp
        solution = solve_ivp(dvx2_dr, r_span, [vx2_0], t_eval=self.geom.r, method='RK45')

        # Check if the integration was successful
        if solution.success:
            vx2 = solution.y[0]
        else:
            raise RuntimeError("Integration failed!")

        # Ensure no negative values in vx2
        vx2 = np.clip(vx2, a_min=0, a_max=None)

        # Set the axial velocity component of the outlet velocity triangle
        self.ot.vx = np.sqrt(vx2)

        # Enforce continuity by adjusting the flow rate if necessary
        diff_q = np.abs(np.abs(self.it.q) - np.abs(simpson(2 * np.pi * self.geom.r * self.ot.vx, dx=self.geom.dr)))

        # if diff_q > self.it.q * 0.1:
        #     raise RuntimeError("Continuity condition not satisfied within tolerance!")

        # Recompute the outlet velocity triangle's absolute angle
        self.ot.alpha = np.arctan(self.ot.vx / self.ot.vtheta)

        return    
        
    #--------------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------ot_triangle_solve() METHOD--------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#    
    def ot_triangle_solve(self, omega: float, radFor: str='radEq', x_idx: int=0) -> None:
        '''Function for subsidiarizing the calculation of the outlet velocity triangle.

        **parameters**:
        :param omega: rotational speed of the turbine [rad/s].
        :param radFor: string representing the radial-equilibrium formulation. Either 'radEq' for forcing radial-equilibrium-equation condition, or 'Bessel' for forcing Bessel-based formulation.
        '''

        #--------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------------------BODY-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#         
        
        # Calculation of the absolute angle of the outlet velocity triangle.
        self.ot.alpha = np.pi/2 - np.arctan((1/np.tan(self.it.alpha))-(self.geom.Z*self.geom.chord*self.lcoefs.cthetaCIF)/(4*np.pi*self.geom.r*np.sin(np.abs(self.it.beta))**2))                           
        # The downstream absolute axial velocity component is set equal to the upstream one, by modelization assumption.
        self.ot.vx = copy.deepcopy(self.it.vx)
        # The tangential component is obtained via trigonometrical relation.        
        self.ot.vtheta = np.sign(omega)*np.abs(self.ot.vx)/np.tan(self.ot.alpha)            
        
        # Conditional for determining which scheme is to be applied for the radial formulation.
        if radFor == "Bessel":
            # Imposting Bessel-based formulation.
            self.bessel_imposition(x_idx=x_idx)
        elif radFor == "radEq":
            # Imposing radial-equilibrium-based formulation.
            self.radEq_imposition()
        elif radFor == "radEq2":
            # Imposing improved radial-equilibrium-based formulation.
            self.radEq_imposition_2()
        
        # Calculating the flow-rate at the outlet velocity triangle, which should coincide with the inlet flow-rate due to continuity enforcement.
        self.ot.q = simpson(2*np.pi*self.geom.r*self.ot.vx, dx=self.geom.dr)
        # Setting the non-dimensional flow-coefficient at the output triangle.
        self.ot.Phi = self.ot.q/(omega*(2*self.geom.rtip)**3)
        # The pitagorean theorem is used for calculating the absolute velocity magnitude of the outlet velocity triangle.
        self.ot.v = np.sqrt(self.ot.vx**2 + self.ot.vtheta**2)        
        # The axial velocity component of the relative velocity of the outlet velocity triangle is set equal to the absolute axial component, by trigonometric relations.
        self.ot.wx = self.ot.vx
        # The tangential component of the relative velocity of the outlet velocity triangle is obtained from vector identity.
        self.ot.wtheta = self.ot.vtheta - self.ot.U     
        # The pitagorean theorem is used for calculating the relative velocity magnitude of the outlet velocity triangle.
        self.ot.w = np.sqrt(self.ot.wx**2 + self.ot.wtheta**2)       
        # The relative angle of the outlet velocity triangle is obtained from trigonometric relations.
        self.ot.beta = np.arctan(self.ot.wx/self.ot.wtheta)

        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------RETURN------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#

        # return statement
        return

    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------load_calculation() METHOD--------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#    
    def load_calculation(self, omega: float, rel_vel: float, axial_vel_ref: np.ndarray, total_vel_ref: np.ndarray, dq_aux: float) -> None:
        '''Function for subsidiarizing the integral load calculations.

        **parameters**:
        :param omega: rotational speed of the turbine [rad/s].
        :param rel_vel: relative velocity reference.
        :param axial_vel_ref: axial velocity reference.
        :param total_vel_ref: total velocity reference.
        :param dq_aux: element-wise differential flow-rate reference.
        '''

        #--------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------------------BODY-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#         
        
        # Calculation of the differential torque. 
        self.eout.difftau = 0.5*self.flow.rho*self.geom.chord*self.geom.Z*self.lcoefs.cthetaCIF*(rel_vel**2)*self.geom.r
        # Calculation of integrated torque.
        self.eout.tau = simpson(self.eout.difftau, dx=self.geom.dr)    
        # Calculation of torque coefficient.
        if 0 not in axial_vel_ref and 0 not in total_vel_ref:
            self.lcoefs.ctau = self.eout.tau/(0.5*self.flow.rho*(total_vel_ref[len(total_vel_ref)//2]**2)*self.geom.chord*self.geom.rtip*(1 - self.geom.hub_to_tip_ratio)*self.geom.Z*axial_vel_ref[len(axial_vel_ref)//2])
        # Calculation of the differential power, multiplying the torque with the angular velocity.
        self.eout.diffpower = self.eout.difftau*np.abs(omega)
        # Calculation of integrated power.
        self.eout.power = simpson(self.eout.diffpower, dx=self.geom.dr)
        # Computing the non-dimensional output power coefficient.
        self.eout.Pi = self.eout.power/(self.flow.rho*(np.abs(omega)**3)*(2*self.geom.rtip)**5)
        # Conditional for setting the radially-varying powers, pressure drops and efficiencies to 0 in case beta1=0.
        if 0 in self.it.beta:
            pass
        else:
            # Calculation of differential static-to-static-pressure-drop.
            self.eout.dp_st_to_st = (self.flow.rho*self.geom.chord*self.geom.Z*self.lcoefs.cxCIF*(rel_vel**2))/(4*np.pi*self.geom.r)       
            # Calculation of static-to-static-pressure-drop.
            self.eout.p_st_to_st = simpson(2*np.pi*self.geom.r*self.eout.dp_st_to_st, dx=self.geom.dr)/simpson(2*np.pi*self.geom.r, dx=self.geom.dr)
            # Calculation of input coefficient.
            if 0 not in axial_vel_ref and 0 not in total_vel_ref:
                self.lcoefs.cinput = self.eout.p_st_to_st*self.it.q/(0.5*self.flow.rho*(total_vel_ref[len(total_vel_ref)//2]**2)*self.geom.chord*self.geom.rtip*(1 - self.geom.hub_to_tip_ratio)*self.geom.Z*axial_vel_ref[len(axial_vel_ref)//2])
            self.eout.dp_tot_to_st = -(np.abs(self.eout.dp_st_to_st) + 0.5*self.flow.rho*(axial_vel_ref**2))
            # Calculation of total-to-static pressure-drop.
            self.eout.p_tot_to_st = simpson(2*np.pi*self.geom.r*self.eout.dp_tot_to_st, dx=self.geom.dr)/simpson(2*np.pi*self.geom.r, dx=self.geom.dr)        
            # Calculation of differential total-to-total pressure-drop.
            self.eout.dp_tot_to_tot = -(np.abs(self.eout.dp_tot_to_st) - np.abs(0.5*self.flow.rho*self.ot.v**2))
            # Calculation of total-to-total pressure-drop.
            self.eout.p_tot_to_tot = simpson(2*np.pi*self.geom.r*self.eout.dp_tot_to_tot, dx=self.geom.dr)/simpson(2*np.pi*self.geom.r, dx=self.geom.dr)
            # Computing the non-dimensional power coefficient.
            self.eout.Psi = np.abs(self.eout.p_tot_to_st/(self.flow.rho*(np.abs(omega)**2)*(2*self.geom.rtip)**2))       
            # Calculation of differential loss power due to viscous effects.       
            self.eout.dpvisc = 0.5*self.flow.rho*self.geom.chord*self.geom.Z*self.lcoefs.cdtc*(rel_vel**3)    
            # Calculation of loss power due to viscous effects.
            self.eout.pvisc = simpson(self.eout.dpvisc, dx=self.geom.dr)
            # Calculation of differential loss power due to outlet kinetic velocity.
            self.eout.dpk = 0.5*self.flow.rho*(self.ot.v**2)*dq_aux
            # Calculation of loss power due to outlet kinetic energy.
            self.eout.pk = simpson(self.eout.dpk, dx=self.geom.dr)
            # Calculation of input power.
            dpin = self.eout.diffpower + self.eout.dpvisc + self.eout.dpk
            # Calculation of differential power.
            self.eout.deff = self.eout.diffpower/dpin
            # Calculation of efficiency.                
            self.eout.eff = simpson(self.eout.diffpower, dx=self.geom.dr)/simpson(dpin, dx=self.geom.dr)
            # Computing the efficiency derived from non-dimensional parameters.
            self.eout.Eta = np.abs(self.eout.Pi/(self.eout.Psi*self.ot.Phi))

        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------RETURN------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#

        # return statement
        return

    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------cl_cd_interp() METHOD----------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#                
    def cl_cd_interp(self, gamma: float, blade_index: int) -> tuple:
        '''Function for subsidiarizing the two-fold double-interpolation upon a blade element for computing its (local) lift and drag coefficients.

        **parameters**:
        :param gamma: relative angle-of-attack to which the blade element is subjected.
        :param blade_index: index of the blade element being considered.
        
        **return**:
        :return: 2-tuple containing the interpolated lift and drag coefficients.
        
        **rtype**:
        :rtype: tuple.
        '''

        #--------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------------------BODY-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#        
        
        # Computing the data for the bilinear interpolation.
        alpha, alpha_0ind, diffalpha, floorind, ceilind, lifts, drags, Remin, Remax = self.interp_data_calculation(gamma=gamma, blade_index=blade_index)
        # Maximum and minimum lift values are determined, given the current angle-of-attack, so that the double interpolation scheme is applied.        
        clminmin, clminmax = lifts[0][floorind], lifts[0][ceilind]
        clmaxmin, clmaxmax = lifts[1][floorind], lifts[1][ceilind]
        # Calling the double_interp() method with current angle-of-attack and Reynolds number, thus determining the lift value.)
        cl = mt.interpolation.doubleInterp(np.floor(gamma), Remin, np.ceil(gamma), Remax, gamma, self.lcoefs.Re[blade_index], clminmin, clmaxmin, clminmax, clmaxmax)
        # Maximum and minimum drag values are determined, given the current angle-of-attack, so that the double interpolation scheme is applied.
        cdminmin, cdminmax = drags[0][floorind], drags[0][ceilind]
        cdmaxmin, cdmaxmax = drags[1][floorind], drags[1][ceilind]
        # Calling the double_interp() method with current angle-of-attack and Reynolds number, thus determining the drag value.       
        cd = mt.interpolation.doubleInterp(np.floor(gamma), Remin, np.ceil(gamma), Remax, gamma, self.lcoefs.Re[blade_index], cdminmin, cdmaxmin, cdminmax, cdmaxmax)

        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------RETURN------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        # Return statement.        
        return cl, cd                                   
    
    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------BEM() METHOD---------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    def BEM(self, phi_q_v: Union[list, np.ndarray, float], inputparam: str='vx', mode: str='3DNOAD', radFor: str='radEq', dynamic_interp: bool=True, **kwargs) -> None:
        '''Applies the momentum element method to the current turbine stage.
        
        **parameters**:        
        :param phi_q_v: input parameter value for which the BEM method is applied. It can be either a flow coefficient value (phi), a flow-rate value (q) or an axial velocity value (vx).
        :param inputparam: string determining the type of input parameter. It can be either 'phi' (stipulating flow coefficient), 'flowrate' (flow-rate) or 'vx' (axial velocity). Default is 'vx'.
        :param mode: string determining the model of BEM to be applied. It can be either '3DNOAD' (semi-3D BEM model without actuator disk) or '3DAD' (semi-3D BEM model with actuator disk). Default is '3DNOAD'.
        :param radFor: string representing the radial-equilibrium formulation. Either 'radEq' for forcing radial-equilibrium-equation condition, or 'Bessel' for forcing Bessel-based formulation. Default is 'radEq'.
        :param dynamic_interp: boolean flag determining whether a dynamic interpolation is to be performed. Default is True.
        
        **kwargs**:
        :param x_idx: index of the axial stage at which to compute the outlet velocity field. Default is 0, which means that the outlet velocity is computed at the immediate downstream section of the turbine's rotor (infinitesimal displacement).        
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #------------------------------------------------------ASSERTIONS----------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#        
        
        # Asserts whether the 'mode' input variable is valid. If not, it raises an error message.
        assert mode in ["3DNOAD", "3DAD"], "Provide a valid mode for applying the BEM method, either '3DNOAD' or '3DAD''; current value is " + str(mode)
        
        #--------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------------------BODY-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#        
        
        # Conditional determining which of the BEM models is to be applied.
        if mode == '3DNOAD':            
            # Call to external function 'BEM_wells_noad'.
            self.BEM_wells_noad(phi_q_v, self.omega, inputparam, refvel='it', radFor=radFor, dynamic_interp=dynamic_interp, **kwargs)
        elif mode == '3DAD':
            # Call to external function 'BEM_wells_ads'.
            self.BEM_wells_ad(phi_q_v, self.omega, inputparam, refvel='it', radFor=radFor, dynamic_interp=dynamic_interp, **kwargs)
        
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------RETURN------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        # Return statement.
        return  
    
    #--------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------BEM_wells_noad() METHOD---------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    def BEM_wells_noad(self, phi_q_v: Union[list, np.ndarray, float], omega: float, inputparam: str="vx", refvel: str='it', radFor: str='radEq', dynamic_interp: bool=False, **kwargs) -> None:
        '''Applies the blade-element momentum theory (BEM) to a Wells-type turbine stage.

        The BEM theory is applied as formulated in the Ph.D. dissertation thesis of Ciappi (2021), available in the link https://flore.unifi.it/handle/2158/1245178#.YWXWWRpBzIV.

        **parameters**:
        :param ts: instance of the turbstage class to which the BEM is to be applied.
        :param phi_q_v: input parameter, either a flow coefficient (phi) a flow-rate value (q) or an axial velocity value (vx). It may be either an array-like structure, specifying phi, q or v for each blade element, or a single value, in which the provided value is set equal to all the blade elements.
        :param omega: rotational speed of the turbine [rad/s].
        :param inputparam: string representing the type of input provided; it is "phi" for indicating a flow coefficient, "q" for a flow-rate, "vx" for an axial velocity or "it" for a pre-established inlet velocity triangle. Default is "vx".
        :param refvel: string representing the referential velocity employed for calculating loads and velocity-dependent outputs; it is "it" for setting the referential velocity equal to the relative velocity magnitude of the inlet velocity triangle, or 'ave' for setting it equal to the relative velocity magnitude based on the actuator disk value (the axial component being the average of downstream and upstream velocities). Default is "it".
        :param radFor: string representing the radial-equilibrium formulation. Either 'radEq' for forcing radial-equilibrium-equation condition, or 'Bessel' for forcing Bessel-based formulation. Default is 'radEq'.
        :param dynamic_interp: boolean flag determining whether a dynamic interpolation is to be performed. Default is True.
        
        **kwargs**:
        :param x_idx: index of the axial stage at which to compute the outlet velocity field. Default is 0, which means that the outlet velocity is computed at the immediate downstream section of the turbine's rotor (infinitesimal displacement).
        '''

        #--------------------------------------------------------------------------------------------------------------------#
        #------------------------------------------------------ASSERTIONS----------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        # Assertion for checking that the input parameter matches an accepted value.
        assert inputparam in ["phi", "q", "vx", "it"], "Provide a valid inputparam mode: 'phi', 'q', 'vx' or 'it'."
        # Assertion for checking that the string representing the referential velocity "refvel" has a valid value.
        assert refvel in ["it", "ave"], "Provide a valid 'refvel' value: 'it' or 'ave'."
        # Conditional for asserting that the input parameter has the same length of the array of blade elements.
        if isinstance(phi_q_v, np.ndarray) or isinstance(phi_q_v, list):
            assert len(phi_q_v) == len(self.geom.r), 'In case the input variable is meant to be an array  provide it so that its length equals the number of blade elements. Input array has a length of ' + str(len(phi_q_v)) + ', and number of blade elements is ' + str(len(self.geom.r)) + '.'
            
        #--------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------------------BODY-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#             

        # Changing the type of input data to float, in case it is necessary.
        if isinstance(phi_q_v, np.ndarray):
            # Action in case the input data is in np.ndarray format.
            if "float" not in phi_q_v.dtype.name:
                phi_q_v = phi_q_v.astype(float)
            else:
                pass
        elif isinstance(phi_q_v, list):
            # Action in case the input data is in list format.
            phi_q_v = [float(_) for _ in phi_q_v]
        else:
            # Action in case the input data is in single variable format, but not float (integer possibly).
            if type(phi_q_v) != type(float):
                phi_q_v = float(phi_q_v)
        # Setting the value of the 'x_idx' parameter.
        x_idx = kwargs['x_idx'] if 'x_idx' in kwargs.keys() else 0
        
        # Upstream velocity-triangle calculation.
        Um, phi, dq_aux, vx, v, w = self.it_triangle_solve(phi_q_v=phi_q_v, omega=omega, inputparam=inputparam, refvel=refvel)
        # Reynolds calculation.
        self.reynolds_calculation(vel=w)

        # Loop running over the inlet velocity triangle's relative angle values, performing bilinear-interpolation of cl and cd data.
        for e, _ in enumerate(self.it.gamma):
            # Recasting the relative angle-of-attack into degrees, for comparison with the angles-of-attack stored in the polars database.
            _ *= 180/np.pi            
            cl, cd = self.cl_cd_interp(gamma=_, blade_index=e)
            self.lcoefs.cl[e] = cl
            self.lcoefs.cd[e] = cd        
            
        # Performing the curve-fitting if the value of the flow parameter is not 0 (null lift value).
        if phi != 0:
            # Conditional for ruling out if the dynamic-interpolation-based fitting is to be performed.
            if dynamic_interp:
                # Lift-fitting.
                self.lcoefs.cl = mt.functionpp.dynamicFitting(self.it.gamma*180/np.pi, self.lcoefs.cl, 2, resmin=1e-4)
                # Drag-fitting.     
                self.lcoefs.cd = mt.functionpp.dynamicFitting(self.it.gamma*180/np.pi, self.lcoefs.cd, 2, resmin=1e-4)     

        # Computation of differential loads, induced drag, thrust and tangential-force coefficients.
        self.differential_load_calculation(vel=w)

        # Computation of interference-corrected (solidity-based semi-empirical correction) thrust and tangential-force coefficients.
        self.cif_correction()

        # Conditional for passing in case the inlet relative angle-of-attack (beta) is null (no flow).
        if 0 in self.it.beta:
            pass
        else:            
            # Downstream velocity-triangle calculation.
            self.ot_triangle_solve(omega=omega, radFor=radFor, x_idx=x_idx)

        # Calculation of torques, powers, pressure drops and efficiencies.
        self.load_calculation(omega=omega, rel_vel=w, axial_vel_ref=vx, total_vel_ref=v, dq_aux=dq_aux)
            
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------RETURN------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#             
            
        # Return statement.
        return

    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------BEM_wells_ad() METHOD----------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    def BEM_wells_ad(self, phi_q_v: Union[list, np.ndarray, float], omega: float, inputparam: str="vx", refvel: str='it', radFor: str='radEq', maxiter: int=100,
                     dynamic_interp: bool=True, **kwargs) -> None:
        '''Applies the BEM coupled with the actuator disc model to a Wells turbine stage.

        **parameters**:
        :param ts: instance of the turbstage class to which the BEM is to be applied.
        :param phi_q_v: input parameter, either a flow coefficient (phi) a flow-rate value (q) or an axial velocity value (vx). It may be either an array-like structure, specifying phi, q or v for each blade element, or a single value, in which the provided value is set equal to all the blade elements.
        :param omega: rotational speed of the turbine [rad/s].
        :param inputparam: string representing the type of input provided; it is "phi" for indicating a flow coefficient, "q" for a flow-rate, "vx" for an axial velocity or "it" for a pre-established inlet velocity triangle. Default is "vx".
        :param refvel: string representing the referential velocity employed for calculating loads and velocity-dependent outputs; it is "it" for setting the referential velocity equal to the relative velocity magnitude of the inlet velocity triangle, or 'ave' for setting it equal to the relative velocity magnitude based on the actuator disk value (the axial component being the average of downstream and upstream velocities). Default is "it".
        :param radFor: string representing the radial-equilibrium formulation. Either 'radEq' for forcing radial-equilibrium-equation condition, or 'Bessel' for forcing Bessel-based formulation. Default is 'radEq'.
        :param maxiter: integer representing the maximum number of iterations to perform. Default is 100.
        :param dynamic_interp: boolean flag determining whether a dynamic interpolation is to be performed. Default is True.
        
        **kwargs**:
        :param x_idx: index of the axial stage at which to compute the outlet velocity field. Default is 0, which means that the outlet velocity is computed at the immediate downstream section of the turbine's rotor (infinitesimal displacement).        
        '''

        #--------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------------------BODY-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#        
        
        if radFor in ['radEq', 'radEq2']:            
            # Calling the BEM_wells_noad() method in the first step. The 'refvel' magnitude is set to 'it',
            # as the referential magnitude is taken from inlet velocity triangle in the first step.
            self.BEM_wells_noad(phi_q_v, omega, inputparam, refvel='it', radFor=radFor, dynamic_interp=dynamic_interp, **kwargs)
            # Instantiating the 'diff_powout' variable and setting its value to 1; it is used for ensuring energy convservation convergency.
            diff_powout = 1
            # Instantiating the 'powout' variable and setting it equal to the total-to-total pressure drop.
            powout = self.eout.power + self.eout.pvisc + self.eout.pk
            # While loop for ensuring energy consrevation.
            ind = 0
            while diff_powout > 1e-4:    
                # Calling the BEM_wells_noad() method with the 'refvel' parameter set to 'ave'.
                self.BEM_wells_noad(phi_q_v, omega, inputparam="it", refvel="ave", radFor=radFor, dynamic_interp=dynamic_interp, **kwargs)
                # Recomputing the 'diff_powout' value.
                diff_powout = np.abs(powout - (self.eout.power + self.eout.pvisc + self.eout.pk))
                # Reassigning the 'powout' value to current step's total-to-total pressure drop.
                powout = self.eout.power + self.eout.pvisc + self.eout.pk
                ind += 1
                if ind > maxiter:
                    break
        else:
            # Calling the BEM_wells_noad() method in the first step. The 'refvel' magnitude is set to 'it',
            # as the referential magnitude is taken from inlet velocity triangle in the first step.
            self.BEM_wells_noad(phi_q_v, omega, inputparam, refvel='it', radFor=radFor, dynamic_interp=dynamic_interp, **kwargs)
                
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------RETURN------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        # Return statement.
        return
    
#--------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------class turb--------------------------------------------------#             
#--------------------------------------------------------------------------------------------------------------------#
class turb:
    '''Class storing overall geometrical and analytical information about a turbine.'''
    
    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------DEFINING METHODS-------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    
    #--------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------__init__() METHOD-------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#  
    def __init__(self) -> None:
        '''Initializes a given instance of a class containing a turbine.'''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------DECLARING ATTRIBUTES-----------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        # Number of stages.
        self._numstages = 0
        # Types of stages.
        self._typestages = list()
        # Stages list.
        self._stages = list()
        # Global dimensional torque.
        self._torque = list()
        # Dimensoinal torques.
        self._torque_stages = list()
        # Global dimensional power.
        self._power = list()
        # Dimensional powers.
        self._power_stages = list()
        # Global static-to-static pressure loss.
        self._pst = list()
        # Static-to-static pressure losses.
        self._pst_stages = list()
        # Global total-to-total pressure loss.
        self._ptt = list()
        # Total-to-total pressure losses.
        self._ptt_stages = list()
        # Dimensionless flow-rates.
        self._Phi = list()
        # Global dimensionless powers.
        self._Pi = list()
        # Dimensionless powers correspodning to each stage.
        self._Pi_stages = list()
        # Global dimensionless pressure losses.
        self._Psi = list()
        # Dimensionless pressure losses correspodning to each stage.
        self._Psi_stages = list()
        # Global efficiency.
        self._Eff = list()
        # Efficiencies corresponding to each stage.
        self._Eff_stages = list()
        
    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------DECLARING PROPERTIES-----------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------# 
    
    # Declaration of get property numstages for getting the number of stages.
    @property
    def numstages(self) -> int:
        return self._numstages
    # Declaration of get property typestages for getting the list of the types of stages.
    @property
    def typestages(self) -> list:
        return self._typestages
    # Declaration of get property stages for getting the list of stages.
    @property
    def stages(self) -> list:
        return self._stages
    # Declaration of get property torque for getting the glboal dimensional torque.
    @property
    def torque(self) -> list:
        return self._torque
    # Declaration of get property torque_stages for getting the dimensional torques corresponding to each stage.
    @property
    def torque_stages(self) -> list:
        return self._torque_stages
    # Declaration of get property power for getting the glboal dimensional torque.
    @property
    def power(self) -> list:
        return self._power
    # Declaration of get property power_stages for getting the dimensional powers corresponding to each stage.
    @property
    def power_stages(self) -> list:
        return self._power_stages    
    # Declaration of get property pst for getting the glboal dimensional static-to-static pressure loss.
    @property
    def pst(self) -> list:
        return self._pst
    # Declaration of get property pst_stages for getting the dimensional static-to-static pressure losses corresponding to each stage.
    @property
    def pst_stages(self) -> list:
        return self._pst_stages        
    # Declaration of get property ptt for getting the glboal dimensional total-to-total pressure loss.
    @property
    def ptt(self) -> list:
        return self._ptt
    # Declaration of get property ptt_stages for getting the dimensional total-to-total pressure losses corresponding to each stage.
    @property
    def ptt_stages(self) -> list:
        return self._ptt_stages            
    # Declaration of get property Phi for getting the list of dimensionless flow-rates.
    @property
    def Phi(self) -> list:
        return self._Phi
    # Declaration of get property Pi for getting the list of global dimensionless powers.
    @property
    def Pi(self) -> list:
        return self._Pi
    # Declaration of get property Pi_stages for getting the list of dimensionless powers corresponding to each stage.
    @property
    def Pi_stages(self) -> list:
        return self._Pi_stages
    # Declaration of get property Psi for getting the list of global dimensionless pressure losses.
    @property
    def Psi(self) -> list:
        return self._Psi
    # Declaration of get property Psi_stages for getting the list of dimensionless pressure losses corresponding to each stage.
    @property
    def Psi_stages(self) -> list:
        return self._Psi_stages
    # Declaration of get property Eff for getting the list of global efficiencies.
    @property
    def Eff(self) -> list:
        return self._Eff
    # Declaration of get property Eff_stages for getting the list of efficiencies corresponding to each stage.
    @property
    def Eff_stages(self) -> list:
        return self._Eff_stages       
    
    # Declaration of set property torque for global dimensional torque list.
    @torque.setter 
    def torque(self, value: list) -> None:    
        self._torque = value
        return    
    # Declaration of set property torque_stages for dimensional torques list corresponding to each stage.
    @torque_stages.setter 
    def torque_stages(self, value: list) -> None:    
        self._torque_stages = value
        return
    # Declaration of set property power for global dimensional power list.
    @power.setter 
    def power(self, value: list) -> None:    
        self._power = value
        return    
    # Declaration of set property power_stages for dimensional powers list corresponding to each stage.
    @power_stages.setter 
    def power_stages(self, value: list) -> None:    
        self._power_stages = value
        return
    # Declaration of set property power pst global dimensional static-to-static pressure loss list.
    @pst.setter 
    def pst(self, value: list) -> None:    
        self._pst = value
        return    
    # Declaration of set property pst_stages for dimensional static-to-static pressure loss list corresponding to each stage.
    @pst_stages.setter 
    def pst_stages(self, value: list) -> None:    
        self._pst_stages = value
        return
    # Declaration of set property power ptt global dimensional total-to-total pressure loss list.
    @ptt.setter 
    def ptt(self, value: list) -> None:    
        self._ptt = value
        return    
    # Declaration of set property ptt_stages for dimensional total-to-total pressure loss list corresponding to each stage.
    @ptt_stages.setter 
    def ptt_stages(self, value: list) -> None:    
        self._ptt_stages = value
        return    
    # Declaration of set property Phi for dimensionless flow-rate list.
    @Phi.setter 
    def Phi(self, value: list) -> None:    
        self._Phi = value
        return
    # Declaration of set property Pi for global dimensionless power list.
    @Pi.setter 
    def Pi(self, value: list) -> None:    
        self._Pi = value
        return
    # Declaration of set property Pi_stages for dimensionless power list corresponding to each stage.
    @Pi_stages.setter 
    def Pi_stages(self, value: list) -> None:    
        self._Pi_stages = value
        return
    # Declaration of set property Psi for global dimensionless pressure loss list.
    @Psi.setter 
    def Psi(self, value: list) -> None:    
        self._Psi = value
        return
    # Declaration of set property Psi_stages for dimensionless pressure loss list corresponding to each stage.
    @Psi_stages.setter 
    def Psi_stages(self, value: list) -> None:    
        self._Psi_stages = value
        return    
    # Declaration of set property Eff for global efficiency list.
    @Eff.setter 
    def Eff(self, value: list) -> None:    
        self._Eff = value
        return
    # Declaration of set property Eff_stages for efficiency list corresponding to each stage.
    @Eff_stages.setter 
    def Eff_stages(self, value: list) -> None:    
        self._Eff_stages = value
        return
    
    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------DEFINING METHODS-------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------------------#     
    
    #--------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------reset_props() METHOD------------------------------------------------# 
    #--------------------------------------------------------------------------------------------------------------------#
    def reset_props(self) -> None:
        '''Resets the output global and stage-wise properties to empty lists.
        
        Necessary whenever a turbine stage is added or removed, or the BEM method re-applied to a turbine instance.       
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------BODY--------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#        
        
        for prop_root_to_reset in ["torque", "power", "pst", "ptt", "Phi", "Pi", "Psi", "Eff"]:
            props_to_reset = [_ for _ in dir(self) if prop_root_to_reset in _]
            for prop_to_reset in props_to_reset:
                setattr(self, prop_to_reset, list())
    
    #--------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------create_or_add_stage() METHOD--------------------------------------------# 
    #--------------------------------------------------------------------------------------------------------------------#
    def create_or_add_stage(self,
                 pos: int=-1,
                 mode: str='create',
                 ts: Union[None, turbstage]=None,
                 ttype: turbCnt.TYPES=turbCnt.TYPES.Wells,
                 omega: float=3600,                 
                 N: int=50,
                 rcas: float=0.25,
                 hub_to_tip_ratio: float=0.75,
                 chord: Union[list, np.ndarray, float]=0.117,
                 angpitch: Union[list, np.ndarray, float]=0,
                 airfoil: Union[geCnt.PROFILES, np.ndarray, str]=geCnt.PROFILES.NACA0015,
                 tip_percent: float=0.5,
                 Z: int=7,
                 p: float=101325,
                 T: float=288.15,
                 R: float=287.058,
                 nu: float=1.81e-5) -> None:
        '''Creates or adds a turbine stage on the turbine, provided the input parameters for the stage.                
        
        **parameters**:
        :param pos: position on which to create the turbine stage. Default is -1, which meansthat the turbine stage is appended to the existing ones (if no turbine stage is present, then the turbine stage is set in the first position). Set it to 0 to prepend a stage.
        :param cr_or_add: string representing if a turbine stage is to be created ('create'), or added ('add'). Default is 'create'.
        :param ts: a turbstage instance in case an existing turbine stage is to be added. Default is None.
        :param ttype: string specifying turbine type; either 'Wells' or 'Impulse'. Default is 'Wells'.
        :param omega: rotational speed of turbine stage, in [RPM]. Default is 3600 [RPM].
        :param N: number of blade elements (discretization). Default is 50.
        :param rcas: casing radius. Default is 0.25.
        :param hub_to_tip_ratio: hub to tip ratio value. Default is 0.75.
        :param chord: chord value. Default is 0.117. It can be an element-wise array.
        :param angpitch: angular pitch value. Default is 0. It can be an element-wise array.
        :param airfoil: geometry of the airfoils of the blade. Default is NACA0015, meaning that the airfoil geometry is constant throughout the blade. It can be an element-wise array.
        :param tip_percent: tip gap value, in terms of chord percentage. Default is 0.5%.
        :param Z: number of blades. Default is 7.
        :param p: atmospheric pressure in [Pa]. Default value is 101325 [Pa].
        :param T: ambient temperature in [K]. Default value is 288.15 [K].
        :param R: gas constant in [J/(kg·K)]. Default value is 287.058 [J/(kg·K)]
        :param nu: gas viscosity in [kg/(m·s)]. Default value is 1.81e-5 [kg/(m·s)]                 
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        # Asserting if the provided input parameter 'mode' has a valid value.
        assert mode in ["create", "add"], "Please provide a valid mode input, either 'create' or 'add'."

        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------BODY--------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#        
        
        # Emptying output property lists.
        self.reset_props()
        
        # Conditional for instantiating 'numlist' variable. Just done if the number of stages is > 1.
        if self._numstages != 0:            
            # If the 'pos' variable is not set to append the created turbine stage, then 'numlist' must equal the stages beginning from 'pos + 1' and ending in the number of stages.
            if pos != -1:
                numlist = [_ + 1 for _ in range(pos, self._numstages)]
            # Otherwise, the numlist is an empty list.
            else:
                numlist = list()
        
        # If the 'pos' variable is set to append the created turbine stage, then 'num' must equal the number of stages + 1.
        if pos == -1:
            num = self._numstages + 1
        # Otherwise, it is necessary to overright all the stages that are after the position at which the stage to be created is to be inserted.
        else:
            # Conditional for ensuring that the number of stages is equal or greater than 1.
            if self._numstages != 0:
                # The 'num' variable is set to the first entry of 'numlist' in this case.
                num = numlist[0]
                # Loop running over the reverse 'numlist'. It gets the turbine stages in reverse order, form the last one to the 'pos + 1' variable.
                for i in numlist[::-1]:                    
                    # Getting the turbine stage instance.
                    ts = getattr(self, "turbstage" + str(i))
                    # Setting a property named 'turbstage(i+1)', and setting it equal to the turbine stage instance.
                    setattr(self, "turbstage" + str(i + 1), ts)
                    # Deleting turbine stage instance and corresponding output properties.
                    delattr(self, "turbstage" + str(i))
                    # Deleting instance from '_stages' list.
                    del self._stages[-1]
                
        # Conditional for either creating a turbine stage instance or adding an existing one.
        if mode == 'create':
            # If 'mode' is set to 'create', creating turbine stage instance.
            ts = turbstage(ttype=ttype,
                               omega=omega,                 
                               N=N,
                               rcas=rcas,
                               hub_to_tip_ratio=hub_to_tip_ratio,
                               chord=chord,
                               angpitch=angpitch,
                               airfoil=airfoil,
                               tip_percent=tip_percent,
                               Z=Z,
                               p=p,
                               T=T,
                               R=R,
                               nu=nu)
        else:
            # Otherwise, checking wheter the input turbine stage instance is valid.
            if ts == None:
                # If it is not, raising a type error.
                raise TypeError('No turbine stage provided to add. Please provide an existing turbine stage')
            else:
                # Otherwise, setting the 'ts' variable equal to the input turbine stage value.
                ts = ts
        
        # Setting an attribute for the created turbine instance, and setting it to that instance.
        setattr(self, "turbstage" + str(num), ts)
        # Appending the created turbine instance to '_stages'.
        self._stages.append(ts)
        # Conditional for appending the deleted turbine instances to '_stages'.
        if pos == -1:
            # 'pass' command in case the 'pos' variable is set to append the the created turbine stage to the existing stages.
            pass
        elif self._numstages != 0 and pos != -1:
            # If 'numlist' is not an empty list, it means that the created stage has been introduced somewhere between the previous stages, so that those existing stages where removed from the list.
            if len(numlist) != 0:
                for i in numlist:
                    # Appending the removed turbine stages to the '_stages' list.
                    self._stages.append(getattr(self, "turbstage" + str(i + 1)))
            # Otherwise, just appending the removed turbine stage to the '_stages' list.
            else:
                self._stages.append(getattr(self, "turbstage" + str(num)))
        # Updating the number of stages.
        self._numstages += 1
        # Updating the type of stages.
        self._typestages.append(self._stages[-1].ttype)
        
        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------------RETURN-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#        
        
        # Return statement.
        return

    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------remove_stage() METHOD------------------------------------------------# 
    #--------------------------------------------------------------------------------------------------------------------#
    def remove_stage(self, ind: int) -> None:
        '''Removes an existing turbine stage from the turbine.
        
         **parameters**:
        :param ind: integer determining the index of the list of stages that is removed.
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        # Asserting that the number of stages is larger than 0; that is, that there are turbine stage instances on the current turbine instance.
        assert self._numstages != 0, "There are no turbine stage instances in the current turbine instance. Skipping removal action."
        # Getting the maximum turbine stage number.
        maxturbstage = int([_ for _ in dir(self) if "turbstage" in _][-1].split("turbstage")[-1])
        # Asserting that the provided index to remove falls within the existing number of turbine stages.
        assert 0 < ind < maxturbstage, "The provided index does not fall within the existing number of turbine stages. Provide an index value between 1 and " + str(maxturbstage) + "."
        
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------BODY--------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#         
        
        # Emptying output property lists.
        self.reset_props()
        
        # The 'numlist' equals the stages beginning from 'ind + 1' and ending in the number of stages.            
        numlist = [_ + 1 for _ in range(ind, self._numstages - 1)]
        
        # Removing the turbine stage from the stages.
        del self._stages[ind - 1]
        # Deleting referred encapsulated turbine instance and its corresponding property.
        delattr(self, "turbstage" + str(ind))
        # Updating the number of stages.
        self._numstages -= 1
        # Updating the type of stages.
        del self._typestages[ind - 1]
        
        # Loop over 'numlist' for updating the numbers of the turbine stage instances and their corresponding properties
        for i in numlist:
            # Getting the turbine stage instance.
            ts = getattr(self, "turbstage" + str(i))
            # Setting a property named 'turbstage(i-1)', and setting it equal to the turbine stage instance.
            setattr(self, "turbstage" + str(i - 1), ts)
            # Deleting referred encapsulated turbine instance and its corresponding property.
            delattr(self, "turbstage" + str(i))
            
        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------------RETURN-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#             
            
        # Return statement.
        return
    
    #--------------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------BEM() METHOD----------------------------------------------------# 
    #--------------------------------------------------------------------------------------------------------------------#
    def BEM(self,
            phi_q_vs: Union[list, np.ndarray, float],
            inpres: Union[list, np.ndarray, float]=None,
            intemp: Union[list, np.ndarray, float]=None,
            inputparam: str='vx',
            mode: str='3DAD',
            radFor: str='radEq',
            reset: bool=True,
            dynamic_interp: bool=True,
            **kwargs) -> None:
        '''Applies the momentum element method to the turbine.
        
        **parameters**:
        :param phi_q_v: input parameter list for which the BEM method is applied. They can be either flow coefficient values (phi), flow-rate values (q) or axial velocity values (vx).
        :param inpres: input pressure list. Default is 'None', which means that a standard value is provided (101325 Pa).
        :param intemp: input temperature list. Default is 'None', which means that a standard value is provided (288.15 K).
        :param inputparam: string determining the type of input parameter. It can be either 'phi' (stipulating flow coefficient), 'flowrate' (flow-rate) or 'vx'
        (axial velocity). Default is 'vx'.
        :param mode: string determining the model of BEM to be applied. It can be either '3DNOAD' (semi-3D BEM model without actuator disk) or '3DAD' (semi-3D BEM model with actuator disk). Default is '3DAD'.
        :param radFor: string representing the radial-equilibrium formulation. Either 'radEq' for forcing radial-equilibrium-equation condition, or 'Bessel' for forcing Bessel-based formulation. Default is 'radEq'.
        :param reset: resets or empties the computed properties so far. Default is True.
        :param dynamic_interp: boolean flag determining whether a dynamic interpolation is to be performed. Default is True.
        
        **kwargs**:
        :param x_idx: index of the axial stage at which to compute the outlet velocity field. Default is 0, which means that the outlet velocity is computed at the immediate downstream section of the turbine's rotor (infinitesimal displacement).        
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        # Asserting that the input pressure parameter matches in type and length with the input parameter.
        if not isinstance(inpres, type(None)):
            assert(type(phi_q_vs) == type(inpres)), "Please provide an input pressure type that matches the input paramter type."
            if isinstance(inpres, np.ndarray) or isinstance(inpres, list):
                assert(len(inpres) == len(phi_q_vs)), "Please provide an input pressure list/array of the same length of the input parameter list/array."
        # Asserting that the input temperature parameter matches in type and length with the input parameter.
        if not isinstance(intemp, type(None)):
            assert(type(phi_q_vs) == type(intemp)), "Please provide an input temperature type that matches the input paramter type."
            if isinstance(intemp, np.ndarray) or isinstance(intemp, list):
                assert(len(intemp) == len(phi_q_vs)), "Please provide an input temperature list/array of the same length of the input parameter list/array."
                
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------BODY--------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        # Recasting float input types to list types.
        if isinstance(phi_q_vs, float):
            phi_q_vs = [phi_q_vs]
            if inpres != None:
                inpres = [inpres]
            if intemp != None:
                intemp = [intemp]
        
        # Emptying output property lists.
        if reset:
            self.reset_props()        
        
        # Getting the list of turbine stages in cardinal order.
        stageproplist = sorted([_ for _ in dir(self) if "turbstage" in _], key=lambda x: int(x.split("turbstage")[-1]))
        
        # Declaring energetic-output-related lists of dimensional variables.
        torque = list()
        power = list()
        pst = list()
        ptt = list()
        
        # Declaring energetic-output-related lists of dimensionless variables.
        Phi = list()
        Pi = list()
        Psi = list()
        Eff = list()
        
        # Loop running over the set of input parameters.
        for phi_q_v in phi_q_vs:
            # Declaring auxiliar variables for computing the global energetic output values.
            inpower = 0
            outpower = 0
            torqueaux = 0
            pstaux = 0
            pttaux = 0
            Piaux = 0            
            Psiaux = 0            
            Effaux = 0
            # Declaring auxiliar lists for storing the dimensional energetic outputs of each turbine stage.
            torque_l = list()
            power_l = list()
            pst_l = list()
            ptt_l = list()
            # Declaring auxiliar lists for storing the dimensionless energetic outputs of each turbine stage.
            Pi_l = list()
            Psi_l = list()
            Eff_l = list()
            # Loop running over the turbine stages and applying the BEM method accordingly.
            for e, stageprop in enumerate(stageproplist):
                # Getting the current turbine stage.
                ts = getattr(self, stageprop)
                # Conditional for determining whether the current stage is the first stage.
                if e == 0:
                    # Setting the input pressures and temperatures.
                    if not isinstance(inpres, type(None)):
                        ts.flow.p = inpres[e]
                    if not isinstance(intemp, type(None)):
                        ts.flow.T = intemp[e]
                    # If it is, applying the BEM method with 'inputparam' set to the input parameter value.
                    ts.BEM(phi_q_v=phi_q_v, inputparam=inputparam, mode=mode, radFor=radFor, dynamic_interp=dynamic_interp, **kwargs)
                    # Appending the dimensionless flow-rate varaible to the 'Phi' list.
                    Phi.append(ts.it.Phi)
                else:
                    # If it is not, then getting the previous turbine stage instance.
                    tsprev = getattr(self, stageproplist[e - 1])
                    # Computing the input pressure to the current stage by substracting the total-to-total pressure losses to the previous stage's input pressure.
                    ts.flow.p = tsprev.flow.p - tsprev.eout.p_tot_to_tot
                    # Performing a deep copy of the previous stage's output velocity triangle, and setting it equal to the current stage's input triangle.
                    ts.it = copy.deepcopy(tsprev.ot)
                    # Calling the 'BEM' method for the current stage with 'inputparam' set to 'it'.
                    ts.BEM(phi_q_v=phi_q_v, inputparam='it', mode=mode, radFor=radFor, dynamic_interp=dynamic_interp)                
                # Adding the dimensionless energetic outputs to the auxiliar variables.
                Piaux += ts.eout.Pi                
                Psiaux += ts.eout.Psi   
                # Adding the dimensional energetic outputs to the auxiliar variables.
                inpower += ts.eout.pvisc + ts.eout.pk + ts.eout.power
                outpower += ts.eout.power
                torqueaux += ts.eout.tau                
                pstaux += ts.eout.p_st_to_st
                pttaux += ts.eout.p_tot_to_tot
                # Appending the dimensional energetic output of the turbine stage to the corresponding list.
                torque_l.append(ts.eout.tau)
                power_l.append(ts.eout.power)
                pst_l.append(ts.eout.p_st_to_st)
                ptt_l.append(ts.eout.p_tot_to_tot)
                # Appending the dimensionless energetic outputs of the turbine stage to the corresponding list.
                Pi_l.append(ts.eout.Pi)
                Psi_l.append(ts.eout.Psi)
                # Conditional for setting the efficiency value to 0 in case the stage's value falls outside the range [0, 1].
                if 0 <= ts.eout.eff <= 1:
                    Eff_l.append(ts.eout.eff)
                else:
                    Eff_l.append(0)
            # Appending the computed global dimensional energetic outputs to the corresponding lists.
            torque.append(torqueaux)
            power.append(outpower)
            pst.append(pstaux)
            ptt.append(pttaux)
            # Appending the computed global dimensionless energetic outputs to the corresponding lists.
            Pi.append(Piaux)
            Psi.append(Psiaux)
            # Conditional for setting the efficiency value to 0 in case the computed value falls outside the range [0, 1].
            if 0 <= outpower/inpower <= 1:
                Eff.append(outpower/inpower)
            else:
                Eff.append(0)
            # Adding the list of each turbine stage's dimensional energetic outputs to the corresponding output variable.
            self.torque_stages.append(torque_l)
            self.power_stages.append(power_l)
            self.pst_stages.append(pst_l)
            self.ptt_stages.append(ptt_l)
            # Adding the list of each turbine stage's dimensionless energetic outputs to the corresponding output variable.
            self.Pi_stages.append(Pi_l)
            self.Psi_stages.append(Psi_l)
            self.Eff_stages.append(Eff_l)
        
        if reset:
            # Transposing list of lists.        
            self.torque_stages = [list(i) for i in zip(*self.torque_stages)]
            self.power_stages = [list(i) for i in zip(*self.power_stages)]
            self.pst_stages = [list(i) for i in zip(*self.pst_stages)]
            self.ptt_stages = [list(i) for i in zip(*self.ptt_stages)]
            self.Pi_stages = [list(i) for i in zip(*self.Pi_stages)]
            self.Psi_stages = [list(i) for i in zip(*self.Psi_stages)]
            self.Eff_stages = [list(i) for i in zip(*self.Eff_stages)]
            # Setting the global dimensional energetic output properties of the turbine to the stored lists. 
            self.torque = torque
            self.power = power
            self.pst = pst
            self.ptt = ptt
            # Setting the global dimensionless energetic output properties of the turbine to the stored lists.
            self.Phi = Phi
            self.Pi = Pi
            self.Psi = Psi
            self.Eff = Eff
        else:
            # Transposing list of lists.        
            self.torque_stages += [list(i) for i in zip(*self.torque_stages)]
            self.power_stages += [list(i) for i in zip(*self.power_stages)]
            self.pst_stages += [list(i) for i in zip(*self.pst_stages)]
            self.ptt_stages += [list(i) for i in zip(*self.ptt_stages)]
            self.Pi_stages += [list(i) for i in zip(*self.Pi_stages)]
            self.Psi_stages += [list(i) for i in zip(*self.Psi_stages)]
            self.Eff_stages += [list(i) for i in zip(*self.Eff_stages)]
            # Setting the global dimensional energetic output properties of the turbine to the stored lists. 
            self.torque += torque
            self.power += power
            self.pst += pst
            self.ptt += ptt
            # Setting the global dimensionless energetic output properties of the turbine to the stored lists.
            self.Phi += Phi
            self.Pi += Pi
            self.Psi += Psi
            self.Eff += Eff

    #--------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------get_localParam_gradDir() METHOD-------------------------------------------# 
    #--------------------------------------------------------------------------------------------------------------------#    
    def get_localParam_gradDir(self,
                               oper_point,
                               oper_param: str='vx',
                               grad_init_point: float=0,
                               grad_point_step: float=1,
                               grad_param: str='angpitch',
                               param: str='Pi') -> tuple:
        '''Computes the maximum gradient's direction of a specific parameter of the turbine at a given operation point.

        **parameters**:
        :param oper_point: operational point of the turbine at which the local optimization is to be carried out. It may be either an axial velocity, a flow-rate or a flow-coefficient value.
        :param oper_param: string determining the type of data passed as operational point. It may be either 'vx' (axial velocity), 'q' (flow-rate) or 'phi' (flow coefficient). Default is 'vx'.
        :param grad_init_point: initial point of the gradient-based scheme from which the searching is began. Default is 0.
        :param grad_point_step: step of the gradient-based scheme for performing the searching. Default is 1.
        :param grad_param: parameter upon which to perform the gradient-based searching. It must be a geometrical parameter of the turbine. Default is 'angpitch', meaning that the angular pitch is employed as a gradient-based searching parameter.
        :param param: turbine outcome parameter upon which to perform the optimization (the gradient-based searching is performed so that a maximum of the 'param' parameter is targeted). Default is 'Pi', meaning that the dimensionless power is maximized. It must be either 'Pi', 'Psi' or 'Eff'.
        
        **return**:
        :return: a 2-tuple containing the maximum value of the maximizing parameter found in the neighbourhood of the operation point, and the sign of the gradient marking the direction to which the searching is to be performed (1 for a positive-valued searching upon the 'grad_param' parameter; -1 for a negative-valued searching).
        
        **rtype**:
        :rtype: tuple.
        '''

        #--------------------------------------------------------------------------------------------------------------------#
        #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#   
        
        # Asserting that the provided 'grad_param' parameter is a geometrical parameter of the turbine.
        assert grad_param in dir(self.turbstage1.geom), "The provided gradient parameter ('grad_param') is not a geometrical parameter of the turbine. Please amend."
        # Asserting that 'param' is any of the ['Psi', 'Pi', 'Eff'].
        assert param in ["Pi", "Eff"], "The provided checking parameter ('param') is not any of the ['Pi', 'Eff'] values. Please amend."

        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------BODY--------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#        

        # Conditional for setting the value of the 'grad_param' attribute to the initial point value of the gradient scheme.        
        setattr(self.turbstage1.geom, grad_param, grad_init_point)
        # Computing the BEM for the current operational point and 'grad_param' value.
        self.BEM(phi_q_vs=[oper_point], inputparam=oper_param, reset=True)
        # Getting the corresponding 'param' value and setting it to the 'param0_val' variable.
        param0_val = getattr(self, param)[0]

        # Conditional for setting the value of the 'grad_param' attribute to the 'grad_point_step'-valued positively-displaced value of the gradient scheme.
        setattr(self.turbstage1.geom, grad_param, grad_init_point + grad_point_step)
        # Computing the BEM for the current operational point and 'grad_param' value.
        self.BEM(phi_q_vs=[oper_point], inputparam=oper_param, reset=True)
        # Getting the corresponding 'param' value and setting it to the 'param1_val' variable.
        param1_val = getattr(self, param)[0]

        # Setting the value of the 'grad_param' attribute to the 'grad_point_step'-valued negatively-displaced value of the gradient scheme.          
        setattr(self.turbstage1.geom, grad_param, grad_init_point - grad_point_step)
        # Computing the BEM for the current operational point and 'grad_param' value.
        self.BEM(phi_q_vs=[oper_point], inputparam=oper_param, reset=True)
        # Getting the corresopnding 'param' value and setting it to the 'param2_val' variable.
        param2_val = getattr(self, param)[0]

        # Conditional tree for determining the maximum value within the ['param0_val', 'param1_val', 'param2_val'] set.
        if param1_val > param0_val:
            # If 'param1_val' is the maximum value, then setting the 'grad_param' attribute to the positively-displaced value and the 'param_maxval' outcome to 'param1_val'; additionally, the 'grad_sgn' outcome is set to 1.
            setattr(self.turbstage1.geom, grad_param, grad_init_point + grad_point_step)
            param_maxval = param1_val
            grad_sgn = 1
        elif param2_val > param0_val:
            # If 'param2_val' is the maximum value, then setting the 'grad_param' attribute to the negatively-displaced value and the 'param_maxval' outcome to 'param2_val'; additionally, the 'grad_sgn' outcome is set to -1.
            setattr(self.turbstage1.geom, grad_param, grad_init_point - grad_point_step)
            param_maxval = param2_val
            grad_sgn = -1
        else:
            # If 'param1_val' is the maximum value, then setting the 'grad_param' attribute to the initial point value and the 'param_maxval' outcome to 'param0_val'; additionally, the 'grad_sgn' outcome is set to 0. 
            setattr(self.turbstage1.geom, grad_param, grad_init_point)
            param_maxval = param0_val
            grad_sgn = 0

        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------------RETURN-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#    
        
        # Return statement.
        return param_maxval, grad_sgn

    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------searchParamMax_uponGradParam() METHOD----------------------------------------# 
    #--------------------------------------------------------------------------------------------------------------------#
    def searchParamMax_uponGradParam(self,
                                     oper_point: float,
                                     init_param_max: float,
                                     oper_param: str='vx',
                                     grad_init_point: float=0,
                                     grad_point_step: float=1,
                                     grad_step_dir: int=1,
                                     grad_param: str='angpitch',
                                     param: str='Pi',
                                     out: str='ext') -> tuple:
        '''It searches for a maximum of a given turbine outcome parameter upon modification of a geometrical parameter (gradient parameter) of the turbine.
        
        **parameters**:
        :param oper_point: operational point of the turbine at which the local optimization is to be carried out. It may be either an axial velocity, a flow-rate or a flow-coefficient value.
        :param init_param_max: initial value adopted by the parameter to be optimized, which is considered as the initial maximum value to be overriden by the following iterations.
        :param oper_param: string determining the type of data passed as operational point. It may be either 'vx' (axial velocity), 'q' (flow-rate) or 'phi' (flow coefficient). Default is 'vx'.        
        :param grad_init_point: initial point of the gradient-based scheme from which the searching is began. Default is 0.
        :param grad_point_step: step of the gradient-based scheme for performing the searching. Default is 1.
        :param grad_step_dir: integer showing the stepping direction adopted during the search. If 1, then the search is undertaken for positive displacements on the gradient parameter. If -1, then the search is undertaken for negative displacements on the gradient parameter. Default is 1.
        :param grad_param: parameter upon which to perform the gradient-based searching. It must be a geometrical parameter of the turbine. Default is 'angpitch', meaning that the angular pitch is employed as a gradient-based searching parameter.
        :param param: turbine outcome parameter upon which to perform the optimization (the gradient-based searching is performed so that a maximum of the 'param' parameter is targeted). Default is 'Pi', meaning that the dimensionless power is maximized. It must be either 'Pi' or 'Eff'.
        :param out: string determining the return type. It must be either 'red' for a reduced return type containing, merely, the optimum value of the parameter to be optimized; or 'ext' for an extended return (see below). Default is 'ext'.
        
        **return**:
        :return: a tuple containing the optimum value of the parameter to be optimized ('out == red') or a 5-tuple containing, respectively: the flow-coefficient, the optimum value of the parameter to be optimized, the pressure coefficient, the power coefficient, and the efficiency of the optimized configuration for the operation point considered ('out == ext').
        
        **rtype**:
        :rtype: tuple.
        '''

        #--------------------------------------------------------------------------------------------------------------------#
        #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#   
        
        # Asserting that the 'grad_param' variable representing the parameter whose variation is to determine the maximum outcome of the 'param' variable of the turbine corresponds to a geometrical parameter of the turbine.                   
        assert grad_param in dir(self.turbstage1.geom), "The provided gradient parameter ('grad_param') is not a geometrical parameter of the turbine. Please amend."
        # Asserting that the 'param' variable representing the outcome parameter of the turbine to be optimized corresponds to any of the outcome parameters available of the turbine.
        assert param in ["Pi", "Eff"], "The provided checking parameter ('param') is none of the ['Pi', 'Eff'] values. Please amend."
        # Asserting that the 'out' variable adopts an acceptable value.
        assert out in ["red", "ext"], "The provided output parameter ('out') is none of the ['red', 'ext'] values. Please amend."

        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------BODY--------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#              

        # Setting the boolean flag 'stop' to 'False'. It has the purpose of breaking the 'while' loop employed for the searching of the maximum.
        stop = False
        # Setting the 'grad_param' attribute of the turbine to its initial value, i.e. 'grad_init_point', as an initialization.
        setattr(self.turbstage1.geom, grad_param, grad_init_point)
        # Setting the value of 'param_max' to 'init_param_max' as an initialization.
        param_max = init_param_max

        # 'while' loop running over the succesively changing values of the geometrical parameter to be optimized.
        while not stop:
            # Getting the current value of the 'grad_param' attribute.
            current_val = getattr(self.turbstage1.geom, grad_param)
            # Varying the current value of the 'grad_param' attribute.
            setattr(self.turbstage1.geom, grad_param, current_val + grad_step_dir*grad_point_step)                
            # Launching the BEM for the current operation point with the newly set 'grad_param' attribute.
            self.BEM(phi_q_vs=[oper_point], inputparam=oper_param)
            # Getting the value of the parameter to be maximized.
            param_val = getattr(self, param)[0]

            # Conditional for checking whether the maximum has been reached. If it has (the current 'param_val' is less than 'param_max') then exiting the loop and proceeding forward.
            if param_val > param_max:
                param_max = param_val
            else:
                stop = True            

        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------------RETURN-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#                                

        # Conditional for determining whether the output gradient parameter of the current turbine is of a list-array type or not.
        if type(getattr(self.turbstage1.geom, grad_param)) in [list, np.ndarray]:
            # If it is, then setting the output 'out_grad_param' to a float value.
            out_grad_param = getattr(self.turbstage1.geom, grad_param)[0]
        else:
            # If it is, then just assigning its value to the output 'out_grad_param' variable.
            out_grad_param = getattr(self.turbstage1.geom, grad_param)                
        # Conditional tree for determining the return statement.        
        if out == 'ext':
            # Return statement for extended outcome.
            return self.Phi[0], out_grad_param, self.Psi[0], self.Pi[0], self.Eff[0]
        elif out == 'red':
            # Return statement for reduced outcome.
            return out_grad_param                            

    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------pitch_optimization() METHOD--------------------------------------------# 
    #--------------------------------------------------------------------------------------------------------------------#
    def pitch_optimization(self,
                            vxlist: list=list(range(1, 30)),   
                            init_point: float=0,
                            ang_init_step: float=1,
                            ang_final_step: float=0.1,
                            red_steps: int=1,
                            backfactor: float=1,                         
                            param: str='Pi',
                            show_progress: bool=True) -> tuple:
        '''Optimizes the pitch upon a given turbine so that it maximizes a given output coefficient for a set of input velocities.
        
        **parameters**:
        :param vxlist: input velocities at which to power-optimize the pitch. Default is a list of values ranging from 1 m/s to 30 m/s, with a step between adjacent input velocities of 1 m/s.
        :param init_point: initial point of the gradient-based scheme from which the searching is began. Default is 0.
        :param ang_init_step: initial angular step for beginning with the optimization. It is only considered when the 'opt_mode' parameter is 'fix', i.e. for an optimization carried out with a fix pitch angle. Default is 1.
        :param ang_final_step: final angular step when performing the optimization in the 'fix' mode. Default is 0.5.
        :param red_steps: number of reduction steps or refinements to be performed for going from 'ang_init_step' to 'ang_final_step' during the optimization executed in the 'fix' mode. Default is 1.
        :param backfactor: float that determines the initial pitch employed at the reduced steps. If 1, it means that the searching starts at the pitch value computed just before the pitch value for which the maximum value of the parameter to be optimized was obtained at the coarser step. The initial pitch value is defined by analogy for any other value of the backfactor. Default is 1.
        :param param: it specifies the parameter to upon which the optimization is based. It may be either 'Pi' (power) or 'Eff' (efficiency). Default is 'Pi'.
        :param show_progress: boolean specifying whether a progress bar is to be displayed while executing the method. Default is 'True'.
        
        **return**:
        :return: a tuple containing five lists: the computed dimensionless flow-coefficients, optimum angular pitches, optimum dimensionless pressure losses, optimum dimensionless powers and optimum efficiencies.
        
        **rtype**:
        :rtypes: tuple.
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#    
    
        # Asserting that 'stall_param' has a valid string value.
        assert param in ['Pi', 'Eff'], "Provide a valid 'param' value, either 'Pi' or 'Eff'."
        # Warning, if 'angpitch_init == angpitch_final', agains 'red_steps' being equal to 0.
        if ang_init_step == ang_final_step and red_steps == 0:            
            warning.warn("The initial and final searching angles are the same. Setting the 'red_steps' parameter value to 0.")
            red_steps = 0
    
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------BODY--------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#    
        
        # Creating auxiliary dimensionless flow-coefficient, pressure-coefficient, power-coefficient and efficiency for end-of-the-pitching-loop storage.
        Phis = list()
        Psis = list()
        Pis = list()
        Effs = list()
        Pitches = list()

        # Computing the reduction factor, i.e. 'red_factor'.
        red_factor = (ang_init_step/ang_final_step)**(1/red_steps) if red_steps != 0 else ang_init_step/ang_final_step

        # Tuned (progress-bar-showing, in case 'show_progress == True') 'for' loop for determining the optimum pitch angle at each operation point.
        for vx in tqdm(vxlist, disable=not show_progress):

            # Setting 'red_step' to 0.
            red_step = 0
            # Initializing the 'point_step' variable to 'angle_init_step'.
            point_step = ang_init_step

            # 'while' loop running over the required reduction steps and performing the optimization search.
            while red_step < red_steps + 1:

                # Conditional tree for determining the initial point for the search.
                if len(Pitches) == 0 and red_step == 0:
                    # If no previous optimal pitch angles have been computed (i.e. it is the first case) and it is the coarsest reduction step, then setting the 'init_point' variable to the 'init_point' value passed as an input argument.
                    init_point = init_point                        
                elif len(Pitches) != 0 and red_step == 0:
                    # If previous optimal pitch angles have been computed (i.e. it is not the first case) and it is the coarsest reduction step, then setting the 'init_point' variable to the last computed optimal pitch angle.                        
                    init_point = Pitches[-1]
                elif red_step != 0:
                    # If it is not the coarsest reduction step, then setting the 'init_point' variable based on the 'backfactor' variable.
                    init_point = Pitch - backfactor*sgn*point_step*red_factor

                # Calling the 'get_localParam_gradDir()' method for getting the searching direction and the initial maximum value for the search.
                init_param_max, sgn = self.get_localParam_gradDir(oper_point=vx, oper_param='vx', grad_init_point=init_point, grad_point_step=point_step, param=param)
                # Conditional for checking whether the required reduction steps have been performed.
                if red_step < red_steps:
                    # Conditional for setting the 'grad_init_point' value to its corresponding float-type value.
                    if type(self.turbstage1.geom.angpitch) in [list, np.ndarray]:
                        grad_init_point = self.turbstage1.geom.angpitch[0]
                    else:
                        grad_init_point = self.turbstage1.geom.angpitch                        
                    # If they have not, then calling the 'searchParamMax_uponGradParam()' with 'out=red' (just the optimal pitch angle is required for proceeding to the finer searching level).
                    Pitch = self.searchParamMax_uponGradParam(oper_point=vx, init_param_max=init_param_max, oper_param='vx', grad_init_point=grad_init_point, grad_point_step=point_step, grad_step_dir=sgn, param=param, out='red')
                    # Incrementing the number of reduction steps performed.
                    red_step += 1
                    # Redcuing the 'point_step' variable by 'red_factor'.
                    point_step /= red_factor
                else:
                    # Conditional for setting the 'grad_init_point' value to its corresponding float-type value.
                    if type(self.turbstage1.geom.angpitch) in [list, np.ndarray]:
                        grad_init_point = self.turbstage1.geom.angpitch[0]
                    else:
                        grad_init_point = self.turbstage1.geom.angpitch                        
                    # If they have, then calling the 'searchParamMax_uponGradParam()' with 'out=ext'.
                    Phi, Pitch, Psi, Pi, Eff = self.searchParamMax_uponGradParam(oper_point=vx, init_param_max=init_param_max, oper_param='vx', grad_init_point=grad_init_point, grad_point_step=point_step, grad_step_dir=sgn, param=param, out='ext')
                    # Breaking the loop.
                    break

            # Appending the obtained optimal parameters to the auxiliary lists.
            Phis.append(Phi)
            Pitches.append(Pitch)
            Psis.append(Psi)
            Pis.append(Pi)
            Effs.append(Eff)

        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------------RETURN-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#    
        
        # Return statement.
        return Phis, Pitches, Psis, Pis, Effs

    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------twist_optimization() METHOD--------------------------------------------# 
    #--------------------------------------------------------------------------------------------------------------------#
    def twist_optimization(self,
                           vxlist: list=list(range(1, 30)),   
                           init_point: float=0,
                           gamma_step: float=0.05,
                           opt_mode: str='free',
                           hubtip_var: str='linear',
                           fmid: float=0.35,
                           tip_twist: float=0,
                           hub_twist_init: float=0,
                           twist_step: float=0.5,
                           show_progress: bool=True) -> tuple:
        '''Optimizes the local pitch (twist) upon a given turbine so that it maximizes a given output coefficient for a set of input velocities.

        ########################################################################################################################################
        ########################################################################################################################################
        ##############################################################WARNING###################################################################
        ########################################################################################################################################
        ########################################################################################################################################
        The current implementation of the method optimizes the twist distribution, directly, upon the database of polars. It has been noticed that, whenever an angular change occurs during the twist optimization that forces the double interpolation scheme to switch its boundary points, a relatively large discontinuity appears in the twist distribution (relatively large means, within this context, large with respect to the 'gamma_step' parameter being employed in the search, which usually lies << 1; an angular difference on the twist values of successive blade elements of 1º may be taken, hence, as a discontiuity).

        There is no easy way to get rid of this behaviour. Fixing it would require either fitting the polar curves for the interval within which the searching of the optimal twist is being performed (an interval that, whatsoever, is not known in advance), or developing an interpolation scheme based on a configurable (i.e. modifiable) number of points so that, whenever a switch in the boundary points occurs, the abrupt effects induced by such a change are attenuated by re-launching the interpolation scheme with the inclusion of the boundary conditions that have been discarded in the last iteration.

        So far, no actions have been taken for solving this issue, as this version of the code merely aims at obtaining a set of preliminary results for the twist optimization. Further modifications may take place in a near future.
        ########################################################################################################################################
        ########################################################################################################################################
        ########################################################END OF WARNING##################################################################
        ########################################################################################################################################
        ########################################################################################################################################     
        
        **parameters**:
        :param vxlist: input velocities at which to power-optimize the pitch. Default is a list of values ranging from 1 m/s to 30 m/s, with a step between adjacent input velocities of 1 m/s.
        :param init_point: initial point of the gradient-based scheme from which the searching is began. Default is 0.
        :param gamma_step: angular step employed during the optimization. In a local pitch (twist) optimum search, it is usually advised to set it to a relatively low value. Default is 0.05.
        :param opt_mode: string representing the type of twist optimization performed. It must be any of the ['free', 'hubtip', 'linhub'] values. If 'free', the optimization is carried out without any constraints, running over each of the elements of the turbine blades; if 'hubtip', then the optimization is carried out just at the hub and tip elements, and either a linear or a quadratic variation is assumed in between; if 'linhub', then it is assumed a user-defined, fixed twist angle at the tip, and the optimization is carried out by assuming a linear variation between the tip and the hub, the twist angle of the latter being the parameter to optimize. Default is 'free'.
        :param hubtip_var: string parameter representing the twist variation law between tip and hub, in the case the 'opt_mode' param adopts the value 'hubtip' (it is ignored otherwise). It must be either 'linear' or 'quadratic'. Default is 'linear'.
        :param fmid: float representing the factor by which the difference between the hub and tip twist values is multiplied for obtaining the twist value at midspan. It must be a value between 0 and 1. Default is 0.35.
        :param tip_twist: float parameter representing the twist angle at the tip (constraint), in the case the 'opt_mode' param adopts the value 'linhub' (it is ignored otherwise). Default is 0.
        :param hub_twist_init: float parameter representing the initial hub twist employed when searching the optimal linear twist distribution when the optimization is launched in 'opt_mode == linhub' mode. Default is 0.
        :param twist_step: float parameter representing the twisting step to be performed at the hub for the optimal searching of the linear twist distribution when the optimization is launched in 'opt_mode == linhub' mode. Default is 0.5.
        :param show_progress: boolean specifying whether a progress bar is to be displayed while executing the method. Default is 'True'.
        
        **return**:
        :return: a tuple containing five lists: the computed dimensionless flow-coefficients, optimum twist distribution for each input coefficient, optimum dimensionless pressure losses, optimum dimensionless powers and optimum efficiencies.
        
        **rtype:
        :rtype: tuple.
        '''

        #--------------------------------------------------------------------------------------------------------------------#
        #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------# 

        # Asserting that the optimization mode is set to an acceptable value.
        assert opt_mode in ['free', 'hubtip', 'linhub'], "The value of the 'opt_mode' parameter must be any of the ['free', 'hubtip', 'linhub']. Please amend."
        # Conditional for performing assertions upon the input parameters in the case the optimization mode is set to 'hubtip'.
        if opt_mode == 'hubtip':
            # Asserting that the 'hubtip_var' parameter is set to an acceptable value.
            assert hubtip_var in ['linear', 'quadratic'], "The value of the 'hubtip' parameter must be either of the ['linear', 'quadratic']. Please amend."
            # Asserting that the 'fmid' parameter lies between 0 and 1.
            assert 0 < fmid < 1, "The value of 'fmid' must be between 0 and 1. Please amend."
    
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------BODY--------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#    
        
        # Creating auxiliary dimensionless flow-coefficient, pressure-coefficient, power-coefficient and efficiency for end-of-the-pitching-loop storage.
        Phis = list()
        Psis = list()
        Pis = list()
        Effs = list()
        Pitches = list()

        # Initializing the angular pitch to an array of zeroed local pitch (twist) values.
        self.turbstage1.geom.angpitch = np.zeros(len(self.turbstage1.geom.r))

        # Tuned (progress-bar-showing, in case 'show_progress == True') 'for' loop for determining the optimum pitch angle at each operation point.
        for vx in tqdm(vxlist, disable=not show_progress):

            # Instantiating an empty list for storing the twist distribution for the current input operation parameter.
            twist = list()

            # Solving the inlet velocity triangle and retrieving the spanwise relative velocity magnitude.
            w = self.turbstage1.it_triangle_solve(phi_q_v=vx, omega=self.turbstage1.omega, inputparam='vx')[-1]
            # Computing the spanwise Reynolds number for each blade element, assuming the inlet relative velocity magnitude as a referential velocity.
            self.turbstage1.reynolds_calculation(vel=w)

            # Conditional for determining which optimization mode is to be executed.
            if opt_mode in ['free', 'hubtip']:
                
                # In the case the optimization mode is any of the ['free', 'tiphub'], determining which is the specific mode being applied and setting the blade elements upon which to perform the twist optimization, depending on the value adopted by the 'opt_mode' parameter.
                if opt_mode == 'free':
                    # If 'opt_mode == free', then the optimization is carried out upon all the blade elements.
                    gammas = self.turbstage1.it.gamma
                    indices = list(np.arange(0, len(self.turbstage1.geom.r)))
                elif opt_mode == 'hubtip':
                    # Otherwise, if 'opt_mode == tiphub', then the optimziation is carried out, merely, upon the tip and hub blade elements.
                    gammas = [self.turbstage1.it.gamma[0], self.turbstage1.it.gamma[-1]]
                    indices = [0, -1]
                
                # Loop running over the inlet velocity triangle's gamma angle values.
                for e, gamma in enumerate(gammas):
    
                    # Recasting the gamma angle to degrees.
                    gamma *= 180/np.pi
                    # Retrieving the current element's relative inlet angle.
                    beta = self.turbstage1.it.beta[indices[e]]                
    
                    # Interpolating for lift and drag coefficients at the current gamma angle.
                    cl, cd = self.turbstage1.cl_cd_interp(gamma=gamma, blade_index=indices[e])
                    # Computing the induced drag coefficient.
                    cdtc = cd + 0.7*np.abs(cl*self.turbstage1.geom.tip_c/(self.turbstage1.geom.AR*self.turbstage1.geom.pitch))
                    # Computing the tangential coefficient.
                    ctheta0 = cl*np.sin(beta) - cdtc*np.cos(beta)
    
                    # Interpolating for lift and drag coefficients at a positively-displaced, infinitesimally-varied gamma angle.
                    cl, cd = self.turbstage1.cl_cd_interp(gamma=gamma + gamma_step, blade_index=indices[e])
                    # Computing the induced drag coefficient.
                    cdtc = cd + 0.7*np.abs(cl*self.turbstage1.geom.tip_c/(self.turbstage1.geom.AR*self.turbstage1.geom.pitch))
                    # Computing the tangential coefficient.
                    ctheta1 = cl*np.sin(beta) - cdtc*np.cos(beta)
    
                    # Interpolating for lift and drag coefficients at a negatively-displaced, infinitesimally-varied gamma angle.
                    cl, cd = self.turbstage1.cl_cd_interp(gamma=gamma - gamma_step, blade_index=indices[e])
                    # Computing the induced drag coefficient.
                    cdtc = cd + 0.7*np.abs(cl*self.turbstage1.geom.tip_c/(self.turbstage1.geom.AR*self.turbstage1.geom.pitch))
                    # Computing the tangential coefficient.
                    ctheta2 = cl*np.sin(beta) - cdtc*np.cos(beta)
    
                    # Conditional tree for determining the maximum value within the ['ctheta0', 'ctheta1', 'ctheta2'] set.
                    if ctheta1 > ctheta0:
                        # If 'ctheta1' is the maximum value, then setting the 'gamma' attribute to the positively-displaced value and the 'ctheta_max' outcome to 'ctheta1'; additionally, the 'sgn' outcome is set to 1.
                        ctheta_max = ctheta1
                        gamma += gamma_step
                        sgn = 1
                    elif ctheta2 > ctheta0:
                        # If 'ctheta2' is the maximum value, then setting the 'gamma' attribute to the negatively-displaced value and the 'ctheta_max' outcome to 'ctheta2'; additionally, the 'sgn' outcome is set to -1.
                        gamma -= gamma_step
                        ctheta_max = ctheta2
                        sgn = -1
                    else:
                        # If 'ctheta0' is the maximum value, then setting the 'gamma' attribute to the initial gamma value and the 'ctheta_max' outcome to 'ctheta0'; additionally, the 'sgn' outcome is set to 0.                    
                        gamma = gamma
                        ctheta_max = ctheta0
                        sgn = 0
    
                    # Setting the loop-stopping boolean flag 'stop' to 'False' before initializing the 'while' loop.
                    stop = False
    
                    # 'while' loop running over the succesively changing values of the current twist angle to be optimized.
                    while not stop:
                        # Modifying the gamma angle.
                        gamma += sgn*gamma_step
                        # Interpolating for lift and drag coefficients at the directionally-displaced, infinitesimally-varied gamma angle.
                        cl, cd = self.turbstage1.cl_cd_interp(gamma=gamma, blade_index=indices[e])
                        # Computing the induced drag coefficient.
                        cdtc = cd + 0.7*np.abs(cl*self.turbstage1.geom.tip_c/(self.turbstage1.geom.AR*self.turbstage1.geom.pitch))
                        # Computing the tangential coefficient.
                        ctheta = cl*np.sin(beta) - cd*np.cos(beta)
                        # Conditional for determining whether a local maximum has been reached.
                        if ctheta > ctheta_max: 
                            # If it has not, then updating the 'ctheta_max' variable's value to that of the newly computed 'ctheta'.
                            ctheta_max = ctheta
                        else:
                            # If it has, then computing the twist angle by substracting the degree-casted beta angle to the optimal gamma angle.
                            twist.append(gamma - beta*180/np.pi)
                            # Stopping the loop.
                            stop = True
                            # Breaking the loop.
                            break
    
                # Conditional for checking whether the optimization of the twist is to follow a user-defined tip-to-hub variation, and imposing such a law accordingly.
                if opt_mode == 'hubtip':
                    # If the twist variation is to follow a user-defined law, then assigning the hub and tip twists to the 'twisthub' and 'twisttip' variables.
                    twisthub = twist[0]
                    twisttip = twist[-1]
                    # Conditional determining which user-defined law is to be applied.
                    if hubtip_var == 'linear':
                        # If the variation is linear, then directly setting a linear variation between the twist and hub.
                        twist = np.linspace(twisthub, twisttip, len(self.turbstage1.geom.r))
                    elif hubtip_var == 'quadratic':
                        # If the variation is quadratic, then setting the midspan twist value to the factorized average of the hub and tip twist values.
                        twistmid = twisthub + fmid*(twisttip - twisthub)
                        # Getting the hub, mid and tip radii values.
                        rhub = self.turbstage1.geom.r[0]
                        rmid = self.turbstage1.geom.r[len(self.turbstage1.geom.r)//2]
                        rtip = self.turbstage1.geom.r[-1]
                        # Computing the a, b, and c parameters of the quadratic expression providing the twist distribution, namely t=a*r**2 + b*r + c.
                        b = (twistmid - twisthub + ((twisttip - twisthub)*(rhub**2 - rmid**2)/(rtip**2 - rhub**2)))/(rmid - rhub + (rhub**2 - rmid**2)/(rtip + rhub))    
                        a = (twisttip - twisthub)/(rtip**2 - rhub**2) - b/(rtip + rhub)
                        c = twisthub - a*(rhub**2) - b*rhub
                        # Initializing a zero-valued twist array and setting it to the computed quadratic distribution.
                        twist = np.zeros(len(self.turbstage1.geom.r))
                        twist += a*self.turbstage1.geom.r**2 + b*self.turbstage1.geom.r + c

            ####################
            ####################
            # DOCUMENT THIS, + CHECK OUT DEFINITION OF PSI IN TERMS OF TOTAL-TO-TOTAL OR TOTAL-TO-STATIC PRESSURE DROP
            ####################
            ####################
            elif opt_mode == 'linhub':

                # Setting linear twist distribution betwen initial hub twist value and tip twist.
                twist0 = np.linspace(hub_twist_init, tip_twist, len(self.turbstage1.geom.r))
                # Imposing twist distribution across blade.
                self.turbstage1.geom.angpitch = twist0
                # Launching BEM.
                self.BEM(phi_q_vs=[vx], inputparam='vx', reset=True)
                # Retrieving Pi value from turbine and setting it to 'pi0' variable.
                pi0 = self.Pi[0]

                # Setting linear twist distribution betwen infinitesimally, positively-displaced initial hub twist value and tip twist.
                twist1 = np.linspace(hub_twist_init + twist_step, tip_twist, len(self.turbstage1.geom.r))
                # Imposing twist distribution across blade.
                self.turbstage1.geom.angpitch = twist1
                # Launching BEM.
                self.BEM(phi_q_vs=[vx], inputparam='vx', reset=True)
                # Retrieving Pi value from turbine and setting it to 'pi1' variable.
                pi1 = self.Pi[0]

                # Setting linear twist distribution betwen infinitesimally, negatively-displaced initial hub twist value and tip twist.
                twist2 = np.linspace(hub_twist_init - twist_step, tip_twist, len(self.turbstage1.geom.r))
                # Imposing twist distribution across blade.
                self.turbstage1.geom.angpitch = twist2
                # Launching BEM.
                self.BEM(phi_q_vs=[vx], inputparam='vx', reset=True)
                # Retrieving Pi value from turbine and setting it to 'pi2' variable.
                pi2 = self.Pi[0]                

                # Conditional tree for determining the maximum value within the ['pi0', 'pi1', 'pi2'] set.
                if pi1 > pi0:                    
                    # If 'pi1' is the maximum value, then setting the twist distribution to the one spanning between the positively-displaced hub twist value and tip twist, setting the 'pimax' value to 'pi1' and, additionally, setting the 'sgn' outcome 1.
                    self.turbstage1.geom.angpitch = twist1
                    pimax = pi1
                    sgn = 1
                elif pi2 > pi0:
                    # If 'pi2' is the maximum value, then setting the twist distribution to the one spanning between the negatively-displaced hub twist value and tip twist, setting the 'pimax' value to 'pi2' and, additionally, setting the 'sgn' outcome -1.
                    self.turbstage1.geom.angpitch = twist2
                    pimax = pi2                    
                    sgn = -1
                elif pi0 > pi1 and pi0 > pi2:
                    # If 'pi1' is the maximum value, then setting the twist distribution to the one spanning between the initial hub twist value and tip twist, setting the 'pimax' value to 'pi0' and, additionally, setting the 'sgn' outcome 0.
                    self.turbstage1.geom.angpitch = twist0
                    pimax = pi0                    
                    sgn = 0

                # Setting the loop-stopping boolean flag 'stop' to 'False' before initializing the 'while' loop.
                stop = False
                # Setting the dummy index for the hub twist value displacmenet to 2.
                i = 2

                # 'while' loop running over the succesively changing values of the current twist angle to be optimized.
                while not stop:                    
                    # Modifying the linear twist distribution between hub and tip.
                    twist = np.linspace(hub_twist_init + sgn*i*twist_step, tip_twist, len(self.turbstage1.geom.r))
                    # Imposing twist distribution across blade.
                    self.turbstage1.geom.angpitch = twist
                    # Launching BEM.
                    self.BEM(phi_q_vs=[vx], inputparam='vx', reset=True)
                    # Retrieving Pi value from turbine and setting it to 'pi'.
                    pi = self.Pi[0]
                    # Conditional for determining whether the maximum pi value has been reached.
                    if pi > pimax: 
                        # If it has not, then updating the 'pimax' value accordingly and incrementing the dummy index 'i' by 1.
                        pimax = pi
                        i += 1
                    else:
                        # Otherwise, setting 'stop' to 'True' and breaking the loop.
                        stop = True
                        break                
                
            # Setting the turbine's angular pitch distribution to the calculated, optimal twist distribution.
            self.turbstage1.geom.angpitch = twist
            # Executing the BEM method with the optimal twist distribution for the current input operational point.
            self.BEM(phi_q_vs=[vx], inputparam='vx', mode='3DAD', reset=True)
            # Appending the obtained optimal parameters to the auxiliary lists.
            Phis.append(self.Phi[0])
            Pitches.append(twist)
            Psis.append(self.Psi[0])
            Pis.append(self.Pi[0])
            Effs.append(self.Eff[0])

        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------------RETURN-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#    
        
        # Return statement.
        return Phis, Pitches, Psis, Pis, Effs                             
            
##########################################################################################################################################
##########################################################################################################################################
#####################################################SPECIFIC FUNCTIONS/METHODS###########################################################
##########################################################################################################################################
##########################################################################################################################################

#--------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------psi_phi_slope_fitter() METHOD--------------------------------------------# 
#--------------------------------------------------------------------------------------------------------------------#
def psi_phi_slope_fitter(Phi: Union[list, np.ndarray],
                         Psi: Union[list, np.ndarray],
                         interpfit = False) -> tuple:
    '''Linear range slope calculation of the psi-phi functional relation of a Wells turbine.
    
     **parameters**:
    :param Phi: dimensionless flow-parameter variable.
    :param Psi: dimensionless pressure-loss variable.
    :param interfit: boolean determining whether an interpolation-based fitting is to be performed on the input data. Default is False.
    
    **return**:
    :return: average slope and ordinate origin of the fitted least-squared equation.
    
    **rtype**:
    :rtype: tuple.
    '''
    
    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------# 
    
    # Asserting that the input variables have the same length.
    assert all([len(_) == len(Phi) for _ in [Phi, Psi]]), "Please provide input data of same length."
    
    #--------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------BODY--------------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#         
    
    # Interpolation-based fitting, if necessary.
    if interpfit:        
        f = interp1d(Psi, Phi, kind='slinear', fill_value='extrapolate')
        Psi_fit = np.linspace(Psi[0], Psi[-1], 1000)
        Phi_fit = f(Psi_fit)
        Psi = Psi_fit
        Phi = Phi_fit
        
    # Computing discrete Psi-step.
    dPsi = (Psi[-1] - Psi[0])/len(Psi)
    # Instantiating a 'slopes' list that will store the different slope values.
    slopes = list()
    # Loop running over the 'Psifit' variable.
    for e, i in enumerate(Psi[:-3]):
        # Computing the least-square-based slope of the fitted Psi-Phi functional relation with a number of progressively less points, and adding such slope to 'slopes'.
        slopes.append(mt.functionpp.leastSquares(np.array(Psi[:-(e+1)]), np.array(Phi[:-(e+1)]))[0])
    # Reverting the 'slopes' list.
    slopes = slopes[::-1]
    # Getting the threshold value at which the dPsi-scaled slopes overcome the unit value, which means that the constancy condition is violated. Such a threshold marks the point at which the fitted Psi-Phi functional relation ceases to be linear.
    threshold_array = [_ > 1 for _ in np.abs(np.diff(slopes))/dPsi]
    if True in threshold_array:
        threshold = len(Psi) - [_ > 1 for _ in np.abs(np.diff(slopes))/dPsi].index(True)
    else:
        threshold = len(Psi) - 1
    # Instantiating 's0s' and 'y0s' variables that will store the slopes and origin-ordinates upon the threshold range.
    s0s = list()
    y0s = list()
    # Loop running over the reversed threshold range.
    for e in range(threshold, 3, -1):
        # Computing the least-square-based slope and origin-ordinate with a number of progressively lower points.
        s0, y0 = mt.functionpp.leastSquares(np.array(Psi[:e]), np.array(Phi[:e]))[:2]
        # Appending the computed slopes and origin-ordinates to the corresponding lists.
        s0s.append(s0)
        y0s.append(y0)
    # Computing the average slopes and origin-ordinates.
    s0ave = np.average(s0s)
    y0ave = np.average(y0s)
    
    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------RETURN-------------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#     
    
    # Return statement.
    return s0ave, y0ave

#--------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------stochastic_analysis() METHOD--------------------------------------------# 
#--------------------------------------------------------------------------------------------------------------------#
def stochastic_analysis(Phi: Union[list, np.ndarray],
                        Psi: Union[list, np.ndarray],
                        Pi: Union[list, np.ndarray],
                        mindev: float=1e-2, maxdev: float=0.2, n: int=30, try_trim=True, devobj_par: str='pressure') -> tuple:
    '''Gaussian-based stochastic analysis upon dimensionless turbine variables.
    
    **parameters**:
    :param Phi: dimensionless flow-parameter variable.
    :param Psi: dimensionless pressure-loss variable.
    :param Pi: dimensionless power variable.
    :param maxdev: float number representing the maximum, normalized standard deviation by which to perform the stochastic analysis. Default is 0.2.
    :param mindev: float number representing the minimum, normalized standard deviation by which to perform the stochastic analysis. Default is 1e-2.
    :param n: integer representing the number of normalized standard deviations to employ for performing the stochastic analysis. Default is 30.
    :param try_trim: boolean for stating whether to try trimming the Psi-Pi functional. Default is True.
    :param devobj_par: string specifying the units in which the standard deviations are provided. Either 'pressure' (for standard deviations on pressure) or 'flow' (standard deviations on flow). Default is 'pressure'.
    
    **return**:
    :return: a tuple containing four arrays: the array of maximum deviations, and the arrays of stochastic dimensionless flow-rates, powers and efficiencies, respectively.
    
    **rtype**:
    :rtype: tuple.
    '''

    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------# 
    
    # Asserting that the input variables have the same length.
    assert all([len(_) == len(Phi) for _ in [Psi, Pi]]), "Please provide input data of same length."

    #--------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------BODY--------------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#             
    
    # Knot-trimming of the psi-pi functional relation, only if 'try_trim' is 'True'.
    if try_trim:
        # Calling knot_trimmer() function upon Psi-Pi functional relation for trimming any possible knot.
        trim_psipi, trim_discr_psipi, discr_inds = mt.functionpp.knotTrimmer(Psi, Pi, retdiscr=True)
    else:
        # Setting 'trim_psipi' and 'trim_discr_psipi' to tuple corresponding to input (Psi, Pi) variables.
        trim_psipi = (Psi, Pi)
        trim_discr_psipi = (Psi, Pi)
    
    # Instantiating discrete Psi variable.
    Psidiscr = np.array(trim_discr_psipi[0])    
    # Instantiating fitted (continuous) Psi variable.
    Psifit = np.array(trim_psipi[0])
    # Computing discrete Psi-step.
    dPsi = (Psifit[-1] - Psifit[0])/len(Psifit)    
    # Instantiating discrete Pi variable.
    Pidiscr = np.array(trim_discr_psipi[1])
    # Instantiating fitted (continuous) Pi variable.
    Pifit = np.array(trim_psipi[1])
    # Instantiating an empty list for storing the discrete Phi variable.
    Phidiscr = list()
    # Conditional for checking whether a trimming operation is necessary on the Phi variable.
    if try_trim:
        if len(discr_inds[0]) > 0:
            # Loop for trimming the Phi variable according to the already trimmed Pi variable.
            for e, discr_ind in enumerate(discr_inds):
                Phidiscr += Phi[discr_ind[0]:discr_ind[1]]
                # In case the trimming operation does not correspond to the last chunk, adding the average coordinates at both sides of the chunk to the Phi variable.
                if e != len(discr_inds) - 1:
                    Phidiscr += [np.average([Phi[discr_inds[e][1]], Phi[discr_inds[e + 1][0]]])]
    else:
        Phidiscr = Phi
    # Interpolating the Psi-Phi functional for fitting the Phi variable.
    f = interp1d(Psidiscr, Phidiscr, kind='slinear', fill_value='extrapolate')
    # Instantiating the fitted (continuous) Phi variable.
    Phifit = f(Psifit)
    # Computing discrete Phi-step
    dPhi = (Phifit[-1] - Phifit[0])/len(Phifit)
    
    # Conditional for discriminating between stochastic processes based on pressure- or flow-based deviations; 'pressure' case.
    if devobj_par == 'pressure':
    
        # Least-square fitting and computation of average slope and ordinate-origin of the phi-psi functional relation.
        s0ave, y0ave = psi_phi_slope_fitter(Phifit, Psifit)    

        # Instantiating the 'devs' variable, an evenly-spaced array of standard deviations.
        devs = np.linspace(mindev, maxdev, n)
        # Computing the maximum gaussian value based on the maximum standard deviation and the last element of 'Psifit'.
        maxgauss = np.exp(-Psifit[-1]**2/(2*maxdev**2))
        # Instantiating the 'Psimingaussval' variable and setting its value to the last element of 'Psifit'.
        Psimingaussval = Psifit[-1]
        # Instantiating the 'mingaussval' variable and setting its value to the outcome of the Gaussian function with the maximum deviation value and 'Psimingaussval'.
        mingaussval = np.exp(-Psimingaussval**2/(2*maxdev**2))
        # Instantiating the 'iPsi' variable.
        iPsi = 0
        # While loop for extending the dimensionless variables until the outcome of the gaussian function for the largest maximum deviation lies below 1e-4.
        while mingaussval > 1e-4:
            # Incrementing 'Psimingaussval' by 'dPsi'.
            Psimingaussval += dPsi
            # Incrementing 'iPsi' by 1.
            iPsi += 1
            # Computing 'mingaussval'.
            mingaussval = np.exp(-Psimingaussval**2/(2*maxdev**2))

        # Instantiating 'Psifit_stoch_add' as the portion of curve required for extending the 'Psifit' variable until getting to the Gaussian limit.
        Psifit_stoch_add = [_ for _ in np.linspace(Psifit[-1], Psimingaussval, iPsi)]
        # Instantiating 'Psifit_stoch' as an extent of the 'Psifit' variable.
        Psifit_stoch = np.array([_ for _ in Psifit] + Psifit_stoch_add)
        # Instantiating 'Phifit_stoch' as the linear extent based on 'Psifit_stoch'.
        Phifit_stoch = s0ave*Psifit_stoch + y0ave
        # Instantiating 'Pifit_stoch' by setting the values of the extent portion to a null value.
        Pifit_stoch = np.array([_ for _ in Pifit] + [0 for _ in Psifit_stoch_add])
        # Computing the available dimensionless power as the product of the extended and fitted Psi and Phi variables, and instantiating the 'PiAvailfit_stoch' variable.
        PiAvailfit_stoch = Psifit_stoch*Phifit_stoch       
        
        # Instantiating the 'avePhi' variable for storing the average value of the stochastic Phi variable.
        avePhi = list()
        # Instantiating the 'avePi' variable for storing the average value of the stochastic Pi variable.
        avePi = list()
        # Instantiating the 'aveEff' variable for storing the average value of the stochastic Eff variable.
        aveEff = list()
        # Loop running over the standard deviations.
        for dev in devs:
            # Computing the average value of the stochastic Phi variable.
            avePhi.append((1/(np.sqrt(2*np.pi)*dev))*simpson(np.exp(-Psifit_stoch**2/(2*dev**2))*Phifit_stoch, dx=dPsi))
            # Computing the average value of the stochastic Pi variable.
            avePi.append((1/(np.sqrt(2*np.pi)*dev))*simpson(np.exp(-Psifit_stoch**2/(2*dev**2))*Pifit_stoch, dx=dPsi))
            # Computing the average value of the stochastic PiAvail variable.
            avePiAvail = (1/(np.sqrt(2*np.pi)*dev))*simpson(np.exp(-Psifit_stoch**2/(2*dev**2))*PiAvailfit_stoch, dx=dPsi)
            # Computing the average value of the stochastic Eff variable.
            aveeff = avePi[-1]/avePiAvail
            if 0 < aveeff < 1:
                aveEff.append(avePi[-1]/avePiAvail)
            else:
                aveEff.append(0)
                
        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------------RETURN-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#                

        # Return statement.
        return devs, avePhi, avePi, aveEff                
                
    # 'flow' case.            
    elif devobj_par == 'flow':
        
        # Least-square fitting and computation of average slope and ordinate-origin of the psi-phi functional relation.
        s0ave, y0ave = psi_phi_slope_fitter(Psifit, Phifit)
        
        # Instantiating the 'devs' variable, an evenly-spaced array of standard deviations.
        devs = np.linspace(mindev, maxdev, n)
        # Computing the maximum gaussian value based on the maximum standard deviation and the last element of 'Phifit'.
        maxgauss = np.exp(-Phifit[-1]**2/(2*maxdev**2))
        # Instantiating the 'Phimingaussval' variable and setting its value to the last element of 'Phifit'.
        Phimingaussval = Phifit[-1]
        # Instantiating the 'mingaussval' variable and setting its value to the outcome of the Gaussian function with the maximum deviation value and 'Phimingaussval'.
        mingaussval = np.exp(-Phimingaussval**2/(2*maxdev**2))
        # Instantiating the 'iPhsi' variable.
        iPhi = 0
        # While loop for extending the dimensionless variables until the outcome of the gaussian function for the largest maximum deviation lies below 1e-4.
        while mingaussval > 1e-4:
            # Incrementing 'Psimingaussval' by 'dPhi'.
            Phimingaussval += dPhi
            # Incrementing 'iPhi' by 1.
            iPhi += 1
            # Computing 'mingaussval'.
            mingaussval = np.exp(-Phimingaussval**2/(2*maxdev**2))

        # Instantiating 'Phifit_stoch_add' as the portion of curve required for extending the 'Phifit' variable until getting to the Gaussian limit.
        Phifit_stoch_add = [_ for _ in np.linspace(Phifit[-1], Phimingaussval, iPhi)]
        # Instantiating 'Phifit_stoch' as an extent of the 'Phifit' variable.
        Phifit_stoch = np.array([_ for _ in Phifit] + Phifit_stoch_add)
        # Instantiating 'Psifit_stoch' as the linear extent based on 'Phifit_stoch'.
        Psifit_stoch = s0ave*Phifit_stoch + y0ave
        # Instantiating 'Pifit_stoch' by setting the values of the extent portion to a null value.
        Pifit_stoch = np.array([_ for _ in Pifit] + [0 for _ in Phifit_stoch_add])
        # Computing the available dimensionless power as the product of the extended and fitted Phi and Psi variables, and instantiating the 'PiAvailfit_stoch' variable.
        PiAvailfit_stoch = Phifit_stoch*Psifit_stoch

        # Instantiating the 'avePsi' variable for storing the average value of the stochastic Psi variable.
        avePsi = list()
        # Instantiating the 'avePi' variable for storing the average value of the stochastic Pi variable.
        avePi = list()
        # Instantiating the 'aveEff' variable for storing the average value of the stochastic Eff variable.
        aveEff = list()
        # Loop running over the standard deviations.
        for dev in devs:
            # Computing the average value of the stochastic Psi variable.
            avePsi.append((1/(np.sqrt(2*np.pi)*dev))*simpson(np.exp(-Phifit_stoch**2/(2*dev**2))*Psifit_stoch, dx=dPhi))
            # Computing the average value of the stochastic Pi variable.
            avePi.append((1/(np.sqrt(2*np.pi)*dev))*simpson(np.exp(-Phifit_stoch**2/(2*dev**2))*Pifit_stoch, dx=dPhi))
            # Computing the average value of the stochastic PiAvail variable.
            avePiAvail = (1/(np.sqrt(2*np.pi)*dev))*simpson(np.exp(-Phifit_stoch**2/(2*dev**2))*PiAvailfit_stoch, dx=dPhi)
            # Computing the average value of the stochastic Eff variable.
            aveeff = avePi[-1]/avePiAvail
            if 0 < aveeff < 1:
                aveEff.append(avePi[-1]/avePiAvail)
            else:
                aveEff.append(0)

        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------------RETURN-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#                

        # Return statement.
        return devs, avePsi, avePi, aveEff

#--------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------linear_airfoil_dist() METHOD--------------------------------------------# 
#--------------------------------------------------------------------------------------------------------------------#
def linear_airfoil_dist(airfoil_hub: geCnt.PROFILES, airfoil_tip: geCnt.PROFILES, N: int) -> list:
    '''It provides a linear airfoil distribution between the hub and tip blade elements.
    
    **parameters**:
    :param airfoil_hub: the airfoil profile at hub.
    :param airfoil_tip: the airfoil profile at tip.
    :param N: the number of blade elements along the span.
    
    **return**:
    :return: a list containing a linearly varying airfoil distribution along the span, if possible.
    
    **rtype**:
    :rtype: list      
    '''

    #--------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------BODY--------------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#    

    if all(["NACA" in _.name for _ in [airfoil_hub, airfoil_tip]]):
        if airfoil_hub.name[4:6] == airfoil_tip.name[4:6]:
            common_char = airfoil_hub.name[4:6]
            all_airfoils = [_.name for _ in geCnt.PROFILES if _.name[4:6] == common_char]
            stop = False
            i = 0
            while not stop:
                char = airfoil_hub.name[6 + i]
                if char not in [str(j) for j in range(0, 10)]:
                    i += 1
                else:
                    stop = True
                    break
            idx = 6 + i
            airfoil_hub_end = int(airfoil_hub.name[idx:])
            airfoil_tip_end = int(airfoil_tip.name[idx:])
            if airfoil_hub_end > airfoil_tip_end:
                airfoils = [_ for _ in all_airfoils if int(_[idx:]) >= airfoil_tip_end and int(_[idx:]) <= airfoil_hub_end][::-1]
            elif airfoil_hub_end < airfoil_tip_end:
                airfoils = [_ for _ in all_airfoils if int(_[idx:]) >= airfoil_hub_end and int(_[idx:]) <= airfoil_tip_end]
            else:
                airfoils = [_ for _ in all_airfoils if int(_[idx:]) == airfoil_hub_end]

        same_airfoil_num = N // len(airfoils)
        rest = N % len(airfoils)
        linear_dist = []
        for airfoil in airfoils[1:-1]:
            linear_dist.append(same_airfoil_num*[airfoil])
        if rest == 0:
            linear_dist.insert(0, same_airfoil_num*[airfoils[0]])
            linear_dist.append(same_airfoil_num*[airfoils[-1]])             
        elif rest == 1:
            linear_dist.insert(0, (same_airfoil_num + 1)*[airfoils[0]])
            linear_dist.append(same_airfoil_num*[airfoils[-1]])
        elif rest % 2 == 0:
            linear_dist.insert(0, (same_airfoil_num + 1)*[airfoils[0]])
            linear_dist.append((same_airfoil_num + 1)*[airfoils[-1]])
        elif rest % 2 == 1:
            even = rest // 2
            odd = rest - even
            linear_dist.insert(0, (same_airfoil_num + odd)*[airfoils[0]])
            linear_dist.append((same_airfoil_num + even)*[airfoils[-1]])
        linear_dist = [item for sublist in linear_dist for item in sublist]
    else:
        even = N // 2
        odd = N - even
        linear_dist = odd*[airfoils[0]] + even*[airfoils[1]]

    for idx_airfoil, airfoil in enumerate(linear_dist):
        linear_dist[idx_airfoil] = getattr(geCnt.PROFILES, airfoil)                

    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------RETURN-------------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    
    # Return statement.
    return linear_dist

#--------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------custom_airfoil_dist() METHOD--------------------------------------------# 
#--------------------------------------------------------------------------------------------------------------------#
def custom_airfoil_dist(airfoils: Union[list, np.ndarray], N: int) -> list:
    '''It provides a linear airfoil distribution between the hub and tip blade elements.
    
    **parameters**:
    :param airfoils: list or array containing, in a hub-to-tip direction order, the different geometries to be considered in the spanwise direction.
    :param N: the number of blade elements along the span.
    
    **return**:
    :return: a list containing a linearly varying airfoil distribution along the span.
    
    **rtype**:
    :rtype: list      
    '''

    #--------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------BODY--------------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#     

    same_airfoil_num = N // len(airfoils)
    rest = N % len(airfoils)
    linear_dist = []
    for airfoil in airfoils[1:-1]:
        linear_dist.append(same_airfoil_num*[airfoil])
    if rest == 0:
        linear_dist.insert(0, same_airfoil_num*[airfoils[0]])
        linear_dist.append(same_airfoil_num*[airfoils[-1]])             
    elif rest == 1:
        linear_dist.insert(0, (same_airfoil_num + 1)*[airfoils[0]])
        linear_dist.append(same_airfoil_num*[airfoils[-1]])
    elif rest % 2 == 0:
        linear_dist.insert(0, (same_airfoil_num + 1)*[airfoils[0]])
        linear_dist.append((same_airfoil_num + 1)*[airfoils[-1]])
    elif rest % 2 == 1:
        even = rest // 2
        odd = rest - even
        linear_dist.insert(0, (same_airfoil_num + odd)*[airfoils[0]])
        linear_dist.append((same_airfoil_num + even)*[airfoils[-1]])
    linear_dist = [item for sublist in linear_dist for item in sublist]

    for idx_airfoil, airfoil in enumerate(linear_dist):
        linear_dist[idx_airfoil] = getattr(geCnt.PROFILES, airfoil)            
                
    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------RETURN-------------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    
    # Return statement.
    return linear_dist

##########################################################################################################################################
##########################################################################################################################################
#######################################################PYGAD-RELATED FUNCTIONS############################################################
##########################################################################################################################################
##########################################################################################################################################

#--------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------instantiate_turbine() METHOD--------------------------------------------# 
#--------------------------------------------------------------------------------------------------------------------#
def instantiate_turbine(constargs: dict, varargs: list, solution: list) -> turb:
    '''It instantiates a turbine instance from the input geometrical parameters.
    
    **parameters**:
    :param constargs: constant parameters, meaning those that are not supposed to vary during a GA run. They hold specified values.
    :param varargs: variable parameters, meaning those that are supposed to vary during a GA run. They are just variable names, and they retrieve their values from the "solution" variable.
    :param solution: a list of values that represent, on a GA run, the genotype chain. They specify the values to be adopted by the variable parameters contained in the "varargs" variable.
    
    **return**:
    :returs: turbine instance.
    
    **rtype**:
    :rtype: turbine.
    '''
   
    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#

    # Setting all geometrical variables to None. It serves the purpose of asserting their values further on.
    ttype = None
    Z = None
    sigmahub = None
    sigmatip = None
    hub_to_tip_ratio = None
    airfoil_dist = None
    tip_percent = None

    # Asserting that 'varargs' and 'solution' have the same length.
    assert len(solution) == len(varargs), "Genotype chain and variable parameters must have the same length."    
        
    # Setting values of constant arguments other than the geometrical ones to None.
    N = None
    omega = None
    rcas = None
    airfoils = None
    mode = None
    # Running over the 'constargs' keys and setting the corresponding values to local variable instances.
    for key in constargs.keys():
        if key == "ttype":
            ttype = constargs[key]
        elif key == "N":
            N = constargs[key]
        elif key == "omega":
            omega = constargs[key]
        elif key == "rcas":
            rcas = constargs[key]
        elif key == "Mode":
            mode = constargs[key]
            assert mode in ["mono", "mono_wNmax", "double", "double_wNmax", "counter", "counter_wNmax", "pitchopt", "pitchopt_wNmax", "twistopt", "twistopt_wNmax"], "Provide a valid mode ('mono', 'mono_wNmax', 'double', 'double_wNmax', 'counter', 'counter_wNmax', 'pitchopt', 'pitchopt_wNmax', 'twistopt', 'twistopt_wNmax'). Current value is {mode}".format(mode=mode)
        elif key == "Z":
            Z = constargs[key]
        elif key == "sigma_hub":
            sigmahub = constargs[key]
        elif key == "sigma_tip":
            sigmatip = constargs[key]
        elif key == "hub_to_tip_ratio":
            hub_to_tip_ratio = constargs[key]
        elif key == "airfoil_dist":
            airfoil_dist = constargs[key]
        elif key == "tip_percent":
            tip_percent = constargs[key]

    # Setting values of variable arguments.    
    if "Z" in varargs:
        ind = varargs.index("Z")
        Z = solution[ind]
    if "sigma_hub" in varargs:
        ind = varargs.index("sigma_hub")
        sigmahub = solution[ind]
    if "sigma_tip" in varargs:
        ind = varargs.index("sigma_tip")
        sigmatip = solution[ind]
    if "hub_to_tip_ratio" in varargs:
        ind = varargs.index("hub_to_tip_ratio")
        hub_to_tip_ratio = solution[ind]
    if "airfoil_dist" in varargs:
        ind = varargs.index("airfoil_dist")
        airfoil_dist = solution[ind]
    if "tip_percent" in varargs:
        ind = varargs.index("tip_percent")
        tip_percent = solution[ind]

    # For loop asserting that all geometrical arguments have been assigned a value.
    for _ in ["ttype", "Z", "sigmahub", "sigmatip", "hub_to_tip_ratio", "airfoil_dist", "tip_percent"]:
        assert locals()[_] != None, _ + " has not been specified, neither as a constant parameter nor as a gene value to be optimised."

        
    #--------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------BODY--------------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#            
    
    # Instantiating monoplane Wells turbine with retrieved phenotype values.
    turb_ = turb()   
    turb_.create_or_add_stage(ttype=ttype,
                              omega=omega,
                              N=N,
                              Z=Z,
                              rcas=rcas,
                              hub_to_tip_ratio=hub_to_tip_ratio,
                              tip_percent=tip_percent)
    
    # Instantiating doubleplane Wells turbine in case 'Mode' is either of 'double', 'double_wNmax'.
    if 'double' in mode:
        turb_.create_or_add_stage(omega=omega,
                                  N=N,
                                  Z=Z,
                                  rcas=rcas,
                                  hub_to_tip_ratio=hub_to_tip_ratio,
                                  tip_percent=tip_percent)
    
    # Instantiating counterplane Wells turbine in case 'Mode' is either of 'counter', 'counter_wNmax'.
    elif 'counter' in mode:
        turb_.create_or_add_stage(omega=-omega,
                                  N=N,
                                  Z=Z,
                                  rcas=rcas,
                                  hub_to_tip_ratio=hub_to_tip_ratio,
                                  tip_percent=tip_percent)
    
    # Retrieving radial array for solidity-based radial chord distribution calculation.
    r = turb_.turbstage1.geom.r
    # Retrieving tip radial coordinate.
    rtip = r[-1]
    # Computing the parameters for the linear chord distribution along the radius (c0 is the ordinate, m is the slope).
    c0 = 2*np.pi*hub_to_tip_ratio*rtip*(sigmatip - sigmahub)/(Z*(hub_to_tip_ratio - 1))
    m = 2*np.pi*(hub_to_tip_ratio*sigmahub - sigmatip)/(Z*(hub_to_tip_ratio - 1))
    # Computing the linear chord distribution along the radius.
    chord = c0 + m*r
    # Setting chordal distribution.
    turb_.turbstage1.geom.chord = chord
    # Setting tip percent.
    turb_.turbstage1.geom.tip_percent = tip_percent

    if len(airfoil_dist) == 2:
        # Calling linear_airfoil_dist() function with calculated hub and tip geometries.
        turb_.turbstage1.geom.airfoil = linear_airfoil_dist(airfoil_dist[0], airfoil_dist[1], len(turb_.turbstage1.geom.r))
    elif len(airfoil_dist) > 2:
        turb_.turbstage1.geom.airfoil = custom_airfoil_dist(airfoil_dist, len(turb_.turbstage1.geom.r))
    # Extending span-wise distributions to second stage, if necessary.
    if mode in ["double", "double_wNmax", "counter", "counter_wNmax"]:
        turb_.turbstage2.geom.chord = chord
        turb_.turbstage2.geom.tip_percent = tip_percent
        turb_.turbstage2.geom.airfoil = turb_.turbstage1.geom.airfoil
        
    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------RETURN-------------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#        
    
    # Return statement.
    return turb_

#--------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------single_fitness_func() METHOD-------------------------------------------# 
#--------------------------------------------------------------------------------------------------------------------#
def single_fitness_func(ga_instance=None, solution=None, solution_idx=None, **kwargs) -> tuple:
    
    '''It applies a weightedless fitness function (no-site-variant, but site-specific).
    
    **parameters**:
    :param ga_instance: instance of a pygad genetic-algorithm object. Default is None.
    :param solution: a list of values that represent, on a GA run, the genotype chain. They specify the values to be adopted by the variable parameters contained in the "varargs" variable.
    :param solution_idx: index of the genotype chain within the current population.
    :param constargs (kwarg): constant parameters, meaning those that are not supposed to vary during a GA run. They hold specified values.
    :param varargs (kwarg): variable parameters, meaning thoste that are supposed to vary during a GA run. They hold genotype values contained in the "solution" variable.
    :param mode_fitness (kwarg): string specifying how the GA instance is run. Either "GA" (for running an instance of the GA algorithm) or "NoGA" for calling the function as a plain subroutine.
    :param out (kwarg): string specifying the type of output. Either 'fitness' (for getting a standard fitness output) or 'stfitness' (for getting stochastic-related fitness output).
    :param turb (kwarg): instance of a 'turbine' class; required in the case the "mode_fitness" variable is "NoGA".
    :param devobj_par (kwarg): string specifying the units in which the standard deviations are provided. Either 'pressure' (for standard deviations on pressure) or 'flow' (standard deviations on flow).
    :param slope_calc (kwarg): string specifying the parameters whereby the phi-psi slope is calculated. Either 'wOptPitch' (the 'psi' values employed are those that correspond to the optimum pitch cases) or 'wStandard' (the standard 'psi' values are employed).
    :param fit_param (kwarg): string specifying the parameter that is optimized in the fitness function. Any of the 'eff' (stochastic efficiency), 'pi' (stochastic dimensionless power) or 'pow' (stochastic dimensional power).
    
    **return**:
    :return: either a tuple containing fitness information, or a tuple containing stochastic-related fitness information.
    
    **rtype**:
    :rtype: tuple.
    '''
    
    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------# 
    
    # Getting kwargs keys.
    kwkeys = kwargs.keys()
    # Asserting that 'constargs' is present, and assigning it to 'constargs' variable.
    assert 'constargs' in kwkeys, "'constargs' variable not provided. It must be passed as a kwarg."
    constargs = kwargs['constargs']
    # Asserting that 'varargs' is present, and assigning it to 'varargs' variable.
    assert 'varargs' in kwkeys, "'varargs' variable not provided. It must be passed as a kwarg."
    varargs = kwargs['varargs']
    # Asserting that 'mode_fitness' is present, and assigning it to 'mode_fitness' variable.
    assert 'mode_fitness' in kwkeys, "'mode_fitness' variable not provided. It must be passed as a kwarg."
    mode_fitness = kwargs['mode_fitness']
    if mode_fitness == 'GA':
        # Asserting that 'solution' and 'solution_idx' are not None (have valid values)."
        assert not (solution is None), "Provide a valid 'solution' parameter."
        assert not (solution_idx is None), "Provide a valid 'solution_idx' parameter."
    else:        
        # Asserting that 'turb' is present, and assigning it to 'turb' variable.
        assert 'turb' in kwkeys, "'turb' variable not provided. It must be passed as a kwarg."
        turb = kwargs['turb']
        # Assertig that 'turb' is not None (has a valid value).
        assert not (turb is None), "Provide a valid 'turb' parameter."    
    # Asserting that 'out' is present, and assigning it to 'out' variable.
    assert 'out' in kwkeys, "'out' variable not provided. It must be passed as a kwarg."
    out = kwargs['out']
    # Asserting that 'out_params' is present, and assigning it to 'out_params' variable.
    assert 'out_params' in kwkeys, "'out_prams' variable not provided. It must be passed as a kwarg."
    out_params = kwargs['out_params']  
    
    #--------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------BODY--------------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#      
    
    # Setting constargs.
    cpobj = None
    devobj = None
    Nmin = None
    Nmax = None
    mode = None
    param = 'Pi'
    opt_mode = 'free'
    hubtip_var = 'linear'
    devobj_par = 'pressure'
    slope_calc = 'wStandard'
    fit_param = 'eff'
    
    # Loop running over the passed keyword arguments and setting their values to the corresponding variables, if necessary.
    args = ["cpobjs", "devobjs", "weights", "Nmin", "Nmax", "Mode"]
    for key in constargs.keys():
        if key == "cpobjs":
            cpobjs = constargs[key]
        elif key == "devobjs":
            devobjs = constargs[key]
        elif key == "weights":
            weights = constargs[key]
        elif key == "Nmin":
            Nmin = constargs[key]
        elif key == "Nmax":
            Nmax = constargs[key]
        elif key == "Mode":
            mode = constargs[key]
        elif key == "param":
            param = constargs[key]            
        elif key == "opt_mode":
            opt_mode = constargs[key]
        elif key == "hubtip_var":
            hubtip_var = constargs[key]
        elif key == "devobj_par":
            devobj_par = constargs[key]
        elif key == "slope_calc":
            slope_calc = constargs[key] 
        elif key == "fit_param":
            fit_param = constargs[key]
            
    # Getting corresponding cpto table according to the input parameter 'devobj_par'.
    cpto = cpto_pstd if devobj_par == 'pressure' else cpto_qstd            

    # Instantiating turbine and checking for already computed genes.
    if mode_fitness == 'GA':
        turb = instantiate_turbine(constargs, varargs, solution)
        # Converting solution to tuple-form for serving as dictionary entry.
        tup = tuple(solution)
        if tup in ga_instance.processed_genes.keys():
            # If so, returning fitness value.
            return tuple(ga_instance.processed_genes[tup])

    # The 'Phitot', 'Psitot' and 'Pitot' variables are for storing the dimensionless flow-parameter, pressure and power resulting from the succesive application of the BEM method.
    Phitot = list()
    Psitot = list()
    Pitot = list()
    # The 'vxlist' variable stores the input velocities to be BEM-computed.
    vxlist = list(np.arange(1, 46))

        #--------------------------------------------------------------------------------------------------------------------#
        #-----------------------------------------------------COMPUTING BEMS-------------------------------------------------#
        #--------------------------------------------------------------------------------------------------------------------#    
    
    # Conditional for when the algorithm is run in 'mono' mode (monoplane Wells turbine).
    if 'mono' in mode:
        # Applying BEM method to current phenotype-turbine when running in 'mono' mode.
        turb.BEM(vxlist, inputparam='vx', mode='3DAD')
        # Storing dimensionless variables in previously defined parameters.
        Phitot = turb.Phi
        Psitot = turb.Psi
        Pitot = turb.Pi

    # Conditional for when the algorithm is run in 'counter' mode (counterrotating Wells turbine).
    elif 'double' in mode or 'counter' in mode:

        # Creating auxiliary variables of the dimensionless flow-parameter and pressure for each turbine stage.
        Phi1 = list()
        Psi1 = list()
        Phi2 = list()
        Psi2 = list()
        
        # For loop over the input velocities to which the BEM method is applied succesively.
        for vx in vxlist:
            turb.BEM([vx], inputparam='vx', mode='3DAD')
            # Storing each turbine stage's dimensionless parameters in the auxiliary variables.
            Phi1.append(turb.turbstage1.it.Phi)
            Psi1.append(turb.turbstage1.eout.Psi)
            Phi2.append(turb.turbstage2.it.Phi)
            Psi2.append(turb.turbstage2.eout.Psi)
            # Storing global (turbine-wise) dimensionless variables in previously defined parameters.
            Phitot.append(turb.Phi[0])
            Psitot.append(turb.Psi[0])
            Pitot.append(turb.Pi[0])

    # Conditional for when the algorithm is run in 'pitchopt', 'pitchopt_wNmax', 'twistopt' or 'twistop_wNmax' mode.
    elif mode in ["pitchopt", "pitchopt_wNmax", "twistopt", "twistopt_wNmax"]:
        # Conditional for determining which optimization function is to be called.
        if "pitchopt" in mode:
            # Calling the 'pitch_optimization()' function.
            Phitot, optangpitchespow, optpsispow, optpispow, opteffspow = turb.pitch_optimization(vxlist=vxlist, param=param, show_progress=False)
        elif "twistopt" in mode:
            # Calling the 'twist_optimization()' function.
            Phitot, optangpitchespow, optpsispow, optpispow, opteffspow = turb.twist_optimization(vxlist=vxlist, opt_mode=opt_mode, hubtip_var=hubtip_var, show_progress=False)
    # ABCDE.
            
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------COMPUTING THE ROTATIONAL SPEEDS-----------------------------------------#     
        #--------------------------------------------------------------------------------------------------------------------#         
        
    # Conditional for 'mono' case. The rotational speed is computed directly.
    if 'mono' in mode:
        # Least-square fitting and computation of average slope and ordinate-origin of the phi-psi functional relation.
        s0ave, y0ave = psi_phi_slope_fitter(Phi=turb.Phi, Psi=turb.Psi, interpfit=True)
        # K instantiation.
        K = s0ave
        # Computation of the rotational speeds for each cpobj.
        Ns = [(2*K*cpobj*turb.turbstage1.geom.rtip/turb.turbstage1.flow.rho)*60/(2*np.pi) for cpobj in cpobjs]
        
        # Conditional for checking whether the fitness function is to be considered with the Nmax modification.
        if 'wNmax' in mode:            
            # Loop running over the calculated rotational speeds for checking them and, in case of overcoming the upper limit, setting their value to Nmax and recalculating the standard deviation value of the pressure fluctuations by look-up table interpolation.
            for e, N in enumerate(Ns):
                # Checker conditional.
                if N > Nmax:
                    # Modifying Ns[e] value.
                    Ns[e] = Nmax                
                    # Modifying cpobjs[e] value.
                    cpobjs[e] = Nmax*(2*np.pi)/((2*K*turb.turbstage1.geom.rtip/turb.turbstage1.flow.rho)*60)                    
                    # Getting maximum and minimum indices for interpolating in the cpto-pstd look-up table.
                    indexmin = [cpobjs[e] - _ < 0 for _ in cpto['cpto']].index(True) - 1
                    indexmax = [_ - cpobjs[e] > 0 for _ in cpto['cpto']].index(True)
                    # Getting the maximum and minimum cpobjs (x-axis) for interpolation.
                    cpobjmin = cpto['cpto'][indexmin]
                    cpobjmax = cpto['cpto'][indexmax]
                    # Getting the maximum and minimum pstds (y-axis) for interpolation.
                    pstdmin = cpto[e + 1][indexmin]
                    pstdmax = cpto[e + 1][indexmax]
                    # Calling 'double_interp' function for recalculating the standard deviation of pressure and modifying devobjs[e] value.
                    devobjs[e] = mt.interpolation.doubleInterp(x1=cpobjmin, y1=0, x2=cpobjmax, y2=0, x=cpobjs[e], y=0, f11=pstdmin, f12=0, f21=pstdmax, f22=0)
            
    # Conditional for 'double' and 'counter' cases, which own a double turbine stage. The rotational speed is computed by iteration and assuming that both stages are solidary (rotate at the same speed).
    elif mode in ["double", "double_wNmax", "counter", "counter_wNmax"]:
        # Least-square fitting and computation of average slope and ordinate-origin of the phi-psi functional relations of both stages.
        s0ave1, y0ave1 = psi_phi_slope_fitter(Phi=Phi1, Psi=Psi1, interpfit=True)
        s0ave2, y0ave2 = psi_phi_slope_fitter(Phi=Phi2, Psi=Psi2, interpfit=True)
        # Least-square fitting and computation of average slope and ordinate-origin of the global phi-psi functional relation.
        s0avetot, y0avetot = psi_phi_slope_fitter(Phi=Phitot, Psi=Psitot, interpfit=True)
        # K instantiation.
        K = s0avetot
        # Computation of psi-standard-deviation based on the assumption of linearity between stds.
        devobjs = [devobj/np.average([s0ave1, s0ave2]) for devobj in devobjs]                       
        # Setting the 'Ns', 'cpobj1s' and 'cpobj2s' lists that will store the values of the computed rotational speeds and the distributed damping factors of the stages.
        Ns = list()
        cpobj1s = list()
        cpobj2s = list()
        for cpobj in cpobjs:
            # Boolean 'N1N2match' variable for checking the convergence of the will-be-computed rotational speeds of the turbine stages.
            N1N2match = False
            # Setting the 'factor' variable to 0.5. It determines how the damping coefficient is distributed between the stages.
            factor = 0.5                
            # While loop until matching the rotational speeds of both turbine stages by modifying the distributive factor of the damping coefficient between stages.
            while not N1N2match:
                # Computing the complementary damping factors of the stages based on the value of the 'factor' variable.
                cpobj1 = cpobj*factor
                cpobj2 = cpobj*(1 - factor)
                # Computing the rotational speeds of the stages.
                N1 = np.abs((2*s0ave1*cpobj1*turb.turbstage1.geom.rtip/turb.turbstage1.flow.rho)*60/(2*np.pi))
                N2 = np.abs((2*s0ave2*cpobj2*turb.turbstage1.geom.rtip/turb.turbstage1.flow.rho)*60/(2*np.pi))
                # Checking for rotational-speed-convergence.
                N1N2match = np.abs(N1 - N2) < 5
                if N1N2match:
                    break
                # Computing the complementary damping factors of the stages based on the value of the 'factor' variable.
                cpobj1 = cpobj*(1 - factor)
                cpobj2 = cpobj*factor
                # Computing the rotational speeds of the stages.
                N1 = np.abs((2*s0ave1*cpobj1*turb.turbstage1.geom.rtip/turb.turbstage1.flow.rho)*60/(2*np.pi))
                N2 = np.abs((2*s0ave2*cpobj2*turb.turbstage1.geom.rtip/turb.turbstage1.flow.rho)*60/(2*np.pi))
                # Checking for rotational-speed-convergence.
                N1N2match = np.abs(N1 - N2) < 5
                # Reducing distributive factor.
                factor -= 0.0001
                # Checking whether the factor has dropped below 0; if so, then break the loop (it has no physical rationale anymore).
                if factor < 0:
                    break
            # Computing average rotational speed (having converged, picking either of the rotational speeds of the turbine stages or the average itself should not have further relevance).
            Ns.append(0.5*(N1 + N2))
            cpobj1s.append(cpobj1)
            cpobj2s.append(cpobj2)

            # Conditional for checking whether the fitness function is to be considered with the Nmax modification.
            if 'wNmax' in mode:
                # Loop running over the calculated rotational speeds for checking them and, in case of overcoming the upper limit,
                # setting their value to Nmax and recalculating the standard deviation value of the pressure fluctuations by
                # look-up table interpolation.
                for e, N in enumerate(Ns):
                    # Checker conditional.
                    if N > Nmax:
                        # Modifying Ns[e] value.
                        Ns[e] = Nmax
                        # Modifying cpobjs[e] value.
                        cpobjs[e] = Nmax*(2*np.pi)/((2*(np.average([s0ave1,s0ave2])**(-1))*turb.turbstage1.geom.rtip/turb.turbstage1.flow.rho)*60)
                        cpobj1s[e] = cpobjs[e]*(1 - factor)
                        cpobj2s[e] = cpobjs[e]*factor
                        # Getting maximum and minimum indices for interpolating in the cpto-pstd look-up table.
                        indexmin = [cpobjs[e] - _ < 0 for _ in cpto['cpto']].index(True) - 1
                        indexmax = [_ - cpobjs[e] > 0 for _ in cpto['cpto']].index(True)
                        # Getting the maximum and minimum cpobjs (x-axis) for interpolation.
                        cpobjmin = cpto['cpto'][indexmin]
                        cpobjmax = cpto['cpto'][indexmax]
                        # Getting the maximum and minimum pstds (y-axis) for interpolation.
                        pstdmin = cpto[e + 1][indexmin]
                        pstdmax = cpto[e + 1][indexmax]
                        # Calling 'double_interp()' function for recalculating the standard deviation of pressure and modifying devobjs[e] value.
                        devobjs[e] = mt.interpolation.doubleInterp(x1=cpobjmin, y1=0, x2=cpobjmax, y2=0, x=cpobjs[e], y=0, f11=pstdmin, f12=0, f21=pstdmax, f22=0)       
    
    # Conditional for 'pitchopt', 'pitchopt_wNmax', 'twistopt' and 'twistopt_wNmax' cases. The rotational speed must be calculated for each sea-state and optimal pitch.
    elif mode in ["pitchopt", "pitchopt_wNmax", "twistopt", "twistop_wNmax"]:
        # Least-square fitting and computation of average slope and ordinate-origin of the phi-psi functional relation.
        if slope_calc == "wOptPitch":
            s0ave, y0ave = psi_phi_slope_fitter(Phi=Phitot, Psi=optpsispow, interpfit=True)
        elif slope_calc == "wStandard":
            # Applying BEM method to current phenotype-turbine when running in 'mono' mode.
            turb.BEM(vxlist, inputparam='vx', mode='3DAD')
            s0ave, y0ave = psi_phi_slope_fitter(Phi=turb.Phi, Psi=turb.Psi, interpfit=True)
        # K instantiation.
        K = s0ave
        # Computation of the rotational speeds for each cpobj.
        Ns = [(2*K*cpobj*turb.turbstage1.geom.rtip/turb.turbstage1.flow.rho)*60/(2*np.pi) for cpobj in cpobjs] 
        
        # Conditional for checking whether the fitness function is to be considered with the Nmax modification.
        if 'wNmax' in mode:
            # Loop running over the calculated rotational speeds for checking them and, in case of overcoming the upper limit,
            # setting their value to Nmax and recalculating the standard deviation value of the pressure fluctuations by
            # look-up table interpolation.
            for e, N in enumerate(Ns):
                # Checker conditional.
                if N > Nmax:
                    # Modifying Ns[e] value.
                    Ns[e] = Nmax
                    # Modifying cpobjs[e] value.
                    cpobjs[e] = Nmax*(2*np.pi)/((2*K*turb.turbstage1.geom.rtip/turb.turbstage1.flow.rho)*60)
                    # Getting maximum and minimum indices for interpolating in the cpto-pstd look-up table.
                    indexmin = [cpobjs[e] - _ < 0 for _ in cpto['cpto']].index(True) - 1
                    indexmax = [_ - cpobjs[e] > 0 for _ in cpto['cpto']].index(True)
                    # Getting the maximum and minimum cpobjs (x-axis) for interpolation.
                    cpobjmin = cpto['cpto'][indexmin]
                    cpobjmax = cpto['cpto'][indexmax]
                    # Getting the maximum and minimum pstds (y-axis) for interpolation.
                    pstdmin = cpto[e + 1][indexmin]
                    pstdmax = cpto[e + 1][indexmax]
                    # Calling 'double_interp' function for recalculating the standard deviation of pressure and modifying devobjs[e] value.
                    devobjs[e] = mt.interpolation.doubleInterp(x1=cpobjmin, y1=0, x2=cpobjmax, y2=0, x=cpobjs[e], y=0, f11=pstdmin, f12=0, f21=pstdmax, f22=0)
        
        # Declaring auxiliary variables for storing optimum dimensionless variables for each recomputed power-optimized sea-state curve.
        phis = list()        
        optangs = list()
        optpsis = list()
        optpis = list()
        # Loop running over the calculated rotational speeds for each sea-state.
        for N in Ns:            
        # Setting rotational speed of turbine to calculated output value.
            turb.turbstage1.omega = N*2*np.pi/60
            # Conditional for determining which optimization function is to be called.
            if "pitchopt" in mode:
                # Calling the 'pitch_optimization()' function.
                Phitot, optangpitchespow, optpsispow, optpispow, opteffspow = turb.pitch_optimization(vxlist=vxlist, param=param, show_progress=False)
            elif "twistopt" in mode:
                # Calling the 'twist_optimization()' function.
                Phitot, optangpitchespow, optpsispow, optpispow, opteffspow = turb.twist_optimization(vxlist=vxlist, opt_mode=opt_mode, hubtip_var=hubtip_var, show_progress=False)
            # Appending results to auxiliary variables.
            phis.append(Phitot)
            optangs.append(optangpitchespow)
            optpsis.append(optpsispow)
            optpis.append(optpispow)     
               
    # Recasting the units of the target standard deviation of the flow parameter.
    if devobj_par == "pressure":
        devobjs = [devobj/(turb.turbstage1.flow.rho*((Ns[e]*2*np.pi/60)**2)*(2*turb.turbstage1.geom.rtip)**2) for e, devobj in enumerate(devobjs)]
    elif devobj_par == "flow":
        devobjs = [devobj/((Ns[e]*2*np.pi/60)*(2*turb.turbstage1.geom.rtip)**3) for e, devobj in enumerate(devobjs)]

    # Computing minimum and maximum devobj value and storing it in 'max_devobj' variable. The values are set to 0.85 times the minimum value (so that the gaussian does not start, precisely, at the minimum point and, thus, such a point is captured correctly) and the maximum divided by 0.85 (equal reason, but for the opposite end of the gaussian).
    min_devobj = np.min(devobjs)*0.85
    max_devobj = np.max(devobjs)/0.85
        
        #--------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------PERFORMING STOCHASTIC ANALYSIS-----------------------------------------#           
        #--------------------------------------------------------------------------------------------------------------------#     
    
    # Declaring list-type variablese for storing output dimensionless standard deviations, powers, efficiencies and fitnesses.
    dev_mininds = list()
    pi_mininds = list()
    pow_mininds = list()
    eff_mininds = list()
    fitnesses = list()
    
    # Performing the stochastic analysis for the 'mono', 'double' and 'counter' cases.    
    if mode in ['mono', 'mono_wNmax', 'double', 'double_wNmax', 'counter', 'counter_wNmax']:
        # Applying the 'stochastic_analysis' method.
        turb_stoch = stochastic_analysis(Phitot, Psitot, Pitot, mindev=min_devobj, maxdev=max_devobj, n=100)                
        # Set the output standard deviations, stochastic dimensionless powers and stochastic efficiencies to 'devs', 'pis' and 'effs', respectively.
        devs = turb_stoch[0]
        pis = turb_stoch[2]
        effs = turb_stoch[3]
        # Interpolate between 'devs' and 'effs' slinearly.
        f_eff = interp1d(devs, effs, kind='slinear', fill_value='extrapolate')
        # Interpolate between 'devs' and 'pis' slinearly.
        f_pi = interp1d(devs, pis, kind='slinear', fill_value='extrapolate')
        # Perform the fitting for getting the stochastic efficiency value at the target standard deviation.
        dev_fit = np.linspace(devs[0], devs[-1], 1000)
        pi_fit = f_pi(dev_fit)
        eff_fit = f_eff(dev_fit)
        # Initializing fitness value to 0.
        fitness = 0        
        # Loop running over the calculated rotational speeds for the sea-states.
        for e, N in enumerate(Ns):
            # Conditional for checking whether the computed rotational speed stands between the limits of the provided rotational speeds.
            if Nmin < N <= Nmax:
                # Compute the index at which the minimum difference is obtained between the target standard deviation and the stochastic
                # efficiency points computed from the fitting.
                minind = np.argmin(np.array([np.abs(_ - devobjs[e]) for _ in dev_fit]))
                dev_mininds.append(dev_fit[minind])
                pi_mininds.append(pi_fit[minind])
                eff_mininds.append(eff_fit[minind])
                # Conditional for computing the fitness depending on the fitness paramter to be evaluated; case 'eff' (efficiency).
                if fit_param == 'eff':
                    # Summing to the computation of the fitness value.
                    fitness += weights[e]*eff_fit[minind]
                    fitnesses.append(weights[e]*eff_fit[minind])
                # case 'pi' (dimensionless power).
                elif fit_param == 'pi':
                    # Summing to the computation of the fitness value.
                    fitness += weights[e]*pi_fit[minind]
                    fitnesses.append(weights[e]*pi_fit[minind])
                # case 'pow' (dimensional power).
                elif fit_param == 'pow':
                    # Summing to the computation of the fitness value.                    
                    fitness += weights[e]*pi_fit[minind]*(turb.turbstage1.flow.rho*((N*2*np.pi/60)**3)*(2*turb.turbstage1.geom.rtip)**5)
                    fitnesses.append(weights[e]*pi_fit[minind]*(turb.turbstage1.flow.rho*((N*2*np.pi/60)**3)*(2*turb.turbstage1.geom.rtip)**5))
                # Conditional for when the algorithm is run in 'double' or 'counter' mode.
                if mode in ['double', 'double_wNmax', 'counter', 'counter_wNmax']:
                    # The 'fitness' values are set to the stochastic efficiency value, the computed rotational speed, and the distributed
                    # damping values between the stages.
                    if eff_fit[minind] == 0:
                        Ns[e] = 0
                        cpobj1s[e] = 0
                        cpobj2s[e] = 0
                    else:
                        pass
            # In case the computed rotational speed does not fall between the limits of the provided rotational speed values, then return a null fitness value, with its corresponding side-values being set to 0 as well.
            else:
                dev_mininds.append([0])
                eff_mininds.append([0])                
                fitness += 0
                fitnesses.append([0])
                Ns[e] = 0
                # Adding null 'cpobj1s' and 'cpobj2s' values for the 'double' and 'counter' cases.
                if mode in ['double', 'double_wNmax', 'counter', 'counter_wNmax']:
                    cpobj1s[e] = 0
                    cpobj2s[e] = 0    

    # Performing the stochastic analysis for the 'pitchopt', 'pitchopt_wNmax', 'twistopt' and 'twistopt_wNmax' cases.   
    elif mode in ["pitchopt", "pitchopt_wNmax", "twistopt", "twistopt_wNmax"]:
        # Declaring auxiliary variables for the dimensionless standard deviations, powers, efficiencies and their corresponding fitted curves.
        dev_mins = list()
        pi_mins = list()
        pow_mins = list()
        eff_mins = list()
        fits = list()
        dev_fit = list()
        pi_fit = list()
        pow_fit = list()
        eff_fit = list()        
        # Declaring 'vxlist_optangpitches' variable for storing the optimum pitch angles for each input velocity at each sea-state.
        vxlist_optangpitches = list()
        # Initializing fitness value to 0.
        fitness = 0        
        # Loop running over the sea-states and computing the stochastic calculations accordingly; storing the values in 'turb_stochs'.
        for e, N in enumerate(Ns):
            # Applying the 'stochastic_analysis' method.
            turb_stoch = stochastic_analysis(phis[e], optpsis[e], optpis[e], mindev=min_devobj, maxdev=max_devobj, n=100)
            # Set the output standard deviations and stochastic efficiencies to 'devs' and 'effs', respectively.
            devs = turb_stoch[0]
            pis = turb_stoch[2]
            effs = turb_stoch[3]
            # Interpolate between 'devs' and 'effs' slinearly.
            f_eff = interp1d(devs, effs, kind='slinear', fill_value='extrapolate')
            # Interpolate between 'devs' and 'pis' slinearly.
            f_pi = interp1d(devs, pis, kind='slinear', fill_value='extrapolate')
            # Perform the fitting for getting the stochastic power and efficiency values at the target standard deviation.
            dev_f = np.linspace(devs[0], devs[-1], 1000)            
            pi_f = f_pi(dev_f)
            eff_f = f_eff(dev_f)            
            # Conditional for checking whether the computed rotational speed stands between the limits of the provided rotational speeds.
            if Nmin < N <= Nmax:
                # Compute the index at which the minimum difference is obtained between the target standard deviation and the stochastic
                # efficiency points computed from the fitting.
                dev_fit.append(dev_f)
                pi_fit.append(pi_f)
                pow_fit.append(pi_f*(turb.turbstage1.flow.rho*(np.abs(turb.turbstage1.omega)**3)*(2*turb.turbstage1.geom.rtip)**5))
                eff_fit.append(eff_f)
                minind = np.argmin(np.array([np.abs(_ - devobjs[e]) for _ in dev_f]))
                dev_mins.append(dev_f[minind])
                pi_mins.append(pi_f[minind])
                pow_mins.append(pi_f[minind]*(turb.turbstage1.flow.rho*(np.abs(turb.turbstage1.omega)**3)*(2*turb.turbstage1.geom.rtip)**5))
                eff_mins.append(eff_f[minind])
                # Conditional for computing the fitness depending on the fitness paramter to be evaluated; case 'eff' (efficiency).
                if fit_param == 'eff':
                    # Summing to the computation of the fitness value.
                    fitness += weights[e]*eff_f[minind]
                    fits.append(weights[e]*eff_f[minind])
                # case 'pi' (dimensionless power).
                elif fit_param == 'pi':
                    fitness += weights[e]*pi_f[minind]
                    fits.append(weights[e]*eff_f[minind])
                # case 'pow' (dimensional power).
                elif fit_param == 'pow':
                    fitness += weights[e]*pi_fit[minind]*(turb.turbstage1.flow.rho*((N*2*np.pi/60)**3)*(2*turb.turbstage1.geom.rtip)**5)
                    fits.append(weights[e]*pi_fit[minind]*(turb.turbstage1.flow.rho*((N*2*np.pi/60)**3)*(2*turb.turbstage1.geom.rtip)**5))
                vxlist_optangpitches.append(list(itertools.chain.from_iterable(zip(vxlist, optangs[e]))))
            # In case the computed rotational speed does not fall between the limits of the provided rotational speed values, then return a null fitness value, with its corresponding side-values being set to 0 as well.
            else:
                dev_fit.append(np.array([0]))
                pi_fit.append(np.array([0]))
                pow_fit.append(np.array([0]))
                eff_fit.append(np.array([0]))
                dev_mins.append(0)
                pi_mins.append(0)
                pow_mins.append(0)
                eff_mins.append(0)                                
                fitness += 0                
                fits.append(0)
                Ns[e] = 0
                vxlistaux = [0 for _ in vxlist]
                optangpitchespowaux = [0 for _ in optangs[e]]
                vxlist_optangpitches.append(list(itertools.chain.from_iterable(zip(vxlistaux, optangpitchespowaux))))
        dev_mininds.append(dev_mins)
        pi_mininds.append(pi_mins)
        pow_mininds.append(pow_mins)
        eff_mininds.append(eff_mins)
        fitnesses.append(fits)    

        #--------------------------------------------------------------------------------------------------------------------#
        #---------------------------------------------------CONFIGURING OUTPUT-----------------------------------------------#
        #--------------------------------------------------------------------------------------------------------------------#
    
    # Setting the 'fitness_out' parameter with the minimal (reduced) output data-set comprising the fitness value, the slope value K and the geometrical parameters of the solution.
    fitness_out = [fitness] + [K] + [_ for _ in solution]    
    # Conditional for checking whether a reduced output ('out_params'=='red') or an extended one ('out_params'=='ext') is to be returned. In the latter case, an additional conditional tree is executed for populating the 'fitness_out' with the mode-specific data.
    if out_params == "red":
        fitness_out = tuple(fitness_out)
    elif out_params == "ext":
         # Conditional for populating the output 'fitness_out' variable.
        if 'mono' in mode:
            # In the 'mono' case, the output variable comprises the fitness value, the slope 'K' and the rotational speed 'Ns'.
            fitness_out += Ns
            # Recasting the output variable to tuple.
            fitness_out = tuple(fitness_out)
        elif 'double' in mode or 'counter' in mode:
            # In the 'counter' case, the output variable comprises the fitness value, the slope 'K' and the set of rotational speeds and cpobj distributions between the turbine stages for each sea-state.
            for e, _ in enumerate(Ns):
                fitness_out += [Ns[e]]
                fitness_out += [cpobj1s[e]]
                fitness_out += [cpobj2s[e]]
            # Recasting the output variable to tuple.
            fitness_out = tuple(fitness_out)
        elif 'pitchopt' in mode:
            # In the 'pitchopt' case, the output variable comprises the fitness value, the slope 'K', the rotational speed 'Ns' and the set of optimum angular pitches for the input velocities.
            for e, N in enumerate(Ns):
                fitness_out += [Ns[e]]
                fitness_out += vxlist_optangpitches[e]
            # Recasting the output variable to tuple.
            fitness_out = tuple(fitness_out)

    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------RETURN-------------------------------------------------------#           
    #--------------------------------------------------------------------------------------------------------------------#    
    
    # Return statement.
    if out == 'fitness':
        return fitness_out
    else:
        # Conditional for fitness-paramater-based return statement, in case 'out == stfitness'; case 'eff' (efficiency).
        if fit_param == 'eff':
            return dev_fit, eff_fit, dev_mininds, eff_mininds, fitnesses
        # case 'pi' (dimensionless power).
        elif fit_param == 'pi':
            return dev_fit, eff_fit, dev_mininds, pi_mininds, fitnesses
        # case 'pow' (dimensional power).
        elif fit_param == 'pow':
            return dev_fit, eff_fit, dev_mininds, pow_mininds, fitnesses
    
#--------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------weighted_fitness_func() METHOD-------------------------------------------# 
#--------------------------------------------------------------------------------------------------------------------#    
def weighted_fitness_func(ga_instance=None, solution=None, solution_idx=None, **kwargs) -> tuple:
    '''It applies a weighted fitness function (site-variant).
    
    **parameters**:
    :param ga_instance: instance of a pygad genetic-algorithm object.        
    :param solution: a list of values that represent, on a GA run, the genotype chain. They specify the values to be adopted by the variable parameters contained in the "varargs" variable.
    :param solution_idx: index of the genotype chain within the current population.    
    :param constargs (kwarg): list of constant parameters, meaning those that are not supposed to vary during a GA run. Each of them holds specified values.
    :param varargs (kwarg): variable parameters, meaning thoste that are supposed to vary during a GA run. They hold genotype values contained in the "solution" variable.
    :param mode_fitness (kwarg): string specifying how the GA instance is run. Either "GA" (for running an instance of the GA algorithm) or "NoGA" for calling the function as a plain subroutine.
    :param out (kwarg): string specifying the type of output. Either 'fitness' (for getting a standard fitness output) or 'stfitness' (for getting stochastic-related fitness output).
    :param turb (kwarg): instance of a 'turbine' class; required in the case the "mode_fitness" variable is "NoGA".
    :param weights (kwarg): list of weights for pondering the single fitness functions across the provided sites.
    
    **return**:
    :return: either a tuple containing fitness information, or a tuple containing stochastic-related fitness information.
    
    **rtype**:
    :rtype: tuple.
    '''  

    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#           
    #--------------------------------------------------------------------------------------------------------------------#     

    # Getting kwargs keys.
    kwkeys = kwargs.keys()
    # Asserting that 'constargs' is present, and assigning it to 'constargs_list' variable.
    assert 'constargs' in kwkeys, "'constargs' variable not provided. It must be passed as a kwarg."
    constargs_list = kwargs['constargs']
    # Asserting that 'varargs' is present, and assigning it to 'varargs' variable.
    assert 'varargs' in kwkeys, "'varargs' variable not provided. It must be passed as a kwarg."
    varargs = kwargs['varargs']
    # Asserting that 'mode_fitness' is present, and assigning it to 'mode_fitness' variable.
    assert 'mode_fitness' in kwkeys, "'mode_fitness' variable not provided. It must be passed as a kwarg."
    mode_fitness = kwargs['mode_fitness']
    # Asserting that 'out' is present, and assigning it to 'out' variable.
    assert 'out' in kwkeys, "'out' variable not provided. It must be passed as a kwarg."
    out = kwargs['out']    
    # Asserting that 'weights' is present, and assigning it to 'weights' variable.
    assert 'weights' in kwkeys, "'weights' variable not provided. It must be passed as a kwarg."
    weights = kwargs['weights']
    # Asserting that 'weights' variable is a list-type variable.
    assert type(weights) == list, 'Please provide a list of weights.'
    # Asserting that 'constargs_list' and 'weights' variables have the same length.
    assert len(constargs_list) == len(weights), 'Please ensure that the number of different constarg sets equals that of the provided weights.'    
    if mode_fitness == 'GA':
        # Asserting that 'solution' and 'solution_idx' are not None (have valid values)."
        assert not (solution is None), "Provide a valid 'solution' parameter."
        assert not (solution_idx is None), "Provide a valid 'solution_idx' parameter."
        # Assigning 'None' value to 'turb' variable.
        turb = None
    else:        
        # Asserting that 'turb' is present, and assigning it to 'turb' variable.
        assert 'turb' in kwkeys, "'turb' variable not provided. It must be passed as a kwarg."
        turb = kwargs['turb']
        # Asserting that 'turb' is not None (has a valid value).
        assert not (turb is None), "Provide a valid 'turb' parameter."    

    #--------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------BODY--------------------------------------------------------#           
    #--------------------------------------------------------------------------------------------------------------------#
        
    # Converting solution to tuple-form for serving as dictionary entry.
    if hasattr(ga_instance, "processed_genes"):
        tup = tuple(solution)
        if tup in ga_instance.processed_genes.keys():
            # If so, returning fitness value.
            return tuple(ga_instance.processed_genes[tup])

    # Conditional for "mode_fitness='GA'" and "out=='fitness'" or "mode_fitness=='NoGA'" and "out=='fitness'".
    if (mode_fitness=='GA' and out=='fitness') or (mode_fitness=='NoGA' and out=='fitness'):
        # Creating 'fitness_out' and 'fitness_out_aux' lists for storing the outcomes.
        fitness_out = list()
        fitness_out_aux = list()
        # Enumerated for loop for calling 'single_fitness_func' on each of the sites with the corresponding weight value.
        for e, weight in enumerate(weights):
            # In the case it is the first iteration (first site) the outcome is directly damped into the 'fitness_out' variable (initialization).
            if e == 0:
                # Calling the 'single_fitness_func' function.
                fitness_out = single_fitness_func(ga_instance=ga_instance, constargs=constargs_list[e], varargs=varargs, turb=turb, solution=solution, solution_idx=solution_idx, mode_fitness=mode_fitness, out=out)
                # Turning the 'fitness_out' into a list.
                fitness_out = [_ for _ in fitness_out]
                # Pondering the outcome's first entry, which corresponds to the fitness value.
                fitness_out[0] *= weight
            # In the case it is an iteration other than the first one (other sites) the outcome is damped into the 'fitness_out_aux' variable, and then combined with the 'fitness_out' variable for merging results.
            else:
                # Calling the 'single_fitness_func' function.
                fitness_out_aux = single_fitness_func(ga_instance=ga_instance, constargs=constargs_list[e], varargs=varargs, turb=turb, solution=solution, solution_idx=solution_idx, mode_fitness=mode_fitness, out=out)
                # Turning the 'fitness_out_aux' into a list.
                fitness_out_aux = [_ for _ in fitness_out_aux]
                # Pondering the outcome's first entry, which corresponds to the fitness value.
                fitness_out[0] += fitness_out_aux[0]*weight                
                # For loop for merging results (sea-state-related data) with the 'fitness_out' variable.
                for f, _ in enumerate(fitness_out_aux[2 + len(solution):]):
                    # Appending results.
                    fitness_out.append(_)
    # Conditional for "mode_fitness='NoGA'" and "out=='stfitness'" case.
    elif mode_fitness=='NoGA' and out=='stfitness':
        # Creating 'fitness_out' and 'fitness_out_aux' lists for storing the outcomes.
        fitness_out = list()
        fitness_out_aux = list()
        # Enumerated for loop for calling 'single_fitness_func' on each of the sites with the corresponding weight value.
        for e, weight in enumerate(weights):
            # In the case it is the first iteration (first site) the outcome is directly damped into the 'fitness_out' variable (initialization).
            if e == 0:
                # Calling the 'single_fitness_func' function.
                fitness_out = np.array(single_fitness_func(ga_instance=ga_instance, constargs=constargs_list[e], varargs=varargs, turb=turb, solution=solution, solution_idx=solution_idx, mode_fitness=mode_fitness, out=out))
                # Turning the 'fitness_out' into a list.
                fitness_out = list(fitness_out)
                # For loop running over the 'stfitness' outputs that have to do with the discrete standard deviation and efficiency values, and the discrete fitness values.
                for i in range(2, 5):
                    # Turning the corresponding array into a list and recasting [0]-like null-valued lists into plain null values.
                    fitness_out[i] = [_ if _ != [0] else 0 for _ in fitness_out[i]]
                    # Turning the corresponding lists into arrays.
                    fitness_out[i] = np.array(fitness_out[i])
                    # In the case the considered list corresponds to the discrete fitness values, pondering the results.
                    if i == 4:
                        # Pondering the results.
                        fitness_out[i] *= weight
                    # Turning the corresponding arrays into lists.
                    fitness_out[i] = list(fitness_out[i])
            # In the case it is an iteration other than the first one (other sites) the outcome is damped into the 'fitness_out_aux' variable, and then combined with the 'fitness_out' variable for merging results.
            else:
                # Calling the 'single_fitness_func' function.
                fitness_out_aux = np.array(single_fitness_func(ga_instance=ga_instance, constargs=constargs_list[e], varargs=varargs, turb=turb, solution=solution, solution_idx=solution_idx, mode_fitness=mode_fitness, out=out))
                # Turning the 'fitness_out_aux' into a list.
                fitness_out_aux = list(fitness_out_aux)
                # For loop running over the 'stfitness' outputs that have to do with the discrete standard deviation and efficiency values, and the discrete fitness values.
                for i in range(2, 5):
                    # Turning the corresponding array into a list and recasting [0]-like null-valued lists into plain null values.
                    fitness_out_aux[i] = [_ if _ != [0] else 0 for _ in fitness_out_aux[i]]
                    # In the case the considered list corresponds to the discrete fitness values, pondering the results and combining them with 'fitness_out' variable (not an appending, but a merging operation takes place, without modifying the length of the outcome list).
                    if i == 4:
                        # Pondering and merging results.
                        fitness_out[i] += list(np.array(fitness_out_aux[i])*weight)
                    # Otherwise, an appending operation takes place, modifying the length of the outcome list.
                    else:
                        # Appending 'fitness_out_aux' outcome to 'fitness_out' variable.
                        fitness_out[i] += fitness_out_aux[i]
        
    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------RETURN-------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------------------#         

    # Recasting 'fitness_out' into tuple.
    fitness_out = tuple(fitness_out)
    # Return statement.
    return fitness_out

#--------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------on_start() METHOD-------------------------------------------------# 
#--------------------------------------------------------------------------------------------------------------------#
def on_start(ga_instance, **kwargs) -> None:    
    '''Function that is executed at the initialization stage of a GA run. It creates two files: one for damping the computed genes after each generation has been completed; and another one for damping the set of different computed genes after the GA generations have been completed.

    **parameters**:
    :param ga_instance: instance of a pygad genetic-algorithm object.
    :param constargs (kwarg): constant parameters, meaning those that are not supposed to vary during a GA run. They hold specified values. It is a dictionary-type variable whose "mode" key's value is used for setting the format of the strings to be damped into the output file.    
    :param writepops (kwarg): it is a boolean-type parameter that will tell whether the properties are to be written to files.
    :param rootfilename (kwarg): it is a string-type parameter that specifies the root directory in which the damping files are to be stored.
    :param pars_to_write (kwarg): a list containing the parameters to be written into the file.
    :param num_sea_states (kwarg): an integer specifying the number of sea-states considered during the optimization. It is used for setting the format of the strings to be damped into the output file.
    '''

    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------# 

    # Getting kwargs keys.
    kwkeys = kwargs.keys()
    # Asserting that 'constargs' is present, and assigning its "mode" key's value to 'Mode' variable.
    assert 'constargs' in kwkeys, "'constargs' variable not provided. It must be passed as a kwarg."
    constargs = kwargs['constargs']
    if type(constargs) != list:
        pass
    else:
        constargs = constargs[0]
    mode = constargs['Mode']
    # Asserting that 'writepops' is present, and assigning it to 'writepops' variable.
    assert 'writepops' in kwkeys, "'writepops' not provided. It must be passed as a kwarg."
    writepops = kwargs['writepops']
    # Asserting that 'rootfilename' is present, and assigning it to 'rootfilename' variable.
    assert 'rootfilename' in kwkeys, "'rootfilename' not provided. It must be passed as a kwarg."
    rootfilename = kwargs['rootfilename']
    # Asserting that 'pars_to_write' is present, and assigning it to 'pars_to_write' variable.
    assert 'pars_to_write' in kwkeys, "'pars_to_write' not provided. It must be passed as a kwarg."
    pars_to_write = kwargs['pars_to_write']
    # Asserting that 'num_sea_states' is present, and assigning it to 'num_sea_states' variable.
    assert 'num_sea_states' in kwkeys, "'num_sea_states' not provided. It must be passed as a kwarg."
    num_sea_states = kwargs['num_sea_states']
    # Asserting that 'out_params' is present, and assigning it to 'out_params' variable.
    assert 'out_params' in kwkeys, "'out_prams' variable not provided. It must be passed as a kwarg."
    out_params = kwargs['out_params']    
    
    #--------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------BODY--------------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#     
    
    # Setting "processed_genes" attribute as an empty dictionary; it will store the set of all computed genes during the calculation.
    setattr(ga_instance, "processed_genes", dict())
    # Conditional for checking wheter the populations and computed genes are to be written to files.
    if writepops:
        # If so, checking whether the 'rootfilename.txt' generic file exists.
        rootfilenameaux = ""
        if os.path.isfile(".".join([rootfilename, "txt"])):
            # If so, then iterate over programatically modified filenames (modified by adding _1, _2, _3...to the root file).
            isfile = True
            i = 1
            # While loop for the modification of the filename.
            while isfile:
                rootfilenameaux = "_".join([_ for _ in rootfilename.split("_")[:-1]] +  [str(i)])
                isfile = os.path.isfile(".".join([rootfilenameaux, "txt"]))
                i += 1
        # Setting 'filename' attribute to the provided/computed 'rootfilename.txt' string. It will store generation-wise populations.
        filename = rootfilenameaux if rootfilenameaux != "" else rootfilename
        setattr(ga_instance, "filename", ".".join([filename, "txt"]))
        pars_to_write_ = [_ for _ in pars_to_write]
        if out_params == 'ext':
            if 'mono' in mode:
                if type(num_sea_states) != list:
                    pars_to_write_ += ["N_sea-state_" + str(i) for i in range(1, num_sea_states + 1)]
                else:
                    for e, ss in enumerate(num_sea_states):
                        pars_to_write_ += ["N_sea-state_site-" + str(e + 1) + "_" + str(i) for i in range(1, ss + 1)]
            elif 'double' in mode or 'counter' in mode:
                for i in range(1, num_sea_states + 1):
                    pars_to_write_.append("N_sea-state_" + str(i))
                    pars_to_write_.append("cpobj1_sea-state_" + str(i))
                    pars_to_write_.append("cpobj2_sea-state_" + str(i))
            elif 'pitchopt' in mode:
                for i in range(1, num_sea_states + 1):
                    pars_to_write_ += ["N_sea_state_" + str(i)]
                    for j in range(1, 46):
                        pars_to_write_ += ["vx_" + str(j) + "_N_" + str(i), "pitchopt_" + str(j) + "_N_" + str(i)]
        pars_to_write_str = "        ".join(pars_to_write_) + "\n"
        with open(ga_instance.filename, "w") as f:
            f.write(pars_to_write_str)
            f.write("Generation #1" + "-"*(len("".join(pars_to_write_)) + 8*(len(pars_to_write_) - 1) - len("Generation #1")) + "\n")            
        # Setting 'filename_proc_genes' to the provided/computed 'rootfilename_proc_genes.txt' string. It will store the set of all computed genes during the calculation.
        setattr(ga_instance, "filename_proc_genes", ".".join([filename + "_proc_genes", "txt"]))
    else:
        # Otherwise, setting both 'filename' and 'filename_proc_genes' to mpty strings.
        setattr(ga_instance, "filename", "")
        setattr(ga_instance, "filename_proc_genes", "")    
            
    
#--------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------on_fitness() METHOD------------------------------------------------# 
#--------------------------------------------------------------------------------------------------------------------#
def on_fitness(ga_instance, last_population_fitness) -> None:
    '''Function that is executed before the fitness functions are computed at each GA generation. It checks whether any of the genes to be computed have been calculated before and, if so, it skips the calculation. It is intended for computational time-saving.

    **parameters**:
    :param ga_instance: instance of a pygad genetic-algorithm object.
    :param last_generation_fitness: the fitness values of the last population.
    '''
    
    #--------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------BODY--------------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#    
    
    # Loop over the genes of the currently run population to store them in the 'ga_instance.processed_genes' dictionary.
    for e, _ in enumerate(ga_instance.population):
        tup = tuple(_)
        # If the current gene has already been processed, then pass.
        if tup in ga_instance.processed_genes.keys():
            pass
        # Otherwise, add it to the processed genes' pool.
        else:
            ga_instance.processed_genes[tup] = tuple(last_population_fitness[e])
    
#--------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------on_generation() METHOD----------------------------------------------# 
#--------------------------------------------------------------------------------------------------------------------#
def on_generation(ga_instance, **kwargs) -> None:
    '''Function that is executed after each GA generation has been completed. It damps the overall set of computed genes in the last generation and their fitness values into a file.

    **parameters**:
    :param ga_instance: instance of a pygad genetic-algorithm object.
    
    **kwargs**:
    :param pars_to_write: a list containing the parameters to be written into the file.
    :param num_sea_states: an integer specifying the number of sea-states considered during the optimization. It is used for setting the format of the strings to be damped into the output file.
    :param constargs: constant parameters, meaning those that are not supposed to vary during a GA run. They hold specified values. It is a dictionary-type variable whose "mode" key's value is used for setting the format of the strings to be damped into the output file.
    :param decimals_list: a list containing the number of decimals to be included in the output file for each of the variables contained in the "pars_to_write" parameter.
    '''
    
    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#
    
    # Getting kwargs keys.
    kwkeys = kwargs.keys()
    # Asserting that 'pars_to_write' is present, and assigning it to 'pars_to_write' variable.
    assert 'pars_to_write' in kwkeys, "'pars_to_write' variable not provided. It must be passed as a kwarg."
    pars_to_write = kwargs['pars_to_write']
    # Asserting that 'num_sea_states' is present, and assigning it to 'num_sea_states' variable.
    assert 'num_sea_states' in kwkeys, "'num_sea_states' variable not provided. It must be passed as a kwarg."
    num_sea_states = kwargs['num_sea_states']
    # Asserting that 'constargs' is present, and assigning its "mode" key's value to 'Mode' variable.
    assert 'constargs' in kwkeys, "'constargs' variable not provided. It must be passed as a kwarg."
    constargs = kwargs['constargs']
    if type(constargs) != list:
        pass
    else:
        constargs = constargs[0]
    mode = constargs['Mode']
    # Asserting that 'decimals_list' is present, and assigning it to 'decimals_list' variable.
    assert 'decimals_list' in kwkeys, "'decimals_list' variable not provided. It must be passed as a kwarg."
    decimals_list = kwargs['decimals_list']
    # Asserting that 'out_params' is present, and assigning it to 'out_params' variable.
    assert 'out_params' in kwkeys, "'out_prams' variable not provided. It must be passed as a kwarg."
    out_params = kwargs['out_params']    
    
    #--------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------BODY--------------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#    
    
    # Monitoring (re)printing.
    if ga_instance.generations_completed == ga_instance.num_generations:
        print('\t\t Completed generations: ', ga_instance.generations_completed)
    else:
        print('\t\t Completed generations: ', ga_instance.generations_completed, end='\r')
    # If the provided filename is an empty string (no writing operations have been required) then pass.
    if ga_instance.filename == "":
        pass
    # Otherwise, proceed with the writing. 
    else:
        # Performing the file opening and writing operations.
        with open(ga_instance.filename, "a") as f:
            # Instantiating an empty list for storing the data to write.
            write_list = list()            
            if type(ga_instance.last_generation_fitness[0])==np.ndarray:
                if out_params == 'ext':
                    order_ind = len(ga_instance.population[0])
                elif out_params == 'red':
                    order_ind = len(pars_to_write) - 2
            else:
                order_ind = -1
            # Loop over the genes of the population for retrieving the genome and its fitness value.
            for _ in ga_instance.last_generation_fitness:
                if out_params == 'ext':
                    data = list(_[2:2 + len(ga_instance.population[0])]) + list(_[:2]) + list(_[2 + len(ga_instance.population[0]):])
                elif out_params == 'red':
                    data = list(_[2:2 + len(ga_instance.population[0])]) + list(_[:2])
                write_list.append(data)            
            # Sorting list data according to fitness value.            
            write_list = sorted(write_list, key=lambda x: x[order_ind], reverse=True)
            # Recasting to array.
            write_array = np.array(write_list)            
            write_list_str = [str(_) for _ in write_list]
            fmt = list()
            pars_to_write_ = [_ for _ in pars_to_write]
            if out_params == 'ext':
                if 'mono' in mode:
                    if type(num_sea_states) != list:
                        pars_to_write_ += ["N_sea-state_" + str(i) for i in range(1, num_sea_states + 1)]
                        decimals_list += num_sea_states*[0]       
                    else:
                        for e, ss in enumerate(num_sea_states):
                            pars_to_write_ += ["N_sea-state_site-" + str(e + 1) + "_" + str(i) for i in range(1, ss + 1)]
                            decimals_list += [0 for _ in range(ss)]
                elif 'double' in mode or 'counter' in mode:
                    for i in range(1, num_sea_states + 1):
                        pars_to_write_.append("N_sea-state_" + str(i))
                        pars_to_write_.append("cpobj1_sea-state_" + str(i))
                        pars_to_write_.append("cpobj2_sea-state_" + str(i))
                    decimals_list += num_sea_states*[0, 3, 3]
                elif 'pitchopt' in mode:
                    for i in range(1, num_sea_states + 1):
                        pars_to_write_ += ["N_sea_state_" + str(i)]
                        decimals_list += [0]
                        for j in range(1, 46):
                            pars_to_write_ += ["vx_" + str(j) + "_N_" + str(i)]
                            decimals_list += [0]
                            pars_to_write_ += ["pitchopt_" + str(j) + "_N_" + str(i)]
                            decimals_list += [0]
            # Writing to file.            
            for _ in write_array:
                fmt = list()                  
                for g, __ in enumerate(_):
                    if decimals_list[g] == 0:
                        _[g] = int(_[g])
                        diff_length = len(pars_to_write_[g]) - len(str(_[g])[:-2])
                    else:                        
                        diff_length = len(pars_to_write_[g]) - decimals_list[g] - 2
                    fmt.append("%." + str(decimals_list[g]) + "f" + diff_length*" " + 8*" ")
                np.savetxt(f, np.c_[[_]], fmt="".join(fmt))
            pars_to_write_ = "        ".join(pars_to_write_) + "\n"
            f.write("-"*(len(pars_to_write_)) + "\n")
            if ga_instance.generations_completed == ga_instance.num_generations:
                pass
            else:
                f.write(pars_to_write_)
                f.write("Generation #" + str(ga_instance.generations_completed + 1) + "-"*(len(pars_to_write_) - 1 - len("Generation #" + str(ga_instance.generations_completed + 1))) + "\n")
            # Closing the file.
            f.close()
            
#--------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------on_stop() METHOD-------------------------------------------------# 
#--------------------------------------------------------------------------------------------------------------------#
def on_stop(ga_instance, last_generation_fitness, **kwargs) -> None:
    '''Function that is executed after the GA generations have been completed. It damps the set of different computed genes and their fitness values into a file.

    **parameters**:
    :param ga_instance: instance of a pygad genetic-algorithm object.
    :param last_generation_fitness: the fitness values of the last population.
    
    **kwargs**:
    :param pars_to_write: a list containing the parameters to be written into the file.
    :param num_sea_states: an integer specifying the number of sea-states considered during the optimization. It is used for setting the format of the strings to be damped into the output file.
    :param constargs: constant parameters, meaning those that are not supposed to vary during a GA run. They hold specified values. It is a dictionary-type variable whose "mode" key's value is used for setting the format of the strings to be damped into the output file.
    :param decimals_list: a list containing the number of decimals to be included in the output file for each of the variables contained in the "pars_to_write" parameter.
    '''

    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#

    # Getting kwargs keys.
    kwkeys = kwargs.keys()
    # Asserting that 'pars_to_write' is present, and assigning it to 'pars_to_write' variable.
    assert 'pars_to_write' in kwkeys, "'pars_to_write' variable not provided. It must be passed as a kwarg."
    pars_to_write = kwargs['pars_to_write']
    # Asserting that 'num_sea_states' is present, and assigning it to 'num_sea_states' variable.
    assert 'num_sea_states' in kwkeys, "'num_sea_states' variable not provided. It must be passed as a kwarg."
    num_sea_states = kwargs['num_sea_states']
    # Asserting that 'constargs' is present, and assigning its "mode" key's value to 'Mode' variable.
    assert 'constargs' in kwkeys, "'constargs' variable not provided. It must be passed as a kwarg."
    constargs = kwargs['constargs']
    if type(constargs) != list:
        pass
    else:
        constargs = constargs[0]
    mode = constargs['Mode']
    # Asserting that 'decimals_list' is present, and assigning it to 'decimals_list' variable.
    assert 'decimals_list' in kwkeys, "'decimals_list' variable not provided. It must be passed as a kwarg."
    decimals_list = kwargs['decimals_list']
    # Asserting that 'out_params' is present, and assigning it to 'out_params' variable.
    assert 'out_params' in kwkeys, "'out_prams' variable not provided. It must be passed as a kwarg."
    out_params = kwargs['out_params']      
    
    #--------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------BODY--------------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#     
    
    # If the provided filename is an empty string (no writing operations have been required) then pass.
    if ga_instance.filename_proc_genes == "":
        pass
   # Otherwise, proceed with the writing. 
    else:
        # Performing the file opening and writing operations.
        with open(ga_instance.filename_proc_genes, "w") as f:
            # Instantiating an empty list for storing the data to write.
            write_list = list()
            if type(ga_instance.last_generation_fitness[0])==np.ndarray:
                if out_params == 'ext':
                    order_ind = len(ga_instance.population[0])
                elif out_params == 'red':
                    order_ind = len(pars_to_write) - 2
            else:
                order_ind = -1          
            # Loop over the genes of the population for retrieving the genome and its fitness value.
            for _ in ga_instance.processed_genes.keys():
                fit = ga_instance.processed_genes[_]      
                if out_params == 'ext':
                    data = list(fit[2:2 + len(ga_instance.population[0])]) + list(fit[:2]) + list(fit[2 + len(ga_instance.population[0]):])
                elif out_params == 'red':
                    data = list(fit[2:2 + len(ga_instance.population[0])]) + list(fit[:2])
                write_list.append(data)
            # Sorting list data according to fitness value.
            write_list = sorted(write_list, key=lambda x: x[order_ind], reverse=True)
            # Recasting to array.
            write_array = np.array(write_list)
            # Writing to file.
            write_list_str = [str(_) for _ in write_list]            
            fmt = list()
            pars_to_write_ = [_ for _ in pars_to_write]
            if out_params == 'ext':
                if 'mono' in mode:
                    if type(num_sea_states) != list:
                        pars_to_write_ += ["N_sea-state_" + str(i) for i in range(1, num_sea_states + 1)]
                        decimals_list += num_sea_states*[0]       
                    else:
                        for e, ss in enumerate(num_sea_states):
                            pars_to_write_ += ["N_sea-state_site-" + str(e + 1) + "_" + str(i) for i in range(1, ss + 1)]
                            decimals_list += [0 for _ in range(ss)]
                elif 'double' in mode or 'counter' in mode:
                    for i in range(1, num_sea_states + 1):
                        pars_to_write_.append("N_sea-state_" + str(i))
                        pars_to_write_.append("cpobj1_sea-state_" + str(i))
                        pars_to_write_.append("cpobj2_sea-state_" + str(i))
                    decimals_list += num_sea_states*[0, 3, 3]
                elif 'pitchopt' in mode:
                    for i in range(1, num_sea_states + 1):
                        pars_to_write_ += ["N_sea_state_" + str(i)]
                        decimals_list += [0]
                        for j in range(1, 46):
                            pars_to_write_ += ["vx_" + str(j) + "_N_" + str(i)]
                            decimals_list += [0]
                            pars_to_write_ += ["pitchopt_" + str(j) + "_N_" + str(i)]
                            decimals_list += [0]
            # Writing to file.
            pars_to_write_aux = "        ".join(pars_to_write_)
            f.write(pars_to_write_aux + "\n")
            f.write("-"*(len(pars_to_write_aux)) + "\n")
            for _ in write_array:
                fmt = list()
                for g, __ in enumerate(_):
                    if decimals_list[g] == 0:
                        _[g] = int(_[g])
                        diff_length = len(pars_to_write_[g]) - len(str(_[g])[:-2])
                    else:
                        diff_length = len(pars_to_write_[g]) - decimals_list[g] - 2
                    fmt.append("%." + str(decimals_list[g]) + "f" + diff_length*" " + 8*" ")
                np.savetxt(f, np.c_[[_]], fmt="".join(fmt))
            f.write("-"*(len(pars_to_write_aux)) + "\n")
            # Closing the file.
            f.close()

#--------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------plot_turbine() METHOD----------------------------------------------# 
#--------------------------------------------------------------------------------------------------------------------#
def plot_turbine(turb: turb) -> tuple:
    '''Plots front-view of a given turbine desing.
    
    **parameters**:
    :param turb: an instantiated object of the 'turbine' class.
    
    **return**:
    :return: a tuple containing a figure object and axes object.
    
    **rtype**:
    :rtypes: tuple. 
    '''
    
    #--------------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------BODY--------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------------------#     
    
    # Setting blade numbers to 'Z' variable.
    Z = turb.turbstage1.geom.Z
    # Setting 'tip_percent' to tip-percent.
    tip_percent = turb.turbstage1.geom.tip_percent
    # Setting 'rcas' to casing radius.
    rcas = turb.turbstage1.geom.rcas
  
    # Setting 'rhub' to hub radius.
    rhub = turb.turbstage1.geom.rhub
    # Setting 'chub' to hub chord.
    chub = turb.turbstage1.geom.chord[0]
    # Setting 'alphahub0' to angular position of first blade at hub. 
    alphahub0 = np.math.asin(chub/(2*rhub))
    # Setting complex 'zchub1' to complex-positions of the initial placings of the blades at hub.
    zchub1 = [np.math.e**(complex(0, _))*complex(-rhub*np.sin(alphahub0), rhub*np.cos(alphahub0)) for _ in [_*2*np.pi/Z for _ in range(1, Z + 1)]]
    # Setting complex 'zchub2' to complex-positions of the final placings of the blades at hub.
    zchub2 = [np.math.e**(complex(0, _))*complex(rhub*np.sin(alphahub0), rhub*np.cos(alphahub0)) for _ in [_*2*np.pi/Z for _ in range(1, Z + 1)]]    
    
    # Setting 'rtip' to tip radius.
    rtip = turb.turbstage1.geom.rtip
    # Setting 'ctip' to tip chord.
    ctip = turb.turbstage1.geom.chord[-1]
    # Setting 'alphatip0' to angular position of first blade at tip.
    alphatip0 = np.math.asin(ctip/(2*rtip))
    # Setting complex 'zctip1' to complex-positions of the initial placings of the blades at tip.
    zctip1 = [np.math.e**(complex(0, _))*complex(-rtip*np.sin(alphatip0), rtip*np.cos(alphatip0)) for _ in [_*2*np.pi/Z for _ in range(1, Z + 1)]]
    # Setting complex 'zctip2' to complex-positions of the final placings of the blades at tip.
    zctip2 = [np.math.e**(complex(0, _))*complex(rtip*np.sin(alphatip0), rtip*np.cos(alphatip0)) for _ in [_*2*np.pi/Z for _ in range(1, Z + 1)]]
    
    # Declaring 'arcs' and 'arcss' lists to store the arcs that represent the blades sections in a frontal view.
    arcs = list()
    arcss = list()
    
    # Loop running over the hub-tip blade portions.
    for zc1 in [zchub1, zctip1]:
        # Setting 'zc2' portion of hub or tip according to 'zc1' (depending on if it is a hub- or tip-related blade portion).
        zc2 = zchub2 if zc1 == zchub1 else zctip2
        # Setting diameter according to 'zc1' (depending on if it is a hub- or tip-related blade portion).
        d = 2*rhub if zc1 == zchub1 else 2*rtip
        # Loop running over the blade portions.
        for e, _ in enumerate(zc1):
            # Getting signs of hub and tip real and imaginary parts.
            sgnreal1 = np.sign(np.real(zc1[e]))
            sgnimag1 = np.sign(np.imag(zc1[e]))
            sgnreal2 = np.sign(np.real(zc2[e]))
            sgnimag2 = np.sign(np.imag(zc2[e]))
            # Conditionals for setting the angular bounds in degrees by computing the 'offset' and 'multisign' variables for hub and tip.
            if sgnreal1 < 0 and sgnimag1 < 0:
                offset1 = 180
                multsign1 = 1
            elif sgnreal1 > 0 and sgnimag1 < 0:
                offset1 = 360
                multsign1 = 1
            elif sgnreal1 < 0 and sgnimag1 > 0:
                offset1 = 180
                multsign1 = 1
            else:
                offset1 = 0
                multsign1 = 1
            if sgnreal2 < 0 and sgnimag2 < 0:
                offset2 = 180
                multsign2 = 1
            elif sgnreal2 > 0 and sgnimag2 < 0:
                offset2 = 360
                multsign2 = 1
            elif sgnreal2 < 0 and sgnimag2 > 0:
                offset2 = 180
                multsign2 = 1
            else:
                offset2 = 0
                multsign2 = 1
            # Computing the 'theta_1' and 'theta_2' angular bounds for setting the arcs.
            theta_1 = offset1 + multsign1*np.arctan(np.imag(zc1[e])/np.real(zc1[e]))*180/np.pi
            theta_2 = offset2 + multsign2*np.arctan(np.imag(zc2[e])/np.real(zc2[e]))*180/np.pi
            # Conditional for checking that 'theta_1' and 'theta_2' do not depart further than the maximum angular spacing, namely 360/Z.
            if theta_1 > theta_2 and theta_1 - theta_2 < 360/Z:
                st = "1"
                theta1 = theta_2
                theta2 = theta_1
            else:
                if theta_2 - theta_1 > 360/Z:
                    theta1 = theta_2
                    theta2 = theta_1
                else:
                    theta1 = theta_1
                    theta2 = theta_2 
            # Appending patch with computed diameters and angular counds to 'arcs' variable.
            arcs.append(mpl.patches.Arc(xy=(0, 0), width=d, height=d,
                                       theta1=theta1,
                                       theta2=theta2, linewidth=2, color='k'))

    # Setting hub circumference in front-view.
    hub = plt.Circle((0, 0), rhub, facecolor='k', alpha=0.3, edgecolor='k', linewidth=2)
    # Setting hub edge in front-view.
    hubedge = plt.Circle((0, 0), rhub, facecolor='None', edgecolor='k', linewidth=2)
    # Setting tip circumference in front-view.
    tip = plt.Circle((0, 0), rtip, facecolor='None', edgecolor='k', linestyle='--')
    # Setting casing circumference in front-view.
    cas = plt.Circle((0, 0), rcas, facecolor='None', edgecolor='k', linestyle='--')
        
    # Declaring figure and axes as ('fig', 'ax')
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Adding hub patch.
    ax.add_patch(hub)
    # Adding hub edge.
    ax.add_patch(hubedge)
    # Loop running over the blade portions and plotting lines between hubs and tips.
    for e, _ in enumerate(zchub1):
        ax.plot([np.real(zchub1[e]), np.real(zctip1[e])], [np.imag(zchub1[e]), np.imag(zctip1[e])], color='k', linewidth=2)
        ax.plot([np.real(zchub2[e]), np.real(zctip2[e])], [np.imag(zchub2[e]), np.imag(zctip2[e])], color='k', linewidth=2)
    
    # Adding tip patch.
    ax.add_patch(tip)
    # Loop running over the blade portions and adding arc patches.
    for arc in arcs:
        ax.add_patch(arc) 
          
    # Setting x-axis limits.
    ax.set_xlim((-rcas*1.1, rcas*1.05))
    # Setting y-axis limits.
    ax.set_ylim((-rcas*1.1, rcas*1.05))
    # Disabling ticks and axes.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')    
    # Annotating hub radius.
    ax.arrow(0, 0, 0.95*rhub/np.sqrt(2), 0.95*rhub/np.sqrt(2))
    ax.annotate('$r_{hub}$=' + str(rhub)[:4], xy=(0.75*rhub, 0.75*rhub), fontsize=20)
    # Annotating tip radius.
    ax.arrow(0, 0, -0.99*rtip/np.sqrt(2), 0.95*rtip/np.sqrt(2))
    ax.annotate('$r_{tip}$=' + str(rtip)[:4], xy=(-1.01*rtip, 0.7*rtip), fontsize=20)
    # Annotating blade number.
    ax.annotate('Z={Z}'.format(Z=Z), xy=(-0.15*rhub, -rhub/5), fontsize=20)
    # Annotating hub-to-tip-ratio.
    ax.annotate('$\\nu$={hub_to_tip_ratio:.2f}'.format(hub_to_tip_ratio=rhub/rtip), xy=(-0.25*rhub, -2*rhub/5), fontsize=20)
    # Annotating hub and tip chords.
    ax.annotate('$c_{hub}$=' + str(chub)[:4] + '; $c_{tip}$=' + str(ctip)[:4], xy=(-0.65*rhub, -3*rhub/5), fontsize=20)
    # Annotating tip percent.
    ax.annotate('gap={tip_percent}'.format(tip_percent=np.round(tip_percent, 2)), xy=(-0.25*rhub, -4*rhub/5), fontsize=20)

    #--------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------RETURN-------------------------------------------------------#             
    #--------------------------------------------------------------------------------------------------------------------#     
    
    # Return statement.
    return fig, ax

#--------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------CLASS turbine--------------------------------------------------#             
#--------------------------------------------------------------------------------------------------------------------#
class turbine:
    '''Class storing overall information (geometrical, analytical, experimental, CFD) about a turbine.'''

    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------DEFINING METHODS-------------------------------------------------#       
    #--------------------------------------------------------------------------------------------------------------------#

    #--------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------__init__() METHOD------------------------------------------------#       
    #--------------------------------------------------------------------------------------------------------------------# 
    def __init__(self, **kwargs) -> None:
        '''Constructor of the turbine class.
        
        **kwargs**:
        
        :param ttype: 'turbCnt.TYPES' instance specifying the type of turbine to be instantiated.
        :param omega: float specifying the rotational speed (in rpm) at which the turbine is operating. If not provided, default is 3600.
        :param N: integer specifying the number of radial blade elements whereby the turbine is discretized. If not provided, default is 50.
        :param rcas: float specifying the radial dimension of the turbine's casing. If not provided, default is 0.25.
        :param hub_to_tip_ratio: float specifying the hub-to-tip ratio of the turbine. If not provided, default is 0.75.
        :param chord: it can be a list, an array or a float, specifying the chordal dimension (in m) at each radial stage of the turbine. If it is a float, then the same chordal dimension is set at each radial stage. If it is a list or an array, its length must coincide with the amount 'N' of blade elements along the radial direction, and the distribution is set from hub to tip. If not provided, default is 0.117, which means that a constant-value chord of 0.117 meters is set at each radial stage.
        :param angpitch: it can be a list, an array or a float, specifying the angular pitch (in degrees) at each radial stage of the turbine (twist). If it is a float, then the same angular pitch is set at each radial stage (pitch). If it is a list or an array, its length must coincide with the amount 'N' of blade elements along the radial direction, and the distribution is set from hub to tip. If not provided, default is 0, which means that a constant-value pitch angle of 0 degrees (no pitch) is set at each radial stage.
        :param airfoil: it can be a list, an array or a single instance of the geCnt.PROFILES object, specifying the geometrical shape of the airfoil (retrieved from geCnt.PROFILES database) at each radial stage of the turbine. If it is a single element, then the same geometrical shape is set at each radial stage. If it is a list or an array, its length must coincide with the amount 'N' of blade elements along the radial direction, and the distribution is set from hub to tip. If not provided, default is geCnt.PROFILES.NACA0015, which means that a NACA0015 airfoil geometry is set at each radial stage.
        :param tip_percent: float specifying the gap at the tip of the turbine in terms of a percentage of the tip. If not provided, default is 0.5.
        :param Z: integer specifying the number of blades owned by each stage of the turbine. If not provided, default is 7.
        :param p: float specifying the ambient pressure at which the turbine is operating (Pa). If not provided, default is 101325.
        :param T: float specifying the ambient temperature at which the turbine is operating (K). If not provided, default is 288.15.
        :param R: float specifying the specific gas constant of the working fluid of the turbine (J/kg·K). If not provided, default is 287.058.
        :param nu: float specifying the viscosity of the working fluid of the the turbine (N/m·s). If not provided, default is 1.81e-5.               
        :param name: string specifying the name of the arfoil-object. If not provided, default is 'airfoil'.
        :param path: string specifying the root path for airfoil-object-related storage operations. If not provided, default is an empty string (""), meaning that the default path is the current working directory.
        :param load: boolean specifying whether the instance of the object is to be loaded from a previously saved pickle file.
        :param loadExpGeom: boolean specifying whether Tridim-related data-sets are to be loaded. If not provided, default is 'False'. If provided, then the name of the root directory containing the data-files must be provided in the form of a 'TridimSets_root_dir' string kwarg.
        :param TridimSets_root_dir: string specifying the root name of the directory containing the Tridim-related data-sets.
        :param loadExpWT: boolean specifying whether wind-tunnel-related data-sets are to be loaded. If not provided, default is 'False'. If provided, then the name of the root directory containing the data-files must be provided in the form of a 'WTSets_root_dir' string kwarg.
        :param WTSets_root_dir: string specifying the root name of the directory containing the wind-tunnel-related data-sets.
        :param tdmsCoords: boolean specifying whether coordinate-related information is to be retrieved from the TDMS file.
        :param aveWTSets: boolean specifying whether the wind-tunnel-related data-sets are to be averaged. If not provided, default is 'False'. If provided, a 'protocol_WT' kwarg must be provided.
        :param protocol_WT: a 'expWTCnt.PROTOCOLS' object's instance specifying the protocol employed in the wind-tunnel-related data-sets being read.
        :param correctBlockWTSets: boolean specifying whether the wind-tunnel-related data-objects are to be blockage-corrected. If not provided, default is 'False'.
        :param computeUncsWTSets: boolean specifying whether the data in the wind-tunnel-related data-objects are to be uncertainty-computed. If not provided, default is 'False'.
        :param loadCFD: boolean specifying whether cfd-related data-sets are to be loaded. If not provided, default is 'False'. If provided, two parameters in the form of 'SimSets_root_dir' and 'protocol_cfd' must be provided.
        :param SimSets_root_dir: string specifying the root name of the directory containing the cfd-related data-sets.
        :param protocol_cfd: a 'cfdCnt.PROTOCOLS' object's instance specifying the protocol employed in the cfd-related data-sets being read.
        :param aveSimSets: boolean specifying whether the cfd-related data-sets are to be averaged.
        :param idx_start: integer specifying the starting index for the averaging. Employed if the provided protocol is any of the [cfdCnt.PROTOCOLS.fluentOut].
        :param idx_end: integer specifying the ending index for the averaging. Employed if the provided protocol is any of the [cfdCnt.PROTOCOLS.fluentOut].
        :param length: integer specifying the length of the data-file to be averaged. Employed if the provided protocol is any of the [cfdCnt.PROTOCOLS.fluentOut].
        :param save: boolean specifying whether the object is to be saved once processed by the constructor method. If not provided, default is 'False'.
        '''

        #--------------------------------------------------------------------------------------------------------------------#
        #-----------------------------------------------------ASSERTIONS-----------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        #####
        ##### kwargs-setting and asserting code-block.
        #####
        
        # Getting kwargs keys.
        kwkeys = kwargs.keys()        
        # Conditional for checking whether the 'load' argument has been passed as a kwarg. 'False' case.
        if 'load' not in kwkeys:
            # Asserting that the 'geometry' argument has been passed as a kwarg.
            assert 'ttype' in kwkeys, "Provide a 'ttype' named variable of the 'turbCnt.TYPES' class. Please amend."
            # Declaring the 'geometry' variable and setting its value.
            ttype = kwargs['ttype']
            omega = kwargs['omega'] if 'omega' in kwkeys else 3600
            N = kwargs['N'] if 'N' in kwkeys else 50
            rcas = kwargs['rcas'] if 'rcas' in kwkeys else 0.25
            hub_to_tip_ratio = kwargs['hub_to_tip_ratio'] if 'hub_to_tip_ratio' in kwkeys else 0.75
            Z = kwargs['Z'] if 'Z' in kwkeys else 7
            # Declaring the 'chord' variable and setting its value (etiher from kwarg or default).
            chord = kwargs['chord'] if 'chord' in kwkeys else 0.117
            angpitch = kwargs['angpitch'] if 'angpitch' in kwkeys else 0
            airfoil = kwargs['airfoil'] if 'airfoil' in kwkeys else geCnt.PROFILES.NACA0015
            tip_percent = kwargs['tip_percent'] if 'tip_percent' in kwkeys else 0.5
            p = kwargs['p'] if 'p' in kwkeys else 101325
            T = kwargs['T'] if 'T' in kwkeys else 288.15
            R = kwargs['R'] if 'R' in kwkeys else 287.058
            nu = kwargs['nu'] if 'nu' in kwkeys else 1.81e-5
            
        # 'True' case.
        else:
            # Asserting that the 'path' argument has been passed as a kwarg.
            assert 'path' in kwkeys, "Provide a 'path' named string variable specifying the path where the Pickle file from which the data is to be loaded is saved. Please amend."
            # Asserting that the 'name' argument has been passed as a kwarg.
            assert 'name' in kwkeys, "Provide a 'name' named string variable specifying the name of the Pickle file from which the data is to be loaded. Please amend."
            
        # Declaring the 'loadExpGeom' data and setting its value (either from kwargs or default).
        loadExpGeom = kwargs['loadExpGeom'] if 'loadExpGeom' in kwkeys else False
        # Conditional for checking whether experimentally-measured geometrical airfoil data to be loaded.
        if loadExpGeom:
            # Asserting that 'TridimSets_root_dir' has been passed as a kwarg.
            assert 'TridimSets_root_dir' in kwkeys, "A 'TridimSets_root_dir' string kwarg must be provided. Please amend."
            # Setting its value to the 'TridimSets_root_dir' variable.
            TridimSets_root_dir = kwargs['TridimSets_root_dir']
            # Deleting 'TridimSets_root_dir' from kwargs for avoiding further code clashes.
            del kwargs['TridimSets_root_dir']
        
        # Declaring the 'loadExpWT' data and setting its value (either from kwargs or default).
        loadExpWT = kwargs['loadExpWT'] if 'loadExpWT' in kwkeys else False
        # Conditional for checking whether experimentally-measured wind-tunnel-related airfoil data is to be loaded.
        if loadExpWT:
            # Asserting that 'WTSets_root_dir' has been passed as a kwarg.
            assert 'WTSets_root_dir' in kwkeys, "A 'WTSets_root_dir' string kwarg must be provided. Please amend."
            # Setting its value to the 'WTSets_root_dir' variable.
            WTSets_root_dir = kwargs['WTSets_root_dir']
            # Deleting 'WTSets_root_dir' from kwargs for avoiding further code clashes.
            del kwargs['WTSets_root_dir']
            # Asserting that 'protocol_WT' has been passed as a kwarg.
            assert 'protocol_WT' in kwkeys, "A 'protocol_WT' 'expWTCnt.PROTOCOLS' kwarg must be provided. Please amend."
            # Setting its value to the 'protocol_WT' variable.
            protocol_WT = kwargs['protocol_WT']
            # Deleting 'protocol_WT' from kwargs for avoiding further code clashes.
            del kwargs['protocol_WT']
         
        # Declaring the 'loadCFD' data and setting its value (either from kwargs or default).
        loadCFD = kwargs['loadCFD'] if 'loadCFD' in kwkeys else False
        # Conditional for checking whether cfd-related airfoil data is to be loaded.
        if loadCFD:
            # Asserting that 'SimSets_root_dir' has been passed as a kwarg.
            assert 'SimSets_root_dir' in kwkeys, "A 'SimSets_root_dir' string kwarg must be provided. Please amend."
            # Setting its value to the 'SimSets_root_dir' variable.
            SimSets_root_dir = kwargs['SimSets_root_dir']
            # Deleting 'SimSets_root_dir' from kwargs for avoiding further code clashes.
            del kwargs['SimSets_root_dir']
            # Asserting that 'protocol_cfd' has been passed as a kwarg.
            assert 'protocol_cfd' in kwkeys, "A 'protocol_cfd' 'cfdCnt.PROTOCOLS' kwarg must be provided. Please amend."
            # Setting its value to the 'protocol_cfd' variable.
            protocol_cfd = kwargs['protocol_cfd']
            # Deleting 'protocol_cfd' from kwargs for avoiding further code clashes.
            del kwargs['protocol_cfd']
                        
        # Declaring the 'save' variable and setting its value (etiher from kwarg or default).
        save = kwargs['save'] if 'save' in kwkeys else False
        
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------BODY--------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        #####
        ##### Attribute-setting and data-computing code-block.
        #####        
        
        # Conditional checking whether the object is to be instantiated by loading from a previously saved pickle file.     
        if 'load' not in kwargs.keys():
            # Instantiating the 'turbine' attribute of the newly created 'turbine' object with the provided input parameters.
            self.turbine = turb()
            self.turbine.create_or_add_stage(ttype=ttype,
            omega=omega, N=N, rcas=rcas, hub_to_tip_ratio=hub_to_tip_ratio, chord=chord,
            angpitch=angpitch, airfoil=airfoil, tip_percent=tip_percent, Z=Z,
            p=p, T=T, R=R, nu=nu)
            # Instantiating the 'name' attribute of the newly created 'airfoil' object with the provided input parameters.
            self.name = kwargs['name'] if 'name' in kwkeys else 'turbine'
            # Instantiating the 'path' attribute of the newly created 'airfoil' object with the provided input parameters.
            self.path = kwargs['path'] if 'path' in kwkeys else ''
            # Conditional for checking whether experimentally-measured geometrical airfoil data is to be loaded.
            if loadExpGeom:
                # Calling the 'loadExpGeomDataset()' method.
                self.loadExpGeomDataset(TridimSets_root_dir=TridimSets_root_dir, **kwargs)
            # Conditional for checking whether experimentally-measured wind-tunnel-related airfoil data is to be loaded.
            if loadExpWT:
                # Setting the kwargs 'protocol' to the previously declared 'protocol_WT' variable.
                kwargs['protocol'] = protocol_WT
                # Calling the 'loadExpWindDataset()' method.
                self.loadExpWindDataset(WTSets_root_dir=WTSets_root_dir, **kwargs)
            # Conditional for checking whether cfd-related airfoil data is to be loaded.
            if loadCFD:
                # Setting the kwargs 'protocol' to the previously declared 'protocol_cfd' variable.
                kwargs['protocol'] = protocol_cfd
                # Calling the 'loadCFDDataset()' method.
                self.loadCFDDataset(SimSets_root_dir=SimSets_root_dir, **kwargs)
        
        #####
        ##### Object-saving code-block.
        #####            
        
        # Conditional for checking whether the instantiated airfoil object is to be saved.
        if save:
            # Calling the 'save()' method.
            self.save(**kwargs)
            
        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------------RETURN-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#                
        
        # Return statement.
        return        
    
    #--------------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------__new__() METHOD------------------------------------------------#       
    #--------------------------------------------------------------------------------------------------------------------#     
    def __new__(cls, **kwargs):
        '''New constructor of the airfoil class. Used whenever a previously saved pickle file is to be loaded.
        
        **kwargs**:
        :param load: boolean specifying whether the instance of the object is to be loaded from a previously saved pickle file. If not provided, default is 'False'.
        :param path: string specifying the path where the pickle file that stores the object to be loaded is saved. If not provided, default is an empty string ("").
        :param name: string specifying the name fo the pickle file that stores the object to be loaded. If not provided, default is 'airfoil'.
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------BODY--------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        # Getting kwargs keys.
        kwkeys = kwargs.keys()          
        # Declaring the 'load' variable and setting its value (either kwarg or default).
        load = kwargs['load'] if 'load' in kwkeys else False
        
        # Conditional for checking whether a loading operation from a previously saved Pickle file is required. 'True' case.
        if load:
            # Declaring the 'path' variable and setting its value (either kwarg or default).
            path = kwargs['path'] if 'path' in kwkeys else ''
            # Declaring the 'name' variable and setting its value (either kwarg or default).
            name = kwargs['name'] if 'name' in kwkeys else 'turbine'
            # Load the previously saved pickle file.
            with open("/".join([path, name]) + ".pickle", 'rb') as file:
                obj = pickle.load(file)            
                
            #--------------------------------------------------------------------------------------------------------------------#
            #-------------------------------------------------------RETURN-------------------------------------------------------#             
            #--------------------------------------------------------------------------------------------------------------------#                
            
            # Ensuring that the loaded object is an instance of the class being instantiated. 'True' case.
            if isinstance(obj, cls):
                # Return statement.
                return obj
            # 'False' case.
            else:
                # Raising error.
                raise TypeError(f"Pickled object is not of type {cls.__name__}")
                
        #'False case'.
        else:
            # Calling the default '__new__()' method to create a new instance.
            return super().__new__(cls)
        
    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------computeAnlyDataset() METHOD------------------------------------------#       
    #--------------------------------------------------------------------------------------------------------------------#        
    def computeAnlyDataset(self, phi_q_vs, create_or_add, idx_anly: int=None, idx_dataset: int=None, **kwargs) -> None:
        '''Computes the power outcome of a turbine by means of the BEM-based analytical tool.
        
        **parameters**:
        :param phi_q_vs: array of input flow-coefficients, flow-rates or velocities for which to compute the turbine outcome.
        :param idx_anly: integer specifying the index by which to name the analytical dataset to be computed. Default is 'None', which means that the code counts the instances of the 'anlyData' objects created in the data-structure and assigns the next integer resulting from such a count to the name of the object.
        :param idx_dataset: integer specifying the index by which to name the dataset to be computed. Default is 'None', which means that the code counts the instances of the 'anlyDataSet' objects created in the data-structure and assigns the next integer resulting from such a count to the name of the object.
        
        **kwargs**:
        :param inputparam: string specifying whether the input parameter 'phi_q_vs' represents a flow-coefficient ('phi') a flow-rate ('q') or an axial velocity value ('vx'). If not provided, default is 'vx'.
        :param radFor: string specifying the radial formulation employed in the BEM-based code for solving the flow-field downstream the turbine. It may be either 'radEq' (for imposing a radial equilibrium condition on the pressure downstream) or 'Bessel' (for solving the differential equation relating the upstream and downstream flows by means of the Bessel-equation-based formulation). If not provided, default is 'radEq'.
        :param mode: string specifying solving mode employed by the BEM-based code. It may be either '3DAD' (solving it by means of the actuator-disk approach) or '3DNOAD' (for solving it via the more simple formulation. If not provided, default is '3DAD'.
        :param omega: float specifying the rotational speed at which the turbine is operating. If not provided, default is 3000*np.pi/30, which means that the turbine is operating at a rotational speed of 3000 rpm.        
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------BODY--------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#        
        
        #####
        ##### kwargs-setting and asserting code-block.
        #####
        
        # Getting kwargs keys.
        kwkeys = kwargs.keys()
        
        # Setting the 'inputparam' variable according to the input parameter.
        inputparam = ['vx' for _ in phi_q_vs] if 'inputparam' not in kwkeys else kwargs['inputparam']
        # Setting the 'radFor' variable according to the input parameter.
        radFor = 'radEq' if 'radFor' not in kwkeys else kwargs['radFor']
        # Setting the 'mode' variable according to the input parameter.
        mode = '3DAD' if 'mode' not in kwkeys else kwargs['mode']        
        # Setting the 'omega' variable according to the input parameter.
        omega = [3000*np.pi/30 for _ in phi_q_vs] if 'omega' not in kwkeys else kwargs['omega']            
            
        #####
        ##### Data-structure instantiation code-block.
        #####                    
        
        # Conditional for checking whether it is necessary to create a new 'anlyData' instance or add the dataset to an already created one. Case 'create'.
        if create_or_add == 'create':
            
            anlyDatset = anlyDataset()

            # Conditional for checking whether the passed 'idx_anly' argument is 'None'. Case 'True'.
            if idx_anly is None:
                
                # Declaring the 'count' variable and setting it to 0. It will store the number of 'anlyData' instances that exist in the data-structure from which the method is being called.
                count = 0
                
                # Conditional for checking whether 'anlyDataset' instances exist in the current 'airfoil' object.
                if any([isinstance(getattr(self, _), anlyDataset) for _ in dir(self)]):
                    # If so, counting the number of such instances.
                    count = len([_ for _ in dir(self) if "anlyData" in _ and isinstance(getattr(self, _), anlyDataset)])        
                # Setting a newly instantiated 'anlyDataset' object, whose name is based on the counting performed previously.
                setattr(self, "anlyData" + str(count + 1), anlyDatset)
            # Case 'False'.
            else:
                # Setting a newly instantiated 'anlyDataset' object, whose name is based on the passed 'idx' argument.                
                setattr(self, "anlyData" + str(idx_anly), anlyDatset)
                
        # Case 'add'.        
        elif create_or_add == 'add':
            
            # Conditional for checking whether the passed 'idx_anly' argument is 'None'. Case 'True'.
            if idx_anly is None:
                
                # Declaring the 'count' variable and setting it to 0. It will store the number of 'anlyData' instances that exist in the data-structure from which the method is being called.
                count = 0
                
                # Conditional for checking whether 'anlyDataset' instances exist in the current 'airfoil' object.
                if any([isinstance(getattr(self, _), anlyDataset) for _ in dir(self)]):
                    # If so, counting the number of such instances.
                    count = len([_ for _ in dir(self) if "anlyData" in _ and isinstance(getattr(self, _), anlyDataset)])
                    # Getting the corresponding 'anlyDataset' object.
                    anlyDatset = getattr(self, "anlyData" + str(count + 1))
            
            # Case 'false'.
            else:
                # Getting the corresponding 'anlyDataset' object.
                anlyDatset = getattr(self, "anlyData" + str(idx_anly))            
            
        #####
        ##### Analytical computation code-block.
        #####            
            
        # 'for loop running over the 'phi_q_vs' values passed as input arguments.
        for idx_phi_q_vs_, phi_q_vs_ in enumerate(phi_q_vs):
            
            # Instantiating an object of the 'anlyData' type.
            anlyDat = anlyData()
            # Setting an 'Ave'-named basePPClass-based data-structure within the newly created 'anlyDat' object.
            setattr(anlyDat, "Ave", basePPClass())
            # Getting the newly instantiated 'Ave'-named data-sturcture.
            aveDat = getattr(anlyDat, "Ave")
            
            # Setting the rotational speed of the turbine to the corresponding value passed as input argument.
            for turbstage in [getattr(self.turbine, _) for _ in dir(self.turbine) if "turbstage" in _]:
                turbstage.omega = omega
            
            # Calling the internal 'BEM()' method of the turbine object.
            self.turbine.BEM(phi_q_vs=phi_q_vs_,
                             inputparam=inputparam[idx_phi_q_vs_],
                             mode=mode,
                             radFor=radFor,
                             reset=True)    
            
            #####
            ##### Data dumping code-block.
            #####                       
            
            # Conditional for checking whether the passed argument is 'vx' or 'q'. Case 'vx'.
            if inputparam[idx_phi_q_vs_] == 'vx':
                # Computing the flow-rate and setting it to the 'Q' variable of the 'Ave'-named data-structure.
                setattr(aveDat, "Q", np.array(phi_q_vs_*np.pi*(self.turbine.turbstage1.geom.r[-1]**2 - self.turbine.turbstage1.geom.r[0]**2)))
            # Case 'q'.
            elif inputparam[idx_phi_q_vs_] == 'q':
                # Setting the flow-rate to the 'Q' variable of the 'Ave'-named data-sturcture.
                setattr(aveDat, "Q", np.array(phi_q_vs_))
            # Setting the computed torque to the 'torque' variable of the 'Ave'-named data-structure.
            setattr(aveDat, "torque", np.array(self.turbine.torque))
            # Setting the computed static-to-total pressure-drop to the  'deltaP_st' variable of the 'Ave'-named data-structure.
            setattr(aveDat, "deltaP_st", np.array(self.turbine.pst))
            # Setting the computed total-to-total pressure-drop to the 'deltaP_tt' variable of the 'Ave'-named data-structure.
            setattr(aveDat, "deltaP_tt", np.array(self.turbine.ptt))
            # Setting the computed flow-coefficient to the 'phi' variable of the 'Ave'-named data-structure.
            setattr(aveDat, "phi", np.array(self.turbine.Phi))
            # Setting the computed power-coefficient to the 'pi' variable of the 'Ave'-named data-structure.
            setattr(aveDat, "pi", np.array(self.turbine.Pi))
            # Setting the computed pressure-drop coefficient to the 'psi' variable of the 'Ave'-named data-structure.
            setattr(aveDat, "psi", np.array(self.turbine.Psi))
            # Setting the computed efficiency to the 'eff' variable of the 'Ave'-named data-structure.
            setattr(aveDat, "eff", np.array(self.turbine.Eff))
            # Damping the computed data to the corresponding data-set of the 'anlyDataset' object.
            setattr(anlyDatset, "Set" + str(idx_dataset), anlyDat)
        
    #--------------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------loadExpGeomDataset() METHOD-------------------------------------------#       
    #--------------------------------------------------------------------------------------------------------------------#        
    def loadExpGeomDataset(self, TridimSets_root_dir, **kwargs) -> None:
        '''It loads geometry-related data-sets into the current turbine object.
        
        **parameters**:
        :param TridimSets_root_dir: root directory path storing the tridim-related data-sets to be loaded.        
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------BODY--------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#        
        
        # Declaring the 'count' variable and setting it to 0. It will store the number of 'expGeomData' instances that exist in the 'airfoil' object from which the method is being called.
        count = 0
        # Conditional for checking whether 'expGeomData' instances exist in the current 'airfoil' object.        
        if any([isinstance(getattr(self, _), expGeCl.expGeomData) for _ in dir(self)]):
            # If so, counting the number of such instances.
            count = len([_ for _ in dir(self) if "expGeomData" in _ and isinstance(getattr(self, _), expGeCl.expGeomData)])        
        # Setting a newly instantiated 'expGeomData' object, whose name is based on the counting performed previously.
        setattr(self, "expGeomData" + str(count + 1), expGeCl.expGeomData(TridimSets_root_dir="/".join([self.path, TridimSets_root_dir]), readTridimSets=True))
        
        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------------RETURN-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#                
        
        # Return statement.
        return
    
    #--------------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------loadExpWindDataset() METHOD-------------------------------------------#       
    #--------------------------------------------------------------------------------------------------------------------#        
    def loadExpWindDataset(self, WTSets_root_dir, **kwargs) -> None:
        '''It loads wind-tunnel-related data-sets into the current turbine object.
        
        **parameters**:
        :param WTSets_root_dir: root directory path storing the wind-tunnel-related data-sets to be loaded.
        
        **kwargs**:
        :param aveWTSets: boolean specifying whether the data in the 'WTDataSet' objects is to be averaged. If not provided, default is 'False'.
        :param protocol: an instance of the expWTCnt.PROTOCOLS enum class specifying the type of test-protocol applied.
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------BODY--------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        #####
        ##### Kwargs-setting and assertion code-block.
        #####
        
        # Getting kwargs keys.
        kwkeys = kwargs.keys()
        # Declaring the 'aveWTSets' variable and setting its value (either from kwargs or default).
        aveWTSets = kwargs['aveWTSets'] if 'aveWTSets' in kwkeys else False
        # Conditional for checking whether an averaging is to be performed upon the read data.
        if aveWTSets:
            # Asserting that 'protocol' has been passed as kwargs.
            assert 'protocol' in kwkeys, 'Please provide a "protocol"-named kwarg specifying the testing protocol employed for the data-files being read prior to performing the averaging.'
        # Checking whether 'coord_order' has been passed as a kwarg, and is 'None'. Otherwise, setting it to 'None'.
        if "coord_order" not in kwkeys:
            kwargs["coord_order"] = None
        elif "coord_order" in kwkeys and kwargs["coord_order"] is not None:
            kwargs["coord_order"] = None
        # Checking whether 'tdmsCoords' has been passed as a kwarg. Otherwise, setting it to 'False'.
        if "tdmsCoords" not in kwkeys:
            kwargs["tdmsCoords"] = False
            
        #####
        ##### Data-reading and operation code-block.
        #####            
        
        # Declaring the 'count' variable and setting it to 0. It will store the number of 'expWTData' instances that exist in the 'airfoil' object from which the method is being called.
        count = 0
        if any([isinstance(getattr(self, _), expWTCl.expWTData) for _ in dir(self)]):
            # If so, counting the number of such instances.
            count = len([_ for _ in dir(self) if "expWTData" in _ and isinstance(getattr(self, _), expWTCl.expWTData)])
            
        # Conditional for checking whether the instantiation of the 'expWTData' object to be created required an averaging of the data. Case 'True'.
        if aveWTSets:
            # Setting a newly instantiated 'expWTData' object, whose name is based on the counting performed previously, and the averaging process specifications are passed as kwargs.
            setattr(self, "expWTData" + str(count + 1), expWTCl.expWTData(WTSets_root_dir="/".join([self.path, WTSets_root_dir]), readWTSets=True, **kwargs))
        # 'False' case.
        else:
            # Setting a newly instantiated 'expWTData' object, whose name is based on the counting performed previously.
            setattr(self, "expWTData" + str(count + 1), expWTCl.expWTData(WTSets_root_dir="/".join([self.path, WTSets_root_dir]), readWTSets=True))
        
        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------------RETURN-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#                
        
        # Return statement.
        return
    
    #--------------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------loadExpWindDataset() METHOD-------------------------------------------#       
    #--------------------------------------------------------------------------------------------------------------------#        
    def loadCFDDataset(self, SimSets_root_dir, **kwargs) -> None:
        '''It loads CFD-related data-sets into the current turbine object.
        
        **parameters**:
        :param SimSets_root_dir: root directory path storing the cfd-related data-sets to be loaded.
        
        **kwargs**:
        :param aveSimSets: boolean specifying whether the data in the 'SimDataSet' objects is to be averaged. If not provided, default is 'False'.
        :param protocol: an instance of the cfdCnt.PROTOCOLS enum class specifying the type of test-protocol applied.
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------BODY--------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#
        
        #####
        ##### Kwargs-setting and assertion code-block.
        #####
        
        # Getting kwargs keys.
        kwkeys = kwargs.keys()        
        aveSimSets = kwargs['aveSimSets'] if 'aveSimSets' in kwkeys else False
        # Conditional for checking whether an averaging is to be performed upon the read data.
        if aveSimSets:
            # Asserting that 'protocol' has been passed as kwargs.
            assert 'protocol' in kwkeys, 'Please provide a "protocol"-named kwarg specifying the testing protocol employed for the data-files being read prior to performing the averaging.'
            # Declaring 'protocol' and setting it to the passed kwarg values.
            protocol = kwargs['protocol']
            # Deleting the 'protocol' variable from kwargs, so that the kwargs dictionary can be employed further without clashes in the method callings.
            del kwargs['protocol']            
            
        #####
        ##### Data-reading and operation code-block.
        #####            
        
        # Declaring the 'count' variable and setting it to 0. It will store the number of 'expWTData' instances that exist in the 'airfoil' object from which the method is being called.
        count = 0        
        if any([isinstance(getattr(self, _), cfdCl.cfdData) for _ in dir(self)]):
            # If so, counting the number of such instances.
            count = len([_ for _ in dir(self) if "cfdData" in _ and isinstance(getattr(self, _), cfdCl.cfdData)])
            
        # Conditional for checking whether the instantiation of the 'expWTData' object to be created required an averaging of the data. Case 'True'.
        if aveSimSets:
            # Setting a newly instantiated 'expWTData' object, whose name is based on the counting performed previously, and the averaging process specifications are passed as kwargs.
            setattr(self, "cfdData" + str(count + 1), cfdCl.cfdData(SimSets_root_dir="/".join([self.path, SimSets_root_dir]), readSimSets=True, protocol=protocol, **kwargs))
        # 'False' case.
        else:            
            # Setting a newly instantiated 'expWTData' object, whose name is based on the counting performed previously.
            setattr(self, "cfdData" + str(count + 1), cfdCl.cfdData(SimSets_root_dir="/".join([self.path, SimSets_root_dir]), readSimSets=True))
        
        #--------------------------------------------------------------------------------------------------------------------#
        #-------------------------------------------------------RETURN-------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#                
        
        # Return statement.
        return
    
    #--------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------save() METHOD--------------------------------------------------#       
    #--------------------------------------------------------------------------------------------------------------------# 
    def save(self, **kwargs) -> None:
        '''It saves the current instance of the object in a serialized Pickle file for further fast loading.
        
        **kwargs**:
        
        :param name: name of the file that will store the serialized Pickle object. If not provided, default is 'self.name'.
        :param path: path where the file storing the serialized Pickle object is to be saved. If not provided, default is 'self.path'.
        '''
        
        #--------------------------------------------------------------------------------------------------------------------#
        #--------------------------------------------------------BODY--------------------------------------------------------#             
        #--------------------------------------------------------------------------------------------------------------------#    
        
        # Getting kwargs keys.
        kwkeys = kwargs.keys()       
        
        # Dclaring the 'name' variable and setting its value (either kwarg or default).
        name = kwargs['name'] if 'name' in kwkeys else self.name
        # Dclaring the 'path' variable and setting its value (either kwarg or default).
        path = kwargs['path'] if 'path' in kwkeys else self.path
        # Dclaring the 'keep_raw' variable and setting its value (either kwarg or default).
        keep_raw = kwargs['keep_raw'] if 'keep_raw' in kwkeys else True
        
        # Dumping (saving) the current instance of the object into a serialized Pickle file.
        with open("/".join([path, name]) + ".pickle", "wb") as handle:
            if keep_raw:
                pickle.dump(self, handle)
            else:
                for idx_expWTDat, expWTDat in enumerate([_ for _ in dir(self) if isinstance(getattr(self, _), expWTCl.expWTData)]):
                    expWTDataObj = getattr(self, expWTDat)
                    for idx_WTDatSet, WTDatSet in enumerate([_ for _ in dir(expWTDataObj) if isinstance(getattr(expWTDataObj, _), expWTCl.WTDataSet)]):
                        WTDataSetObj = getattr(expWTDataObj, WTDatSet)
                        for idx_WTDat, WTDat in enumerate([_ for _ in dir(WTDataSetObj) if isinstance(getattr(WTDataSetObj, _), expWTCl.WTData)]):
                            WTDataObj = getattr(WTDataSetObj, WTDat)
                            if hasattr(WTDataObj, "rawData"):
                                delattr(WTDataObj, "rawData")
                pickle.dump(self, handle)