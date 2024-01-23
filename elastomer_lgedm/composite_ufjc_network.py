# Import necessary libraries
from __future__ import division
from dolfin import *
from .composite_ufjc import RateIndependentScissionCompositeuFJCUFLFEniCS
from .utility import none_str2nonetype
import numpy as np


class CompositeuFJCNetwork(object):
    
    def __init__(self, material_parameters):

        mp = material_parameters
        
        # Extract chain segment polydispersity information
        self.nu_num = mp["nu_num"]
        nu_list = []
        P_nu_list = []
        
        for nu_indx in range(self.nu_num):
            nu_val = mp["nu_indx_"+str(nu_indx)+"_nu_val"]
            P_nu_val = self.P_nu(mp, nu_val)
            nu_list.append(nu_val)
            P_nu_list.append(P_nu_val)
        
        P_nu_sum = np.sum(P_nu_list)
        
        # Extract fundamental chain composition parameters and chain 
        # scission parameters
        nu_b = none_str2nonetype(mp["nu_b"])
        zeta_b_char = none_str2nonetype(mp["zeta_b_char"])
        kappa_b = none_str2nonetype(mp["kappa_b"])
        zeta_nu_char = none_str2nonetype(mp["zeta_nu_char"])
        kappa_nu = none_str2nonetype(mp["kappa_nu"])
        
        composite_ufjc_ufl_fenics_list = [
            RateIndependentScissionCompositeuFJCUFLFEniCS(nu=nu_list[nu_indx],
                                                nu_b=nu_b,
                                                zeta_b_char=zeta_b_char,
                                                kappa_b=kappa_b,
                                                zeta_nu_char=zeta_nu_char,
                                                kappa_nu=kappa_nu)
            for nu_indx in range(self.nu_num)
        ]
        
        nu_list = [
            Constant(nu_list[nu_indx]) for nu_indx in range(self.nu_num)
        ]
        A_nu_list = [
            composite_ufjc_ufl_fenics_list[nu_indx].A_nu
            for nu_indx in range(self.nu_num)
        ]
        Lambda_nu_ref_list = [
            composite_ufjc_ufl_fenics_list[nu_indx].Lambda_nu_ref
            for nu_indx in range(self.nu_num)
        ]
        P_nu_list = [
            Constant(P_nu_list[nu_indx]) for nu_indx in range(self.nu_num)
        ]
        P_nu_sum = Constant(P_nu_sum)
        
        self.nu_list = nu_list
        self.A_nu_list = A_nu_list
        self.Lambda_nu_ref_list = Lambda_nu_ref_list
        self.P_nu_list = P_nu_list
        self.P_nu_sum = P_nu_sum
        
        # Retain specified parameters
        self.composite_ufjc_ufl_fenics_list = composite_ufjc_ufl_fenics_list

        self.zeta_nu_char = composite_ufjc_ufl_fenics_list[0].zeta_nu_char
        self.kappa_nu = composite_ufjc_ufl_fenics_list[0].kappa_nu
        self.lmbda_nu_ref = composite_ufjc_ufl_fenics_list[0].lmbda_nu_ref
        self.lmbda_c_eq_ref = composite_ufjc_ufl_fenics_list[0].lmbda_c_eq_ref
        self.lmbda_nu_crit = composite_ufjc_ufl_fenics_list[0].lmbda_nu_crit
        self.lmbda_c_eq_crit = composite_ufjc_ufl_fenics_list[0].lmbda_c_eq_crit
        self.xi_c_crit = composite_ufjc_ufl_fenics_list[0].xi_c_crit
        self.lmbda_nu_pade2berg_crit = (
            composite_ufjc_ufl_fenics_list[0].lmbda_nu_pade2berg_crit
        )
        self.lmbda_c_eq_pade2berg_crit = (
            composite_ufjc_ufl_fenics_list[0].lmbda_c_eq_pade2berg_crit
        )
    
    def P_nu(self, material_parameters, nu):
        
        mp = material_parameters

        if mp["nu_distribution"] == "itskov":
            return (
                (1/(mp["Delta_nu"]+1))
                * (1+(1/mp["Delta_nu"]))**(mp["nu_min"]-nu)
            )
        if mp["nu_distribution"] == "uniform":
            return 1. / self.nu_num