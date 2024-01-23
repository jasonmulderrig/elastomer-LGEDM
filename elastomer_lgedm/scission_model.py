"""The module for the composite uFJC scission model implemented in the
Unified Form Language (UFL) for FEniCS specifying the fundamental
analytical scission model.
"""

# Import external modules
from __future__ import division
from dolfin import *

# Import internal modules
from .core import CompositeuFJCUFLFEniCS


class AnalyticalScissionCompositeuFJCUFLFEniCS(CompositeuFJCUFLFEniCS):
    """The composite uFJC scission model class implemented in the
    Unified Form Language (UFL) for FEniCS specifying the fundamental
    analytical scission model.

    This class contains methods specifying the fundamental scission 
    model implemented in the Unified Form Language (UFL) for FEniCS,
    which involve defining both energetic and probabilistic quantities.
    It inherits all attributes and methods from the
    ``CompositeuFJCUFLFEniCS`` class.
    """
    def __init__(self):
        """
        Initializes the ``AnalyticalScissionCompositeuFJCUFLFEniCS``
        class.
        
        Initialize and inherit all attributes and methods from the
        ``CompositeuFJCUFLFEniCS`` class instance.
        """
        CompositeuFJCUFLFEniCS.__init__(self)
    
    def epsilon_nu_sci_hat_ufl_fenics_func(self, lmbda_nu_hat, lmbda_c_eq_hat):
        """Nondimensional segment scission energy.
        
        This function computes the nondimensional segment scission
        energy as a function of the applied segment stretch and the
        applied equilibrium chain stretch. This function is implemented
        in the Unified Form Language (UFL) for FEniCS.
        """
        return (
            self.psi_cnu_ufl_fenics_func(lmbda_nu_hat, lmbda_c_eq_hat)
            + self.zeta_nu_char
        )
    
    def epsilon_cnu_sci_hat_ufl_fenics_func(self, lmbda_nu_hat, lmbda_c_eq_hat):
        """Nondimensional chain scission energy per segment.
        
        This function computes the nondimensional chain scission energy
        per segment as a function of the applied segment stretch and the
        applied equilibrium chain stretch. This function is implemented
        in the Unified Form Language (UFL) for FEniCS.
        """
        return (
            self.epsilon_nu_sci_hat_ufl_fenics_func(lmbda_nu_hat, lmbda_c_eq_hat)
        )

    def u_nu_tot_hat_ufl_fenics_func(
            self, lmbda_nu_hat, lmbda_nu, lmbda_c_eq_hat):
        """Nondimensional total segment potential under an applied chain
        force.
        
        This function computes the nondimensional total segment
        potential under an applied chain force as a function of the
        applied segment stretch, the segment stretch specifying a
        particular state in the energy landscape, and the applied
        equilibrium chain stretch. This function is implemented in the
        Unified Form Language (UFL) for FEniCS.
        """
        return (
            self.u_nu_ufl_fenics_func(lmbda_nu)
            - lmbda_nu * self.xi_c_ufl_fenics_func(lmbda_nu_hat, lmbda_c_eq_hat)
        )
    
    def u_nu_hat_ufl_fenics_func(self, lmbda_nu_hat, lmbda_nu, lmbda_c_eq_hat):
        """Nondimensional total distorted segment potential under an
        applied chain force.
        
        This function computes the nondimensional total distorted
        segment potential under an applied chain force as a function
        of the applied segment stretch, the segment stretch
        specifying a particular state in the energy landscape, and the
        applied equilibrium chain stretch. This function is implemented
        in the Unified Form Language (UFL) for FEniCS.
        """
        return (
            self.u_nu_ufl_fenics_func(lmbda_nu)
            - (lmbda_nu-lmbda_nu_hat)
            * self.xi_c_ufl_fenics_func(lmbda_nu_hat, lmbda_c_eq_hat)
        )
    
    def lmbda_nu_locmin_hat_ufl_fenics_func(self, lmbda_nu_hat, lmbda_c_eq_hat):
        """Segment stretch corresponding to the local minimum of the
        nondimensional total (distorted) segment potential under an 
        applied chain force.
        
        This function computes the segment stretch corresponding to the
        local minimum of the nondimensional total (distorted) segment
        potential under an applied chain force as a function of the
        applied segment stretch and the applied equilibrium chain
        stretch. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        return (
            1.
            + self.xi_c_ufl_fenics_func(lmbda_nu_hat, lmbda_c_eq_hat) / self.kappa_nu
        )
    
    def lmbda_nu_locmax_hat_ufl_fenics_func(self, lmbda_nu_hat, lmbda_c_eq_hat):
        """Segment stretch corresponding to the local maximum of the
        nondimensional total (distorted) segment potential under an 
        applied chain force.
        
        This function computes the segment stretch corresponding to the
        local maximum of the nondimensional total (distorted) segment
        potential under an applied chain force as a function of the
        applied segment stretch and the applied equilibrium chain
        stretch. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        lmbda_nu_hat_cond = conditional(ge(lmbda_nu_hat, 1.+DOLFIN_EPS),
                                        lmbda_nu_hat, 1.+DOLFIN_EPS)
        lmbda_c_eq_hat_cond = conditional(ge(lmbda_c_eq_hat, 1.+DOLFIN_EPS),
                                          lmbda_c_eq_hat, 1.+DOLFIN_EPS)
        cbrt_arg_nmrtr = self.zeta_nu_char**2
        cbrt_arg_dnmntr = (
                self.kappa_nu
                * self.xi_c_ufl_fenics_func(lmbda_nu_hat_cond, lmbda_c_eq_hat_cond)
            )
        cbrt_arg = cbrt_arg_nmrtr / cbrt_arg_dnmntr
        lmbda_nu_locmax_hat_val = 1. + cbrt_arg**(1./3.)
        return (
            conditional(ge(lmbda_nu_hat, 1.+DOLFIN_EPS),
                        lmbda_nu_locmax_hat_val, 1./DOLFIN_EPS)
        )
    
    def e_nu_sci_hat_theoretical_ufl_fenics_func(
            self, lmbda_nu_hat, lmbda_c_eq_hat):
        """Theoretical form of the nondimensional segment scission
        activation energy barrier.
        
        This function computes the nondimensional segment scission
        activation energy barrier as a function of the applied segment
        stretch and the applied equilibrium chain stretch. This function
        is implemented in the Unified Form Language (UFL) for FEniCS.
        """
        lmbda_nu_locmin_hat = (
            self.lmbda_nu_locmin_hat_ufl_fenics_func(lmbda_nu_hat, lmbda_c_eq_hat)
        )
        lmbda_nu_locmax_hat = (
            self.lmbda_nu_locmax_hat_ufl_fenics_func(lmbda_nu_hat, lmbda_c_eq_hat)
        )
        e_nu_sci_hat_val_i = (
            self.u_nu_hat_ufl_fenics_func(lmbda_nu_hat, lmbda_nu_locmax_hat, lmbda_c_eq_hat)
            - self.u_nu_hat_ufl_fenics_func(lmbda_nu_hat, lmbda_nu_locmin_hat, lmbda_c_eq_hat)
        )
        e_nu_sci_hat_val_ii = conditional(gt(lmbda_nu_hat, 1.),
                                          e_nu_sci_hat_val_i, self.zeta_nu_char)
        e_nu_sci_hat_val = conditional(gt(lmbda_nu_hat, self.lmbda_nu_crit),
                                       0., e_nu_sci_hat_val_ii)
        return e_nu_sci_hat_val
    
    def e_nu_sci_hat_ufl_fenics_func(self, lmbda_nu_hat):
        """Nondimensional segment scission activation energy barrier.
        
        This function computes the nondimensional segment scission
        activation energy barrier as a function of the applied segment
        stretch. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        cbrt_arg = (
            self.zeta_nu_char**2 * self.kappa_nu * (lmbda_nu_hat-1.)**2
        )
        e_nu_sci_hat_val_i = (
            0.5 * self.kappa_nu * (lmbda_nu_hat-1.)**2
            - 1.5 * cbrt_arg**(1./3.) + self.zeta_nu_char
        )
        e_nu_sci_hat_val_ii = conditional(gt(lmbda_nu_hat, 1.+self.cond_val),
                                          e_nu_sci_hat_val_i, self.zeta_nu_char)
        e_nu_sci_hat_val = conditional(gt(lmbda_nu_hat, self.lmbda_nu_crit),
                                       0., e_nu_sci_hat_val_ii)
        return e_nu_sci_hat_val
    
    def p_nu_sci_hat_ufl_fenics_func(self, lmbda_nu_hat):
        """Rate-independent probability of segment scission.
        
        This function computes the rate-independent probability of
        segment scission as a function of the applied segment stretch.
        This function is implemented in the Unified Form Language (UFL)
        for FEniCS.
        """
        return exp(-self.e_nu_sci_hat_ufl_fenics_func(lmbda_nu_hat))

    def p_nu_sur_hat_ufl_fenics_func(self, lmbda_nu_hat):
        """Rate-independent probability of segment survival.
        
        This function computes the rate-independent probability of
        segment survival as a function of the applied segment stretch.
        This function is implemented in the Unified Form Language (UFL)
        for FEniCS.
        """
        return 1. - self.p_nu_sci_hat_ufl_fenics_func(lmbda_nu_hat)
    
    def p_c_sur_hat_ufl_fenics_func(self, lmbda_nu_hat):
        """Rate-independent probability of chain survival.
        
        This function computes the rate-independent probability of chain
        survival as a function of the applied segment stretch. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        return self.p_nu_sur_hat_ufl_fenics_func(lmbda_nu_hat)**self.nu
    
    def p_c_sci_hat_ufl_fenics_func(self, lmbda_nu_hat):
        """Rate-independent probability of chain scission.
        
        This function computes the rate-independent probability of chain
        scission as a function of the applied segment stretch. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        return 1. - self.p_c_sur_hat_ufl_fenics_func(lmbda_nu_hat)
    
    def lmbda_nu_p_nu_sci_hat_ufl_fenics_func(self, p_nu_sci_hat_val):
        """Segment stretch as a function of the rate-independent
        probability of segment scission.
        
        This function calculates the segment stretch as a function of
        the rate-independent probability of segment scission. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        cbrt_arg = (self.zeta_nu_char/self.kappa_nu)**2
        pi_tilde = -3. * cbrt_arg**(1./3.)

        rho_tilde = 2. / self.kappa_nu * (self.zeta_nu_char+ln(p_nu_sci_hat_val))

        acos_arg = 3. * rho_tilde / (2.*pi_tilde) * sqrt(-3./pi_tilde)
        acos_arg = conditional(ge(acos_arg, 1.-DOLFIN_EPS),
                               1.-DOLFIN_EPS, acos_arg)
        acos_arg = conditional(le(acos_arg, -1.+DOLFIN_EPS),
                               -1.+DOLFIN_EPS, acos_arg)
        
        cos_arg = 1. / 3. * acos(acos_arg) - 2. * DOLFIN_PI / 3.

        phi_tilde = 2. * sqrt(-pi_tilde/3.) * cos(cos_arg)

        return 1. + sqrt(phi_tilde**3)
    
    def lmbda_nu_p_c_sci_hat_ufl_fenics_func(self, p_c_sci_hat_val):
        """Segment stretch as a function of the rate-independent
        probability of chain scission.
        
        This function calculates the segment stretch as a function of
        the rate-independent probability of chain scission. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        p_nu_sci_hat_val = 1. - (1.-p_c_sci_hat_val)**(1./self.nu)

        return (
            self.lmbda_nu_p_nu_sci_hat_ufl_fenics_func(p_nu_sci_hat_val)
        )
    
    def e_nu_sci_hat_prime_ufl_fenics_func(self, lmbda_nu_hat):
        """Derivative of the nondimensional segment scission activation
        energy barrier taken with respect to the applied segment stretch.
        
        This function computes the derivative of the nondimensional
        segment scission activation energy barrier taken with respect to
        the applied segment stretch as a function of applied segment
        stretch. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        cbrt_arg_nmrtr = self.zeta_nu_char**2 * self.kappa_nu
        cbrt_arg_dnmntr = lmbda_nu_hat - 1.
        cbrt_arg_dnmntr = conditional(ge(cbrt_arg_dnmntr, DOLFIN_EPS),
                                      cbrt_arg_dnmntr, DOLFIN_EPS)
        cbrt_arg = cbrt_arg_nmrtr / cbrt_arg_dnmntr
        e_nu_sci_hat_prime_val_i = (
            self.kappa_nu * (lmbda_nu_hat-1.) - cbrt_arg**(1./3.)
        )
        e_nu_sci_hat_prime_val_ii = conditional(gt(lmbda_nu_hat, 1.+self.cond_val),
                                                e_nu_sci_hat_prime_val_i, -1./DOLFIN_EPS)
        e_nu_sci_hat_prime_val = conditional(gt(lmbda_nu_hat, self.lmbda_nu_crit),
                                             0., e_nu_sci_hat_prime_val_ii)
        return e_nu_sci_hat_prime_val

    def p_nu_sci_hat_prime_ufl_fenics_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of segment
        scission taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch as a function of applied segment
        stretch. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        p_nu_sci_hat_prime_val_i = (
            -self.p_nu_sci_hat_ufl_fenics_func(lmbda_nu_hat)
            * self.e_nu_sci_hat_prime_ufl_fenics_func(lmbda_nu_hat)
        )
        p_nu_sci_hat_prime_val = conditional(gt(lmbda_nu_hat, 1.+self.cond_val),
                                             p_nu_sci_hat_prime_val_i, 0.)
        return p_nu_sci_hat_prime_val
    
    def p_nu_sur_hat_prime_ufl_fenics_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of segment
        survival taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch as a function of applied segment
        stretch. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        p_nu_sur_hat_prime_val_i = (
            -self.p_nu_sci_hat_prime_ufl_fenics_func(lmbda_nu_hat)
        )
        p_nu_sur_hat_prime_val = conditional(gt(lmbda_nu_hat, 1.+self.cond_val),
                                             p_nu_sur_hat_prime_val_i, 0.)
        return p_nu_sur_hat_prime_val
    
    def p_c_sur_hat_prime_ufl_fenics_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of chain
        survival taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch as a function of applied segment stretch. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        p_c_sur_hat_prime_val_i = (
            -self.nu
            * (1.-self.p_nu_sci_hat_ufl_fenics_func(lmbda_nu_hat))**(self.nu-1)
            * self.p_nu_sci_hat_prime_ufl_fenics_func(lmbda_nu_hat)
        )
        p_c_sur_hat_prime_val = conditional(gt(lmbda_nu_hat, 1.+self.cond_val),
                                            p_c_sur_hat_prime_val_i, 0.)
        return p_c_sur_hat_prime_val

    def p_c_sci_hat_prime_ufl_fenics_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of chain
        scission taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch as a function of applied segment stretch. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        p_c_sci_hat_prime_val_i = (
            -self.p_c_sur_hat_prime_ufl_fenics_func(lmbda_nu_hat)
        )
        p_c_sci_hat_prime_val = conditional(gt(lmbda_nu_hat, 1.+self.cond_val),
                                            p_c_sci_hat_prime_val_i, 0.)
        return p_c_sci_hat_prime_val
    
    def epsilon_nu_sci_hat_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional segment scission
        energy function.
        
        This function computes the nondimensional segment scission
        energy as a function of the applied segment stretch. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        return (
            self.psi_cnu_analytical_ufl_fenics_func(lmbda_nu_hat)
            + self.zeta_nu_char
        )
    
    def epsilon_cnu_sci_hat_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional chain scission
        energy function per segment.
        
        This function computes the nondimensional chain scission
        energy per segment as a function of the applied segment stretch.
        This function is implemented in the Unified Form Language (UFL)
        for FEniCS.
        """
        return self.epsilon_nu_sci_hat_analytical_ufl_fenics_func(lmbda_nu_hat)
    
    def e_nu_sci_hat_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional segment scission
        activation energy barrier.
        
        This function computes the nondimensional segment scission
        activation energy barrier as a function of the applied segment
        stretch. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        return self.e_nu_sci_hat_ufl_fenics_func(lmbda_nu_hat)
    
    def p_nu_sci_hat_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        segment scission.
        
        This function computes the rate-independent probability of
        segment scission as a function of the applied segment stretch.
        This function is implemented in the Unified Form Language (UFL)
        for FEniCS.
        """
        return self.p_nu_sci_hat_ufl_fenics_func(lmbda_nu_hat)

    def p_nu_sur_hat_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        segment survival.
        
        This function computes the rate-independent probability of
        segment survival as a function of the applied segment stretch.
        This function is implemented in the Unified Form Language (UFL)
        for FEniCS.
        """
        return self.p_nu_sur_hat_ufl_fenics_func(lmbda_nu_hat)
    
    def p_c_sur_hat_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        chain survival.
        
        This function computes the rate-independent probability of
        chain survival as a function of the applied segment stretch.
        This function is implemented in the Unified Form Language (UFL)
        for FEniCS.
        """
        return self.p_c_sur_hat_ufl_fenics_func(lmbda_nu_hat)
    
    def p_c_sci_hat_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        chain scission.
        
        This function computes the rate-independent probability of
        chain scission as a function of the applied segment stretch.
        This function is implemented in the Unified Form Language (UFL)
        for FEniCS.
        """
        return self.p_c_sci_hat_ufl_fenics_func(lmbda_nu_hat)
    
    def lmbda_nu_p_nu_sci_hat_analytical_ufl_fenics_func(self, p_nu_sci_hat_val):
        """Analytical form of the segment stretch as a function of the
        rate-independent probability of segment scission.
        
        This function calculates the segment stretch as a function of
        the rate-independent probability of segment scission. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        return self.lmbda_nu_p_nu_sci_hat_ufl_fenics_func(p_nu_sci_hat_val)
    
    def lmbda_nu_p_c_sci_hat_analytical_ufl_fenics_func(self, p_c_sci_hat_val):
        """Analytical form of the segment stretch as a function of the
        rate-independent probability of chain scission.
        
        This function calculates the segment stretch as a function of
        the rate-independent probability of chain scission. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        return self.lmbda_nu_p_c_sci_hat_ufl_fenics_func(p_c_sci_hat_val)
    
    def e_nu_sci_hat_prime_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the nondimensional
        segment scission activation energy barrier taken with respect to
        the applied segment stretch.
        
        This function computes the derivative of the nondimensional
        segment scission activation energy barrier taken with respect to
        the applied segment stretch as a function of applied segment
        stretch. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        return self.e_nu_sci_hat_prime_ufl_fenics_func(lmbda_nu_hat)

    def p_nu_sci_hat_prime_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch as a function of applied segment
        stretch. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        return self.p_nu_sci_hat_prime_ufl_fenics_func(lmbda_nu_hat)
    
    def p_nu_sur_hat_prime_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch as a function of applied segment
        stretch. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        return self.p_nu_sur_hat_prime_ufl_fenics_func(lmbda_nu_hat)
    
    def p_c_sur_hat_prime_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch as a function of applied segment stretch. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        return self.p_c_sur_hat_prime_ufl_fenics_func(lmbda_nu_hat)

    def p_c_sci_hat_prime_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch as a function of applied segment stretch. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        return self.p_c_sci_hat_prime_ufl_fenics_func(lmbda_nu_hat)