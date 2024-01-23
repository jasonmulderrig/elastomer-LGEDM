"""The core single-chain module for the composite uFJC model implemented
in the Unified Form Language (UFL) for FEniCS.
"""

# Import external modules
from __future__ import division
from dolfin import *


class CompositeuFJCUFLFEniCS(object):
    """The composite uFJC single-chain model class implemented in the
    Unified Form Language (UFL) for FEniCS.
    
    This class contains methods specifying the core functions
    underpinning the composite uFJC single-chain model independent of
    scission implemented in the Unified Form Language (UFL) for FEniCS.
    """
    def __init__(self):
        pass
    
    def u_nu_har_ufl_fenics_func(self, lmbda_nu):
        """Nondimensional harmonic segment potential energy.
        
        This function computes the nondimensional harmonic segment 
        potential energy as a function of the segment stretch. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        return 0.5 * self.kappa_nu * (lmbda_nu-1.)**2 - self.zeta_nu_char
    
    def u_nu_subcrit_ufl_fenics_func(self, lmbda_nu):
        """Nondimensional sub-critical chain state segment potential
        energy.
        
        This function computes the nondimensional sub-critical chain
        state segment potential energy as a function of the segment
        stretch. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        return self.u_nu_har_ufl_fenics_func(lmbda_nu)
    
    def u_nu_supercrit_ufl_fenics_func(self, lmbda_nu):
        """Nondimensional super-critical chain state segment potential
        energy.
        
        This function computes the nondimensional super-critical chain
        state segment potential energy as a function of the segment
        stretch. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        nmrtr = -self.zeta_nu_char**2
        dnmntr = 2. * self.kappa_nu * (lmbda_nu-1.)**2
        dnmntr = conditional(ge(dnmntr, DOLFIN_EPS), dnmntr, DOLFIN_EPS)
        return nmrtr / dnmntr
    
    def u_nu_ufl_fenics_func(self, lmbda_nu):
        """Nondimensional composite uFJC segment potential energy.
        
        This function computes the nondimensional composite uFJC 
        segment potential energy as a function of the segment stretch.
        This function is implemented in the Unified Form Language (UFL)
        for FEniCS.
        """
        u_nu_subcrit_val = self.u_nu_subcrit_ufl_fenics_func(lmbda_nu)
        u_nu_supercrit_val = self.u_nu_supercrit_ufl_fenics_func(lmbda_nu)
        u_nu_val = conditional(le(lmbda_nu, self.lmbda_nu_crit),
                               u_nu_subcrit_val, u_nu_supercrit_val)
        return u_nu_val
    
    def u_nu_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional composite uFJC segment
        potential energy.
        
        This function computes the nondimensional composite uFJC segment
        potential energy as a function of the applied segment stretch.
        This function is implemented in the Unified Form Language (UFL)
        for FEniCS.
        """
        return self.u_nu_ufl_fenics_func(lmbda_nu_hat)
    
    def u_nu_prime_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the nondimensional
        composite uFJC segment potential energy taken with respect to
        the applied segment stretch.
        
        This function computes the derivative of the nondimensional
        composite uFJC segment potential energy taken with respect to
        the applied segment stretch as a function of applied segment
        stretch. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        u_nu_prime_subcrit_val = self.kappa_nu * (lmbda_nu_hat-1.)
        u_nu_prime_supercrit_nmrtr = self.zeta_nu_char**2
        u_nu_prime_supercrit_dnmntr = self.kappa_nu * (lmbda_nu_hat-1.)**3
        u_nu_prime_supercrit_dnmntr = (
            conditional(ge(u_nu_prime_supercrit_dnmntr, DOLFIN_EPS),
                        u_nu_prime_supercrit_dnmntr, DOLFIN_EPS)
        )
        u_nu_prime_supercrit_val = (
            u_nu_prime_supercrit_nmrtr / u_nu_prime_supercrit_dnmntr
        )
        u_nu_prime_val = conditional(le(lmbda_nu_hat, self.lmbda_nu_crit),
                                     u_nu_prime_subcrit_val,
                                     u_nu_prime_supercrit_val)
        return u_nu_prime_val
    
    def lmbda_c_eq_ufl_fenics_func(self, lmbda_nu):
        """Equilibrium chain stretch.
        
        This function computes the equilibrium chain stretch as a 
        function of the segment stretch. This function is implemented in
        the Unified Form Language (UFL) for FEniCS.
        """
        # sub-critical Pade approximant
        alpha_tilde_psb = 1.
        
        beta_tilde_psb = (self.kappa_nu+3.) * (1.-lmbda_nu)

        trm_i = 2. * self.kappa_nu + 3.
        trm_ii = 2.
        trm_iii = 2. * self.kappa_nu
        gamma_tilde_psb = trm_i * (lmbda_nu**2-trm_ii*lmbda_nu) + trm_iii

        trm_i = self.kappa_nu + 1.
        trm_ii = 3.
        trm_iii = 2.
        trm_iv = self.kappa_nu
        trm_v = 1.
        delta_tilde_psb = (
            trm_i * (trm_ii*lmbda_nu**2-lmbda_nu**3)
            - trm_iii * (trm_iv*lmbda_nu+trm_v)
        )

        pi_tilde_psb_nmrtr = (
            3. * alpha_tilde_psb * gamma_tilde_psb - beta_tilde_psb**2
        )
        pi_tilde_psb_dnmntr = 3. * alpha_tilde_psb**2
        pi_tilde_psb = pi_tilde_psb_nmrtr / pi_tilde_psb_dnmntr

        rho_tilde_psb_nmrtr = (
            2. * beta_tilde_psb**3
            - 9. * alpha_tilde_psb * beta_tilde_psb * gamma_tilde_psb 
            + 27. * alpha_tilde_psb**2 * delta_tilde_psb
        )
        rho_tilde_psb_dnmntr = 27. * alpha_tilde_psb**3
        rho_tilde_psb = rho_tilde_psb_nmrtr / rho_tilde_psb_dnmntr
        
        acos_arg_psb = (
            3. * rho_tilde_psb / (2.*pi_tilde_psb) * sqrt(-3./pi_tilde_psb)
        )
        acos_arg_psb = conditional(ge(acos_arg_psb, 1.-DOLFIN_EPS),
                                   1.-DOLFIN_EPS, acos_arg_psb)
        acos_arg_psb = conditional(le(acos_arg_psb, -1.+DOLFIN_EPS),
                                   -1.+DOLFIN_EPS, acos_arg_psb)
        
        cos_arg_psb = 1. / 3. * acos(acos_arg_psb) - 2. * DOLFIN_PI / 3.

        lmbda_c_eq_psb_val = (
            2. * sqrt(-pi_tilde_psb/3.) * cos(cos_arg_psb)
            - beta_tilde_psb / (3.*alpha_tilde_psb)
        )

        # sub-critical Bergstrom approximant
        bsb_dnmntr_trm = self.kappa_nu * (lmbda_nu-1.)
        bsb_dnmntr_trm = conditional(ge(bsb_dnmntr_trm, DOLFIN_EPS),
                                     bsb_dnmntr_trm, DOLFIN_EPS)
        
        lmbda_c_eq_bsb_val = lmbda_nu - 1. / bsb_dnmntr_trm

        # super-critical Bergstrom approximant
        lmbda_c_eq_bsp_val = (
            lmbda_nu
            - self.kappa_nu / self.zeta_nu_char**2 * (lmbda_nu-1.)**3
        )

        # evaluate the precise value of the equilibrium chain stretch
        lmbda_c_eq_val_i = conditional(gt(lmbda_nu, 1.), lmbda_c_eq_psb_val, 0.)
        lmbda_c_eq_val_ii = conditional(gt(lmbda_nu, self.lmbda_nu_pade2berg_crit),
                                        lmbda_c_eq_bsb_val, lmbda_c_eq_val_i)
        lmbda_c_eq_val = conditional(gt(lmbda_nu, self.lmbda_nu_crit),
                                     lmbda_c_eq_bsp_val, lmbda_c_eq_val_ii)

        return lmbda_c_eq_val

    def lmbda_nu_ufl_fenics_func(self, lmbda_c_eq):
        """Segment stretch.
        
        This function computes the segment stretch as a function of 
        the equilibrium chain stretch. This function is implemented in
        the Unified Form Language (UFL) for FEniCS.
        """
        # sub-critical Pade approximant
        alpha_tilde_psb = 1.

        trm_i = -3. * (self.kappa_nu+1.)
        trm_ii = -(2.*self.kappa_nu+3.)
        beta_tilde_psb_nmrtr = trm_i + lmbda_c_eq * trm_ii
        beta_tilde_psb_dnmntr = self.kappa_nu + 1.
        beta_tilde_psb = beta_tilde_psb_nmrtr / beta_tilde_psb_dnmntr

        trm_i = 2. * self.kappa_nu
        trm_ii = 4. * self.kappa_nu + 6.
        trm_iii = self.kappa_nu + 3.
        gamma_tilde_psb_nmrtr = trm_i + lmbda_c_eq * (trm_ii+lmbda_c_eq*trm_iii)
        gamma_tilde_psb_dnmntr = self.kappa_nu + 1.
        gamma_tilde_psb = gamma_tilde_psb_nmrtr / gamma_tilde_psb_dnmntr

        trm_i = 2.
        trm_ii = 2. * self.kappa_nu
        trm_iii = self.kappa_nu + 3.
        delta_tilde_psb_nmrtr = (
            trm_i - lmbda_c_eq * (trm_ii+lmbda_c_eq*(trm_iii+lmbda_c_eq))
        )
        delta_tilde_psb_dnmntr = self.kappa_nu + 1.
        delta_tilde_psb = delta_tilde_psb_nmrtr / delta_tilde_psb_dnmntr

        pi_tilde_psb_nmrtr = (
            3. * alpha_tilde_psb * gamma_tilde_psb - beta_tilde_psb**2
        )
        pi_tilde_psb_dnmntr = 3. * alpha_tilde_psb**2
        pi_tilde_psb = pi_tilde_psb_nmrtr / pi_tilde_psb_dnmntr

        rho_tilde_psb_nmrtr = (
            2. * beta_tilde_psb**3
            - 9. * alpha_tilde_psb * beta_tilde_psb * gamma_tilde_psb 
            + 27. * alpha_tilde_psb**2 * delta_tilde_psb
        )
        rho_tilde_psb_dnmntr = 27. * alpha_tilde_psb**3
        rho_tilde_psb = rho_tilde_psb_nmrtr / rho_tilde_psb_dnmntr

        acos_arg_psb = (
            3. * rho_tilde_psb / (2.*pi_tilde_psb) * sqrt(-3./pi_tilde_psb)
        )
        acos_arg_psb = conditional(ge(acos_arg_psb, 1.-DOLFIN_EPS),
                                   1.-DOLFIN_EPS, acos_arg_psb)
        acos_arg_psb = conditional(le(acos_arg_psb, -1.+DOLFIN_EPS),
                                   -1.+DOLFIN_EPS, acos_arg_psb)
        
        cos_arg_psb = 1. / 3. * acos(acos_arg_psb) - 2. * DOLFIN_PI / 3.

        lmbda_nu_psb_val = (
            2. * sqrt(-pi_tilde_psb/3.) * cos(cos_arg_psb)
            - beta_tilde_psb / (3.*alpha_tilde_psb)
        )

        # sub-critical Bergstrom approximant
        sqrt_arg_bsb = lmbda_c_eq**2 - 2. * lmbda_c_eq + 1. + 4. / self.kappa_nu
        lmbda_nu_bsb_val = (lmbda_c_eq+1.+sqrt(sqrt_arg_bsb)) / 2.

        # super-critical Bergstrom approximant
        alpha_tilde_bsp = 1.
        beta_tilde_bsp = -3.
        gamma_tilde_bsp = 3. - self.zeta_nu_char**2 / self.kappa_nu
        delta_tilde_bsp = self.zeta_nu_char**2 / self.kappa_nu * lmbda_c_eq - 1.

        pi_tilde_bsp_nmrtr = (
            3. * alpha_tilde_bsp * gamma_tilde_bsp - beta_tilde_bsp**2
        )
        pi_tilde_bsp_dnmntr = 3. * alpha_tilde_bsp**2
        pi_tilde_bsp = pi_tilde_bsp_nmrtr / pi_tilde_bsp_dnmntr

        rho_tilde_bsp_nmrtr = (
            2. * beta_tilde_bsp**3
            - 9. * alpha_tilde_bsp * beta_tilde_bsp * gamma_tilde_bsp
            + 27. * alpha_tilde_bsp**2 * delta_tilde_bsp
        )
        rho_tilde_bsp_dnmntr = 27. * alpha_tilde_bsp**3
        rho_tilde_bsp = rho_tilde_bsp_nmrtr / rho_tilde_bsp_dnmntr

        acos_arg_bsp = (
            3. * rho_tilde_bsp / (2.*pi_tilde_bsp) * sqrt(-3./pi_tilde_bsp)
        )
        acos_arg_bsp = conditional(ge(acos_arg_bsp, 1.-DOLFIN_EPS),
                                   1.-DOLFIN_EPS, acos_arg_bsp)
        acos_arg_bsp = conditional(le(acos_arg_bsp, -1.+DOLFIN_EPS),
                                   -1.+DOLFIN_EPS, acos_arg_bsp)
        
        cos_arg_bsp = 1. / 3. * acos(acos_arg_bsp) - 2. * DOLFIN_PI / 3.

        lmbda_nu_bsp_val = (
            2. * sqrt(-pi_tilde_bsp/3.) * cos(cos_arg_bsp)
            - beta_tilde_bsp / (3.*alpha_tilde_bsp)
        )

        # evaluate the precise value of the segment stretch
        lmbda_nu_val_i = conditional(gt(lmbda_c_eq, 0.), lmbda_nu_psb_val, 1.)
        lmbda_nu_val_ii = conditional(gt(lmbda_c_eq, self.lmbda_c_eq_pade2berg_crit),
                                      lmbda_nu_bsb_val, lmbda_nu_val_i)
        lmbda_nu_val = conditional(gt(lmbda_c_eq, self.lmbda_c_eq_crit),
                                   lmbda_nu_bsp_val, lmbda_nu_val_ii)

        return lmbda_nu_val

    def L_ufl_fenics_func(self, x):
        """Langevin function.

        This function computes the Langevin function of a scalar
        argument. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        y = conditional(ge(x, self.cond_val), x, self.cond_val)
        L_val = 1. / tanh(y) - 1. / y
        return conditional(ge(x, self.cond_val), L_val, 0.)

    def inv_L_ufl_fenics_func(self, lmbda_comp_nu):
        """Jedynak R[9,2] inverse Langevin approximant.
        
        This function computes the Jedynak R[9,2] inverse Langevin
        approximant as a function of the result of the equilibrium
        chain stretch minus the segment stretch plus one. This function
        is implemented in the Unified Form Language (UFL) for FEniCS.
        """
        trm_i = 3.
        trm_ii = -1.00651 * lmbda_comp_nu**2
        trm_iii = -0.962251 * lmbda_comp_nu**4
        trm_iv = 1.47353 * lmbda_comp_nu**6
        trm_v = -0.48953 * lmbda_comp_nu**8
        nmrtr = lmbda_comp_nu * (trm_i+trm_ii+trm_iii+trm_iv+trm_v)
        dnmntr = (1.-lmbda_comp_nu) * (1.+1.01524*lmbda_comp_nu)
        dnmntr = conditional(ge(dnmntr, DOLFIN_EPS), dnmntr, DOLFIN_EPS)
        return nmrtr / dnmntr
    
    def s_cnu_ufl_fenics_func(self, lmbda_comp_nu):
        """Nondimensional chain-level entropic free energy contribution
        per segment as calculated by the Jedynak R[9,2] inverse Langevin
        approximate.
        
        This function computes the nondimensional chain-level entropic
        free energy contribution per segment as calculated by the
        Jedynak R[9,2] inverse Langevin approximate as a function of the
        result of the equilibrium chain stretch minus the segment
        stretch plus one. This function is implemented in the Unified
        Form Language (UFL) for FEniCS.
        """
        ln_arg_i = 1.00000000002049 - lmbda_comp_nu
        ln_arg_ii = lmbda_comp_nu + 0.98498877114821

        trm_i = 0.0602726941412868 * lmbda_comp_nu**8
        trm_ii = 0.00103401966455583 * lmbda_comp_nu**7
        trm_iii = -0.162726405850159 * lmbda_comp_nu**6
        trm_iv = -0.00150537112388157 * lmbda_comp_nu**5
        trm_v = -0.00350216312906114 * lmbda_comp_nu**4
        trm_vi = -0.00254138511870934 * lmbda_comp_nu**3
        trm_vii = 0.488744117329956 * lmbda_comp_nu**2
        trm_viii = 0.0071635921950366 * lmbda_comp_nu
        trm_ix = -0.999999503781195 * ln(ln_arg_i)
        trm_x = -0.992044340231098 * ln(ln_arg_ii)
        trm_xi = -0.0150047080499398
        return (
            trm_i + trm_ii + trm_iii + trm_iv + trm_v + trm_vi + trm_vii
            + trm_viii + trm_ix + trm_x + trm_xi
        )
    
    def s_cnu_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional chain-level entropic
        free energy contribution per segment.
        
        This function computes the nondimensional chain-level entropic
        free energy contribution per segment as a function of the
        applied segment stretch. This function is implemented in the
        Unified Form Language (UFL) for FEniCS.
        """
        lmbda_nu_hat_cond = (
            conditional(ge(lmbda_nu_hat, 1.+self.cond_val),
                        lmbda_nu_hat, 1.+self.cond_val)
        )
        xi_c_hat = self.xi_c_analytical_ufl_fenics_func(lmbda_nu_hat_cond)
        s_cnu_val = (
            self.L_ufl_fenics_func(xi_c_hat) * xi_c_hat
            + ln(xi_c_hat/sinh(xi_c_hat))
        )
        return conditional(ge(lmbda_nu_hat, 1.+self.cond_val), s_cnu_val, 0.)
    
    def psi_cnu_ufl_fenics_func(self, lmbda_nu, lmbda_c_eq):
        """Nondimensional chain-level Helmholtz free energy per segment.
        
        This function computes the nondimensional chain-level Helmholtz
        free energy per segment as a function of the segment stretch and
        the equilibrium chain stretch. This function is implemented in
        the Unified Form Language (UFL) for FEniCS.
        """
        lmbda_comp_nu = lmbda_c_eq - lmbda_nu + 1.
        
        return (
            self.u_nu_ufl_fenics_func(lmbda_nu)
            + self.s_cnu_ufl_fenics_func(lmbda_comp_nu)
        )
    
    def psi_cnu_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional chain-level Helmholtz
        free energy per segment.
        
        This function computes the nondimensional chain-level Helmholtz
        free energy per segment as a function of the applied segment
        stretch. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        return (
            self.u_nu_analytical_ufl_fenics_func(lmbda_nu_hat)
            + self.s_cnu_analytical_ufl_fenics_func(lmbda_nu_hat)
        )
    
    def xi_c_ufl_fenics_func(self, lmbda_nu, lmbda_c_eq):
        """Nondimensional chain force.
        
        This function computes the nondimensional chain force as a
        function of the segment stretch and the equilibrium chain
        stretch. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        lmbda_comp_nu = lmbda_c_eq - lmbda_nu + 1.
        
        return self.inv_L_ufl_fenics_func(lmbda_comp_nu)
    
    def xi_c_analytical_ufl_fenics_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional chain force.
        
        This function computes the nondimensional chain force as a
        function of the applied segment stretch. This function is
        implemented in the Unified Form Language (UFL) for FEniCS.
        """
        return self.u_nu_prime_analytical_ufl_fenics_func(lmbda_nu_hat)
    
    def lmbda_nu_xi_c_hat_ufl_fenics_func(self, xi_c_hat):
        """Segment stretch under an applied chain force.
        
        This function computes the segment stretch under an applied 
        chain force as a function of the applied nondimensional chain
        force xi_c_hat. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.

        Note that if the applied nondimensional chain force value is
        greater than the analytically calculated critical maximum
        nondimensional chain force value xi_c_crit, then a non-physical
        segment stretch value of zero is returned
        """
        xi_c_max = self.xi_c_crit + self.cond_val
        lmbda_nu_val = 1. + xi_c_hat / self.kappa_nu
        return conditional(le(xi_c_hat, xi_c_max), lmbda_nu_val, 0.)