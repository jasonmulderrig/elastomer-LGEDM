# import necessary libraries
from __future__ import division
from dolfin import *
from elastomer_lgedm import ElastomerLGEDMProblem, gmsh_mesher, mesh_topologier
import numpy as np
from scipy import constants
import textwrap


class RefinedNotchedCrack(ElastomerLGEDMProblem):

    def __init__(
            self, L, H, x_notch_point, r_notch,
            notch_fine_mesh_layer_level_num, fine_mesh_elem_size,
            coarse_mesh_elem_size, l_nl):

        self.L = L
        self.H = H
        self.x_notch_point = x_notch_point
        self.r_notch = r_notch
        self.notch_fine_mesh_layer_level_num = notch_fine_mesh_layer_level_num
        self.fine_mesh_elem_size = fine_mesh_elem_size
        self.coarse_mesh_elem_size  = coarse_mesh_elem_size
        self.l_nl = l_nl
        
        ElastomerLGEDMProblem.__init__(self)
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """
        # Problem parameters
        pp = self.parameters["problem"]
        pp["meshtype"] = "refined_notched_crack"

        # Material parameters
        mp = self.parameters["material"]
        # Updated material parameters
        ump = Parameters("material")

        # Fundamental material constants
        k_B = constants.value(u"Boltzmann constant") # J/K
        N_A = constants.value(u"Avogadro constant") # 1/mol
        h = constants.value(u"Planck constant") # J/Hz
        hbar = h / (2*np.pi) # J*sec
        T = 298 # absolute room temperature, K
        beta = 1. / (k_B*T) # 1/J
        omega_0 = 1. / (beta*hbar) # J/(J*sec) = 1/sec
        zeta_nu_char = 298.9
        kappa_nu = 912.2
        nu_b = "None"
        zeta_b_char = "None"
        kappa_b = "None"

        ump.add("k_B", k_B)
        ump.add("N_A", N_A)
        ump.add("h", h)
        ump.add("hbar", hbar)
        ump.add("T", T)
        ump.add("beta", beta)
        ump.add("omega_0", omega_0)
        ump.add("zeta_nu_char", zeta_nu_char)
        ump.add("kappa_nu", kappa_nu)
        ump.add("nu_b", nu_b)
        ump.add("zeta_b_char", zeta_b_char)
        ump.add("kappa_b", kappa_b)

        # Network-level damage
        d_c_lmbda_nu_crit_min = 1.001
        d_c_lmbda_nu_crit_max = 1.005

        ump.add("d_c_lmbda_nu_crit_min", d_c_lmbda_nu_crit_min)
        ump.add("d_c_lmbda_nu_crit_max", d_c_lmbda_nu_crit_max)

        # Non-local interaction length scale
        ump.add("l_nl", self.l_nl)

        # Interaction intensity
        ump.add("n", 1)

        # Define the chain segment number statistics in the network
        ump.add("nu_distribution", "uniform")

        nu_list = [6]
        nu_min = min(nu_list)
        nu_max = max(nu_list)
        nu_num = len(nu_list)
        nu_bar = 6
        Delta_nu = nu_bar - nu_min

        for nu_indx in range(nu_num):
            ump.add("nu_indx_"+str(nu_indx)+"_nu_val", nu_list[nu_indx])
        ump.add("nu_min", nu_min)
        ump.add("nu_max", nu_max)
        ump.add("nu_num", nu_num)
        ump.add("nu_bar", nu_bar)
        ump.add("Delta_nu", Delta_nu)

        # Define chain segment numbers to chunk during deformation
        nu_chunks_list = nu_list
        nu_chunks_num = len(nu_chunks_list)
        nu_chunks_indx_in_nu_list = nu_chunks_list.copy()
        for nu_chunk_indx in range(nu_chunks_num):
            nu_chunks_indx_in_nu_list[nu_chunk_indx] = (
                nu_list.index(nu_chunks_list[nu_chunk_indx])
            )
        nu_chunks_label_list = [
            r'$\nu='+str(nu_list[nu_chunks_indx_in_nu_list[nu_chunk_indx]])+'$'
            for nu_chunk_indx in range(nu_chunks_num)
        ]
        nu_chunks_color_list = ['blue']

        for nu_chunk_indx in range(nu_chunks_num):
            ump.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_nu_val",
                    nu_chunks_list[nu_chunk_indx])
            ump.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_indx_val_in_nu_list",
                    nu_chunks_indx_in_nu_list[nu_chunk_indx])
            ump.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_label",
                    nu_chunks_label_list[nu_chunk_indx])
            ump.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_color",
                    nu_chunks_color_list[nu_chunk_indx])

        ump.add("nu_chunks_num", nu_chunks_num)

        mp.assign(ump)

        # Deformation parameters
        dp = self.parameters["deformation"]

        dp["deformation_type"] = "uniaxial"

        dp["K__G"] = 10
        dp["k_cond_val"] = 1.e-2
        dp["k_g_cond_val"] = 1.e-3

        dp["strain_rate"] = 0.1 # 1/sec
        dp["t_max"] = 5.00 # sec
        dp["t_step"] = 0.02 # sec
        dp["t_step_chunk_num"] = 2

        # Post-processing parameters
        ppp = self.parameters["post_processing"]

        ppp["save_lmbda_c_mesh"] = False
        ppp["save_lmbda_c_eq_mesh"] = False
        ppp["save_lmbda_nu_mesh"] = False
        ppp["save_lmbda_c_tilde_mesh"] = False
        ppp["save_lmbda_c_eq_tilde_mesh"] = False
        ppp["save_lmbda_nu_tilde_mesh"] = False
        ppp["save_lmbda_c_tilde_max_mesh"] = False
        ppp["save_lmbda_c_eq_tilde_max_mesh"] = False
        ppp["save_lmbda_nu_tilde_max_mesh"] = False
        ppp["save_g_mesh"] = False
        ppp["save_upsilon_c_mesh"] = False
        ppp["save_Upsilon_c_mesh"] = False
        ppp["save_d_c_mesh"] = False
        ppp["save_D_c_mesh"] = False
        ppp["save_u_mesh"] = False
        ppp["save_F_mesh"] = False
        ppp["save_sigma_mesh"] = False

        ppp["save_u_mesh"] = True
        ppp["save_lmbda_c_tilde_max_mesh"] = True
        ppp["save_g_mesh"] = True
        ppp["save_D_c_mesh"] = True

    def set_user_parameters_in_lists(self):
        """
        Recast and define particular parameters as attributes in Python
        lists
        """
        mp = self.parameters["material"]
        
        # Chain segment numbers
        self.nu_num = mp["nu_num"]
        nu_list = []

        for nu_indx in range(self.nu_num):
            nu_val = mp["nu_indx_"+str(nu_indx)+"_nu_val"]
            nu_list.append(nu_val)
        
        self.nu_list = nu_list

        # Chain segment number chunks
        self.nu_chunks_num = mp["nu_chunks_num"]
        nu_chunks_list = []
        nu_chunks_indx_in_nu_list = []
        nu_chunks_label_list = []
        nu_chunks_color_list = []

        for nu_chunk_indx in range(self.nu_chunks_num):
            nu_chunks_val = mp["nu_chunk_indx_"+str(nu_chunk_indx)+"_nu_val"]
            nu_chunks_indx_val_in_nu_list = (
                mp["nu_chunk_indx_"+str(nu_chunk_indx)+"_indx_val_in_nu_list"]
            )
            nu_chunks_label = mp["nu_chunk_indx_"+str(nu_chunk_indx)+"_label"]
            nu_chunks_color = mp["nu_chunk_indx_"+str(nu_chunk_indx)+"_color"]

            nu_chunks_list.append(nu_chunks_val)
            nu_chunks_indx_in_nu_list.append(nu_chunks_indx_val_in_nu_list)
            nu_chunks_label_list.append(nu_chunks_label)
            nu_chunks_color_list.append(nu_chunks_color)
        
        self.nu_chunks_list = nu_chunks_list
        self.nu_chunks_indx_in_nu_list = nu_chunks_indx_in_nu_list
        self.nu_chunks_label_list = nu_chunks_label_list
        self.nu_chunks_color_list = nu_chunks_color_list
    
    def set_user_parameters_in_dicts(self):
        """
        Recast and define particular parameters as attributes in Python
        dictionaries
        """
        femp = self.parameters["fem"]

        # Metadata quadrature degree
        self.metadata  = {"quadrature_degree": femp["quadrature_degree"]}

        self.two_dim_vector_indx_dict = {
            "1": 0,
            "2": 1
        }

        # Tensor-to-Voigt notation dictionary
        self.two_dim_tensor2voigt_vector_indx_dict = {
            "11": 0,
            "12": 1,
            "21": 2,
            "22": 3
        }

        # Dictionary for solver parameters
        self.solver_parameters_dict = {"nonlinear_solver": "snes",
                                       "symmetric": True,
                                       "snes_solver": {"linear_solver": "mumps",
                                                       "method": "newtontr",
                                                       "line_search": "cp",
                                                       "preconditioner": "hypre_amg",
                                                       "maximum_iterations": 200,
                                                       "absolute_tolerance": 1e-8,
                                                       "relative_tolerance": 1e-4, # 1e-4 seems to work
                                                       "solution_tolerance": 1e-4, # 1e-4 seems to work
                                                       "report": True,
                                                       "error_on_nonconvergence": False
                                                       }
                                                       }
        
    def prefix(self):
        pp = self.parameters["problem"]
        return self.modelname + "_" + pp["meshtype"] + "_" + "problem"
    
    def define_mesh(self):
        """
        Define the mesh for the problem
        """
        geofile = \
            """
            Mesh.Algorithm = 8;
            notch_fine_mesh_layer_level_num = DefineNumber[ %g, Name "Parameters/notch_fine_mesh_layer_level_num" ];
            fine_mesh_elem_size = DefineNumber[ %g, Name "Parameters/fine_mesh_elem_size" ];
            coarse_mesh_elem_size = DefineNumber[ %g, Name "Parameters/coarse_mesh_elem_size" ];
            x_notch_point = DefineNumber[ %g, Name "Parameters/x_notch_point" ];
            r_notch = DefineNumber[ %g, Name "Parameters/r_notch" ];
            L = DefineNumber[ %g, Name "Parameters/L"];
            H = DefineNumber[ %g, Name "Parameters/H"];
            Point(1) = {0, 0, 0, fine_mesh_elem_size};
            Point(2) = {x_notch_point-r_notch, 0, 0, fine_mesh_elem_size};
            Point(3) = {0, -r_notch, 0, fine_mesh_elem_size};
            Point(4) = {0, -r_notch-notch_fine_mesh_layer_level_num*fine_mesh_elem_size, 0, fine_mesh_elem_size};
            Point(5) = {0, -H/2, 0, coarse_mesh_elem_size};
            Point(6) = {L, -H/2, 0, coarse_mesh_elem_size};
            Point(7) = {L, -r_notch-notch_fine_mesh_layer_level_num*fine_mesh_elem_size, 0, fine_mesh_elem_size};
            Point(8) = {L, r_notch+notch_fine_mesh_layer_level_num*fine_mesh_elem_size, 0, fine_mesh_elem_size};
            Point(9) = {L, H/2, 0, coarse_mesh_elem_size};
            Point(10) = {0, H/2, 0, coarse_mesh_elem_size};
            Point(11) = {0, r_notch+notch_fine_mesh_layer_level_num*fine_mesh_elem_size, 0, fine_mesh_elem_size};
            Point(12) = {0, r_notch, 0, fine_mesh_elem_size};
            Point(13) = {x_notch_point-r_notch, r_notch, 0, fine_mesh_elem_size};
            Point(14) = {x_notch_point, 0, 0, fine_mesh_elem_size};
            Point(15) = {x_notch_point-r_notch, -r_notch, 0, fine_mesh_elem_size};
            Line(1) = {15, 3};
            Line(2) = {3, 4};
            Line(3) = {4, 7};
            Line(4) = {4, 5};
            Line(5) = {5, 6};
            Line(6) = {6, 7};
            Line(7) = {7, 8};
            Line(8) = {8, 9};
            Line(9) = {9, 10};
            Line(10) = {10, 11};
            Line(11) = {11, 8};
            Line(12) = {11, 12};
            Line(13) = {12, 13};
            Circle(14) = {13, 2, 14};
            Circle(15) = {14, 2, 15};
            Curve Loop(21) = {1, 2, 3, 7, -11, 12, 13, 14, 15};
            Curve Loop(22) = {8, 9, 10, 11};
            Curve Loop(23) = {6, -3, 4, 5};
            Plane Surface(31) = {21};
            Plane Surface(32) = {22};
            Plane Surface(33) = {23};
            Mesh.MshFileVersion = 2.0;
            """ % (self.notch_fine_mesh_layer_level_num, self.fine_mesh_elem_size, self.coarse_mesh_elem_size, self.x_notch_point, self.r_notch, self.L, self.H)
        
        geofile = textwrap.dedent(geofile)

        L_string = "{:.1f}".format(self.L)
        H_string = "{:.1f}".format(self.H)
        x_notch_point_string = "{:.1f}".format(self.x_notch_point)
        r_notch_string = "{:.1f}".format(self.r_notch)
        notch_fine_mesh_layer_level_num_string = (
            "{:d}".format(self.notch_fine_mesh_layer_level_num)
        )
        fine_mesh_elem_size_string = "{:.3f}".format(self.fine_mesh_elem_size)
        coarse_mesh_elem_size_string = (
            "{:.1f}".format(self.coarse_mesh_elem_size)
        )

        meshname = (
            L_string + "_" + H_string + "_" + x_notch_point_string
            + "_" + r_notch_string
            + "_" + notch_fine_mesh_layer_level_num_string
            + "_" + fine_mesh_elem_size_string
            + "_" + coarse_mesh_elem_size_string
        )
        
        return gmsh_mesher(geofile, self.prefix(), meshname)

    def define_bcs(self):
        """
        Return a list of boundary conditions
        """
        self.lines = (
            MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        )
        self.lines.set_all(0)

        L = self.L
        H = self.H
        x_notch_point = self.x_notch_point
        r_notch = self.r_notch

        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0., DOLFIN_EPS)
        
        class RightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], L, DOLFIN_EPS)

        class BottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], -H/2., DOLFIN_EPS)
        
        class TopBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], H/2., DOLFIN_EPS)

        class Notch(SubDomain):
            def inside(self, x, on_boundary):
                r_notch_sq = (x[0]-(x_notch_point-r_notch))**2 + x[1]**2
                return r_notch_sq <= (r_notch + DOLFIN_EPS)**2

        LeftBoundary().mark(self.lines, 1)
        RightBoundary().mark(self.lines, 2)
        BottomBoundary().mark(self.lines, 3)
        TopBoundary().mark(self.lines, 4)
        Notch().mark(self.lines, 5)

        mesh_topologier(self.lines, self.prefix(), "lines")

        self.u_y_expression = Expression("u_y", u_y=0., degree=0)

        bc_I = DirichletBC(self.V.sub(0).sub(1), Constant(0.), BottomBoundary())
        bc_II = DirichletBC(self.V.sub(0).sub(0), Constant(0.), RightBoundary())
        bc_III = (
            DirichletBC(self.V.sub(0).sub(1), self.u_y_expression, TopBoundary())
        )

        return [bc_I, bc_II, bc_III]
    
    def F_func(self, t):
        """
        Function defining the deformation
        """
        dp = self.parameters["deformation"]

        return 1 + dp["strain_rate"] * (t-dp["t_min"])
    
    def initialize_lmbda(self):
        lmbda_y = [] # unitless
        lmbda_y_chunks = [] # unitless

        return lmbda_y, lmbda_y_chunks
    
    def store_initialized_lmbda(self, lmbda):
        lmbda_y_val = 1 # assuming no pre-stretching
        
        lmbda_y = lmbda[0]
        lmbda_y_chunks = lmbda[1]
        
        lmbda_y.append(lmbda_y_val)
        lmbda_y_chunks.append(lmbda_y_val)
        
        return lmbda_y, lmbda_y_chunks
    
    def calculate_lmbda_func(self, t_val):
        lmbda_y_val = self.F_func(t_val)

        return lmbda_y_val
    
    def store_calculated_lmbda(self, lmbda, lmbda_val):
        lmbda_y = lmbda[0]
        lmbda_y_chunks = lmbda[1]
        lmbda_y_val = lmbda_val
        
        lmbda_y.append(lmbda_y_val)
        
        return lmbda_y, lmbda_y_chunks
    
    def store_calculated_lmbda_chunk_post_processing(self, lmbda, lmbda_val):
        lmbda_y = lmbda[0]
        lmbda_y_chunks = lmbda[1]
        lmbda_y_val = lmbda_val
        
        lmbda_y_chunks.append(lmbda_y_val)
        
        return lmbda_y, lmbda_y_chunks
    
    def calculate_u_func(self, lmbda):
        lmbda_y = lmbda[0]
        lmbda_y_chunks = lmbda[1]

        u_y = [self.H*(lmbda_y_val-1) for lmbda_y_val in lmbda_y]
        u_y_chunks = [
            self.H*(lmbda_y_chunks_val-1) for lmbda_y_chunks_val in lmbda_y_chunks
        ]

        return u_y, u_y_chunks
    
    def save_deformation_attributes(self, lmbda, u):
        lmbda_y = lmbda[0]
        lmbda_y_chunks = lmbda[1]

        u_y = u[0]
        u_y_chunks = u[1]

        self.lmbda_y = lmbda_y
        self.lmbda_y_chunks = lmbda_y_chunks
        self.u_y = u_y
        self.u_y_chunks = u_y_chunks

    def set_loading(self):
        """
        Update Dirichlet boundary conditions
        """
        self.u_y_expression.u_y = self.u_y[self.t_indx]


if __name__ == '__main__':

    L = 1.0
    H = 1.5
    x_notch_point = 0.5
    r_notch = 0.02
    notch_fine_mesh_layer_level_num = 1
    fine_mesh_elem_size = 0.002
    coarse_mesh_elem_size = 0.1
    l_nl = 0.02
    problem = (
        RefinedNotchedCrack(L, H, x_notch_point, r_notch,
                            notch_fine_mesh_layer_level_num, fine_mesh_elem_size,
                            coarse_mesh_elem_size, l_nl)
    )
    problem.solve()