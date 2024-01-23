# import necessary libraries
from __future__ import division
from dolfin import *
from .default_parameters import default_parameters
from .composite_ufjc_network import CompositeuFJCNetwork
from .utility import generate_savedir, print0, local_project
import sys


class ElastomerLGEDMProblem(object):
    """
    Problem class for composite uFJC networks with non-local scission
    leading to macroscopic fracture. The composite uFJC network here
    exhibits the following characteristics:
    -> Physical dimensionality: Two-dimensional (2D)
        -> Two-dimensional formulation: Plane strain
    -> Incompressibility assumption: Nearly-incompressible
    -> Macro-to-micro deformation assumption: Nonaffine
    -> Micro-to-macro homogenizatio scheme: Eight-chain model
    -> Chain-level load sharing assumption: Equal strain
    -> Chain scission rate-dependence: Rate-independent
    """
    def __init__(self):

        # Set the mpi communicator of the object
        self.comm_rank = MPI.rank(MPI.comm_world)
        self.comm_size = MPI.size(MPI.comm_world)

        # Parameters
        self.parameters = default_parameters()
        self.set_user_parameters()
        self.set_user_parameters_in_lists()
        self.set_user_parameters_in_dicts()
        self.set_user_modelname()

        # Setup filesystem
        self.savedir = generate_savedir(self.prefix())

        # Pre-processing
        ppp = self.parameters["pre_processing"]
        set_log_level(LogLevel.WARNING)
        parameters["form_compiler"]["optimize"] = ppp["form_compiler_optimize"]
        parameters["form_compiler"]["cpp_optimize"] = (
            ppp["form_compiler_cpp_optimize"]
        )
        parameters["form_compiler"]["representation"] = (
            ppp["form_compiler_representation"]
        )
        parameters["form_compiler"]["quadrature_degree"] = (
            ppp["form_compiler_quadrature_degree"]
        )
        info(parameters, True)

        # Mesh
        self.mesh = self.define_mesh()
        # Spatial dimensions of the mesh
        self.dimension = self.mesh.geometry().dim()

        # Material
        mp = self.parameters["material"]
        self.material = CompositeuFJCNetwork(mp)

        # Variational formulation
        self.set_variational_formulation()

        # Set boundary conditions
        self.define_bcs()

        # Deformation
        self.set_applied_deformation()

        # Post-processing
        self.set_post_processing()

        # Solver set up
        self.solver_setup()
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """
        pass

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
        self.metadata = {"quadrature_degree": femp["quadrature_degree"]}

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

        # Dictionaries for solver parameters
        self.solver_parameters_dict = {"nonlinear_solver": "snes",
                                       "symmetric": True,
                                       "snes_solver": {"linear_solver": "mumps",
                                                       "method": "newtontr",
                                                       "line_search": "cp",
                                                       "preconditioner": "hypre_amg",
                                                       "maximum_iterations": 200,
                                                       "absolute_tolerance": 1e-8,
                                                       "relative_tolerance": 1e-7,
                                                       "solution_tolerance": 1e-7,
                                                       "report": True,
                                                       "error_on_nonconvergence": False
                                                        }
                                                        }
    
    def set_user_modelname(self):
        self.modelname = "elastomer_lgedm"
    
    def prefix(self):
        return self.modelname + "_" + "problem"

    def define_mesh(self):
        """
        Define the mesh for the problem
        """
        pass

    def set_variational_formulation(self):
        """
        Define the variational formulation problem to be solved
        """
        femp = self.parameters["fem"]
        mp = self.parameters["material"]
        ppp = self.parameters["post_processing"]

        # Create function space for displacement
        self.V_u = VectorFunctionSpace(self.mesh, "CG", femp["u_degree"])
        # Create function space for non-local chain stretch
        self.V_lmbda_c_tilde = (
            FunctionSpace(self.mesh, "CG", femp["scalar_prmtr_degree"])
        )
        # Create function space for maximal non-local chain stretch
        self.V_lmbda_c_tilde_max = (
            FunctionSpace(self.mesh, "CG", femp["scalar_prmtr_degree"])
        )
        # Create function space for gradient activity
        self.V_g = FunctionSpace(self.mesh, "CG", femp["scalar_prmtr_degree"])
        # # Create function space for normalized anisotropic interaction
        # # tensor
        # self.V_H_tilde = (
        #     TensorFunctionSpace(self.mesh, "DG", femp["tensor_prmtr_degree"])
        # )

        # Create UFL element from the displacement function space
        self.V_u_ufl_elem = self.V_u.ufl_element()
        # Create UFL element from the non-local chain stretch function space
        self.V_lmbda_c_tilde_ufl_elem = self.V_lmbda_c_tilde.ufl_element()
        # Create UFL mixed element
        self.mixed_ufl_elem = (
            MixedElement([self.V_u_ufl_elem, self.V_lmbda_c_tilde_ufl_elem])
        )
        # Define function space for the UFL mixed element
        self.V = FunctionSpace(self.mesh, self.mixed_ufl_elem)

        # Define solution, trial, and test functions for the UFL mixed element
        self.mixed_space = Function(self.V)
        self.dmixed_space = TrialFunction(self.V)
        self.v_mixed_space = TestFunction(self.V)

        self.lmbda_c_tilde_max = Function(self.V_lmbda_c_tilde_max)
        self.g = Function(self.V_g)
        # self.H_tilde = Function(self.V_H_tilde)

        # Initialization
        class MixedSpaceInitialConditions(UserExpression):
            def eval(self, vals, x):
                vals[0] = 0.0 # u_x
                vals[1] = 0.0 # u_y
                vals[2] = 1.0 # lambda_c_tilde
            def value_shape(self):
                return (3,)
        
        class LambdaCTildeMaxInitialCondition(UserExpression):
            def eval(self, vals, x):
                vals[0] = 0.0
        
        class GInitialCondition(UserExpression):
            def eval(self, vals, x):
                vals[0] = 1.0
        
        # class HTildeInitialCondition(UserExpression):
        #     def eval(self, vals, x):
        #         vals[0] = 1
        #         vals[1] = 0
        #         vals[2] = 0
        #         vals[3] = 1
        #     def value_shape(self):
        #         return (2,2)
        
        mixed_space_degree = femp["u_degree"]
        ics = MixedSpaceInitialConditions(degree=mixed_space_degree)
        self.mixed_space.interpolate(ics)
        
        lmbda_c_tilde_max_ic = (
            LambdaCTildeMaxInitialCondition(degree=femp["scalar_prmtr_degree"])
        )
        self.lmbda_c_tilde_max.interpolate(lmbda_c_tilde_max_ic)

        g_ic = GInitialCondition(degree=femp["scalar_prmtr_degree"])
        self.g.interpolate(g_ic)

        # H_tilde_init = HTildeInitialCondition(degree=femp["tensor_prmtr_degree"])
        # self.H_tilde.interpolate(H_tilde_init)

        # Split the mixed function space and the mixed test function
        # space for displacement and non-local chain stretch
        (self.u, self.lmbda_c_tilde) = split(self.mixed_space)
        (self.v_u, self.v_lmbda_c_tilde) = split(self.v_mixed_space)

        # Define objects needed for calculations
        self.I = Identity(len(self.u))
        self.V_CG_scalar = (
            FunctionSpace(self.mesh, "CG", femp["scalar_prmtr_degree"])
        )
        self.V_DG_scalar = (
            FunctionSpace(self.mesh, "DG", femp["scalar_prmtr_degree"])
        )
        self.V_DG_tensor = (
            TensorFunctionSpace(self.mesh, "DG", femp["tensor_prmtr_degree"])
        )

        # Constant H_tilde equals the identity matrix
        self.H_tilde = self.I
        
        self.lmbda_c_tilde_max_val = Function(self.V_DG_scalar)
        self.g_val = Function(self.V_DG_scalar)
        # self.H_tilde_val = Function(self.V_DG_tensor)
        
        if ppp["save_lmbda_c_mesh"]:
            self.lmbda_c_val = Function(self.V_DG_scalar)
        
        if ppp["save_lmbda_c_eq_mesh"]:
            self.lmbda_c_eq_val = Function(self.V_DG_scalar)
        
        if ppp["save_lmbda_nu_mesh"]:
            self.lmbda_nu_val = Function(self.V_DG_scalar)
        
        if ppp["save_lmbda_c_eq_tilde_mesh"]:
            self.lmbda_c_eq_tilde_val = Function(self.V_DG_scalar)
        
        if ppp["save_lmbda_nu_tilde_mesh"]:
            self.lmbda_nu_tilde_val = Function(self.V_DG_scalar)
        
        if ppp["save_lmbda_c_eq_tilde_max_mesh"]:
            self.lmbda_c_eq_tilde_max_val = Function(self.V_DG_scalar)
        
        if ppp["save_lmbda_nu_tilde_max_mesh"]:
            self.lmbda_nu_tilde_max_val = Function(self.V_DG_scalar)
        
        if ppp["save_upsilon_c_mesh"]:
            self.upsilon_c_val = Function(self.V_DG_scalar)
        
        if ppp["save_Upsilon_c_mesh"]:
            self.Upsilon_c_val = Function(self.V_DG_scalar)
        
        if ppp["save_d_c_mesh"]:
            self.d_c_val = Function(self.V_DG_scalar)
        
        if ppp["save_D_c_mesh"]:
            self.D_c_val = Function(self.V_DG_scalar)
        
        if ppp["save_F_mesh"]:
            self.F_val = Function(self.V_DG_tensor)
        
        if ppp["save_sigma_mesh"]:
            self.sigma_val = Function(self.V_DG_tensor)

        # Kinematics
        # deformation gradient tensor
        self.F = self.I + grad(self.u)
        # inverse deformation gradient tensor
        self.F_inv = inv(self.F)
        # volume ratio
        self.J = det(self.F)
        # right Cauchy-Green tensor
        self.C = self.F.T * self.F
        # 2D plane strain form of the trace of right Cauchy-Green
        # tensor, where F_33 = 1 always
        self.I_C = tr(self.C) + 1
        # local chain stretch
        self.lmbda_c = sqrt(self.I_C/3.0)
    
        # Define body force and traction force
        self.b_hat = Constant((0.0, 0.0)) # Body force per unit volume
        self.t_hat = Constant((0.0, 0.0)) # Traction force on the boundary

        # Calculate the weak form for displacement
        self.WF_u = (
            inner(self.first_pk_stress_ufl_fenics_mesh_func(), grad(self.v_u)) * dx(metadata=self.metadata)
            - dot(self.b_hat, self.v_u) * dx(metadata=self.metadata)
            - dot(self.t_hat, self.v_u) * ds
        )

        # Calculate the weak form for non-local chain stretch
        self.WF_lmbda_c_tilde = (
            self.v_lmbda_c_tilde * self.lmbda_c_tilde * dx(metadata=self.metadata)
            + mp["l_nl"]**2 * self.g * dot(grad(self.v_lmbda_c_tilde), self.H_tilde*grad(self.lmbda_c_tilde)) * dx(metadata=self.metadata)
            - self.v_lmbda_c_tilde * self.lmbda_c * dx(metadata=self.metadata)
        )

        # Calculate the overall weak form
        self.WF = self.WF_u + self.WF_lmbda_c_tilde

        # Calculate the Gateaux derivative
        self.Jac = derivative(self.WF, self.mixed_space, self.dmixed_space)
    
    def define_bcs(self):
        """
        Define boundary conditions by returning a list of boundary
        conditions
        """
        self.bc = []
    
    def set_applied_deformation(self):
        """
        Set the applied deformation history
        """
        dp = self.parameters["deformation"]
        cond_val = 1e-8
        
        if dp["t_step"] > dp["t_max"]:
            error_message = """\
                Error: The time step is larger than the total \
                deformation time!
                """
            sys.exit(error_message)

        # Initialize the chunk counter and associated constants/lists
        chunk_counter = 0
        chunk_indx_val = 0
        chunk_indx = []

        # Initialization step
        t_val = dp["t_min"] # sec
        t = [] # sec
        t_chunks = [] # sec
        lmbda = self.initialize_lmbda()

        # Append to appropriate lists
        t.append(t_val)
        t_chunks.append(t_val)
        chunk_indx.append(chunk_indx_val)
        lmbda = self.store_initialized_lmbda(lmbda)

        # Update the chunk iteration counter
        chunk_counter += 1
        chunk_indx_val += 1

        # Advance to the first time step
        t_val += dp["t_step"]

        while t_val <= (dp["t_max"]+cond_val):
            # Calculate displacement at a particular time step
            lmbda_val = self.calculate_lmbda_func(t_val)

            # Append to appropriate lists
            t.append(t_val)
            lmbda = self.store_calculated_lmbda(lmbda, lmbda_val)

            if chunk_counter == dp["t_step_chunk_num"]:
                # Append to appropriate lists
                t_chunks.append(t_val)
                chunk_indx.append(chunk_indx_val)
                lmbda = (
                    self.store_calculated_lmbda_chunk_post_processing(lmbda, lmbda_val)
                )

                # Update the time step chunk iteration counter
                chunk_counter = 0

            # Advance to the next time step
            t_val += dp["t_step"]
            chunk_counter += 1
            chunk_indx_val += 1
        
        u = self.calculate_u_func(lmbda)

        # If the endpoint of the chunked applied deformation is not
        # equal to the true endpoint of the applied deformation, then
        # give the user the option to kill the simulation, or proceed on
        if chunk_indx[-1] != len(t)-1:
            error_message = """\
                The endpoint of the chunked applied deformation is not \
                equal to the endpoint of the actual applied deformation. \
                Do you wish to kill the simulation here? If so, say yes. \
                If not, the simulation will proceed on.
                """
            terminal_statement = input(error_message)
            if terminal_statement.lower() == 'yes':
                sys.exit()
            else: pass
        
        self.t = t
        self.t_chunks = t_chunks
        self.chunk_indx = chunk_indx
        self.save_deformation_attributes(lmbda, u)
    
    def F_func(self, t):
        """
        Function defining the deformation
        """
        dp = self.parameters["deformation"]

        return 1 + dp["strain_rate"] * (t-dp["t_min"])
    
    def initialize_lmbda(self):
        lmbda_x = [] # unitless
        lmbda_x_chunks = [] # unitless

        return lmbda_x, lmbda_x_chunks
    
    def store_initialized_lmbda(self, lmbda):
        lmbda_x_val = 1 # assuming no pre-stretching
        
        lmbda_x = lmbda[0]
        lmbda_x_chunks = lmbda[1]
        
        lmbda_x.append(lmbda_x_val)
        lmbda_x_chunks.append(lmbda_x_val)
        
        return lmbda_x, lmbda_x_chunks
    
    def calculate_lmbda_func(self, t_val):
        lmbda_x_val = self.F_func(t_val)

        return lmbda_x_val
    
    def store_calculated_lmbda(self, lmbda, lmbda_val):
        lmbda_x = lmbda[0]
        lmbda_x_chunks = lmbda[1]
        lmbda_x_val = lmbda_val
        
        lmbda_x.append(lmbda_x_val)
        
        return lmbda_x, lmbda_x_chunks
    
    def store_calculated_lmbda_chunk_post_processing(self, lmbda, lmbda_val):
        lmbda_x = lmbda[0]
        lmbda_x_chunks = lmbda[1]
        lmbda_x_val = lmbda_val
        
        lmbda_x_chunks.append(lmbda_x_val)
        
        return lmbda_x, lmbda_x_chunks
    
    def calculate_u_func(self, lmbda):
        lmbda_x = lmbda[0]
        lmbda_x_chunks = lmbda[1]

        u_x = [lmbda_x_val-1 for lmbda_x_val in lmbda_x]
        u_x_chunks = [
            lmbda_x_chunks_val-1 for lmbda_x_chunks_val in lmbda_x_chunks
        ]

        return u_x, u_x_chunks
    
    def save_deformation_attributes(self, lmbda, u):
        lmbda_x = lmbda[0]
        lmbda_x_chunks = lmbda[1]

        u_x = u[0]
        u_x_chunks = u[1]

        self.lmbda_x = lmbda_x
        self.lmbda_x_chunks = lmbda_x_chunks
        self.u_x = u_x
        self.u_x_chunks = u_x_chunks

    def set_post_processing(self):
        """
        Set up post-processing files and arrays
        """
        ppp = self.parameters["post_processing"]

        self.file_results = (
            XDMFFile(MPI.comm_world, self.savedir+ppp["file_results"])
        )
        self.file_results.parameters["rewrite_function_mesh"] = (
            ppp["rewrite_function_mesh"]
        )
        self.file_results.parameters["flush_output"] = (
            ppp["flush_output"]
        )
        self.file_results.parameters["functions_share_mesh"] = (
            ppp["functions_share_mesh"]
        )

    def solver_setup(self):
        """
        Setup the weak form solver
        """
        self.problem = (
            NonlinearVariationalProblem(self.WF, self.mixed_space,
                                        self.bc, J=self.Jac)
        )
        self.solver = NonlinearVariationalSolver(self.problem)

        self.solver.parameters.update(self.solver_parameters_dict)
        info(self.solver.parameters, True)

        (self.u, self.lmbda_c_tilde) = self.mixed_space.split(deepcopy=True)

    def solve(self):
        """
        Solve the evolution problem via solving the weak form through
        the applied deformation evolution
        """
        # Time stepping
        for t_indx, t_val in enumerate(self.t):
            # Update time stepping
            self.t_indx = t_indx
            self.t_val = t_val
            print0("\033[1;32m--- Time step # {0:2d}: t = {1:.3f} for the weak form network deformation ---\033[1;m".format(t_indx, t_val))
            self.set_loading()

            # Solve and account for network irreversibility
            self.solve_step()
            
            # Force all parallelization to unify after the solution step
            # and before post-processing
            MPI.barrier(MPI.comm_world)

            # Post-processing
            if self.t_indx in self.chunk_indx:
                self.post_processing()

            # Force all parallelization to unify after post-processing
            # and before user-defined post-processing
            MPI.barrier(MPI.comm_world)

            self.user_post_processing()
        
        # Force all parallelization to unify after user-defined
        # post-processing and before finalization
        MPI.barrier(MPI.comm_world)
    
    def set_loading(self):
        """
        Update Dirichlet boundary conditions
        """
        pass
    
    def solve_step(self):
        """
        Solve the weak form
        """
        print0("Displacement and non-local chain stretch problem")
        (iter, converged) = self.solver.solve()
        
        (self.u, self.lmbda_c_tilde) = self.mixed_space.split(deepcopy=True)

        # update maximal non-local chain stretch to account for network
        # irreversibility
        self.lmbda_c_tilde_max.vector()[:] = (
            project(conditional(gt(self.lmbda_c_tilde, self.lmbda_c_tilde_max), self.lmbda_c_tilde, self.lmbda_c_tilde_max), self.V_lmbda_c_tilde_max).vector()
        )

        # update the gradient activity to account for the decreasing
        # non-local interaction for increasing non-local damage
        self.g.vector()[:] = (
            project(self.g_ufl_fenics_mesh_func(), self.V_g).vector()
        )

        # update the normalized anisotropic interaction tensor to
        # account for the shape, size, and orientation of non-local
        # interactions at the microscale of the deformed polymer network
        # figure out how to do this later
    
    def post_processing(self):
        """
        Post-processing results
        """
        ppp  = self.parameters["post_processing"]

        if ppp["save_lmbda_c_mesh"]:
            local_project(self.lmbda_c, self.V_DG_scalar, self.lmbda_c_val)
            self.lmbda_c_val.rename("Chain stretch", "lmbda_c")
            self.file_results.write(self.lmbda_c_val, self.t_val,
                                    encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_lmbda_c_eq_mesh"]:
            for nu_chunk_indx in range(self.nu_chunks_num):
                nu_indx = self.nu_chunks_indx_in_nu_list[nu_chunk_indx]
                lmbda_c_eq___nu_chunk_val = (
                    self.lmbda_c_eq_ufl_fenics_mesh_func(nu_indx)
                )
                local_project(lmbda_c_eq___nu_chunk_val, self.V_DG_scalar,
                              self.lmbda_c_eq_val)

                nu_str = str(self.nu_list[nu_indx])
                name_str = "Equilibrium chain stretch nu = " + nu_str
                prmtr_str = "lmbda_c_eq nu = " + nu_str

                self.lmbda_c_eq_val.rename(name_str, prmtr_str)
                self.file_results.write(self.lmbda_c_eq_val, self.t_val,
                                        encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_lmbda_nu_mesh"]:
            for nu_chunk_indx in range(self.nu_chunks_num):
                nu_indx = self.nu_chunks_indx_in_nu_list[nu_chunk_indx]
                lmbda_nu___nu_chunk_val = (
                    self.lmbda_nu_ufl_fenics_mesh_func(nu_indx)
                )
                local_project(lmbda_nu___nu_chunk_val, self.V_DG_scalar,
                              self.lmbda_nu_val)

                nu_str = str(self.nu_list[nu_indx])
                name_str = "Segment stretch nu = " + nu_str
                prmtr_str = "lmbda_nu nu = " + nu_str

                self.lmbda_nu_val.rename(name_str, prmtr_str)
                self.file_results.write(self.lmbda_nu_val, self.t_val,
                                        encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_lmbda_c_tilde_mesh"]:
            self.lmbda_c_tilde.rename("Non-local chain stretch", "lmbda_c_tilde")
            self.file_results.write(self.lmbda_c_tilde, self.t_val,
                                    encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_lmbda_c_eq_tilde_mesh"]:
            for nu_chunk_indx in range(self.nu_chunks_num):
                nu_indx = self.nu_chunks_indx_in_nu_list[nu_chunk_indx]
                lmbda_c_eq_tilde___nu_chunk_val = (
                    self.lmbda_c_eq_tilde_ufl_fenics_mesh_func(nu_indx)
                )
                local_project(lmbda_c_eq_tilde___nu_chunk_val, self.V_DG_scalar,
                              self.lmbda_c_eq_tilde_val)

                nu_str = str(self.nu_list[nu_indx])
                name_str = "Non-local equilibrium chain stretch nu = " + nu_str
                prmtr_str = "lmbda_c_eq_tilde nu = " + nu_str

                self.lmbda_c_eq_tilde_val.rename(name_str, prmtr_str)
                self.file_results.write(self.lmbda_c_eq_tilde_val, self.t_val,
                                        encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_lmbda_nu_tilde_mesh"]:
            for nu_chunk_indx in range(self.nu_chunks_num):
                nu_indx = self.nu_chunks_indx_in_nu_list[nu_chunk_indx]
                lmbda_nu_tilde___nu_chunk_val = (
                    self.lmbda_nu_tilde_ufl_fenics_mesh_func(nu_indx)
                )
                local_project(lmbda_nu_tilde___nu_chunk_val, self.V_DG_scalar,
                              self.lmbda_nu_tilde_val)

                nu_str = str(self.nu_list[nu_indx])
                name_str = "Non-local segment stretch nu = " + nu_str
                prmtr_str = "lmbda_nu_tilde nu = " + nu_str

                self.lmbda_nu_tilde_val.rename(name_str, prmtr_str)
                self.file_results.write(self.lmbda_nu_tilde_val, self.t_val,
                                        encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_lmbda_c_tilde_max_mesh"]:
            self.lmbda_c_tilde_max.rename("Maximal non-local chain stretch",
                                          "lmbda_c_tilde_max")
            self.file_results.write(self.lmbda_c_tilde_max, self.t_val,
                                    encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_lmbda_c_eq_tilde_max_mesh"]:
            for nu_chunk_indx in range(self.nu_chunks_num):
                nu_indx = self.nu_chunks_indx_in_nu_list[nu_chunk_indx]
                lmbda_c_eq_tilde_max___nu_chunk_val = (
                    self.lmbda_c_eq_tilde_max_ufl_fenics_mesh_func(nu_indx)
                )
                local_project(lmbda_c_eq_tilde_max___nu_chunk_val,
                              self.V_DG_scalar, self.lmbda_c_eq_tilde_max_val)

                nu_str = str(self.nu_list[nu_indx])
                name_str = (
                    "Maximal non-local equilibrium chain stretch nu = " + nu_str
                )
                prmtr_str = "lmbda_c_eq_tilde_max nu = " + nu_str

                self.lmbda_c_eq_tilde_max_val.rename(name_str, prmtr_str)
                self.file_results.write(self.lmbda_c_eq_tilde_max_val, self.t_val,
                                        encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_lmbda_nu_tilde_max_mesh"]:
            for nu_chunk_indx in range(self.nu_chunks_num):
                nu_indx = self.nu_chunks_indx_in_nu_list[nu_chunk_indx]
                lmbda_nu_tilde_max___nu_chunk_val = (
                    self.lmbda_nu_tilde_max_ufl_fenics_mesh_func(nu_indx)
                )
                local_project(lmbda_nu_tilde_max___nu_chunk_val, self.V_DG_scalar,
                              self.lmbda_nu_tilde_max_val)

                nu_str = str(self.nu_list[nu_indx])
                name_str = "Maximal non-local segment stretch nu = " + nu_str
                prmtr_str = "lmbda_nu_tilde_max nu = " + nu_str

                self.lmbda_nu_tilde_max_val.rename(name_str, prmtr_str)
                self.file_results.write(self.lmbda_nu_tilde_max_val, self.t_val,
                                        encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_g_mesh"]:
            self.g.rename("Gradient activity", "g")
            self.file_results.write(self.g, self.t_val,
                                    encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_upsilon_c_mesh"]:
            for nu_chunk_indx in range(self.nu_chunks_num):
                nu_indx = self.nu_chunks_indx_in_nu_list[nu_chunk_indx]
                upsilon_c___nu_chunk_val = (
                    self.upsilon_c_ufl_fenics_mesh_func(nu_indx)
                )
                local_project(upsilon_c___nu_chunk_val, self.V_DG_scalar,
                              self.upsilon_c_val)

                nu_str = str(self.nu_list[nu_indx])
                name_str  = "Chain survival nu = " + nu_str
                prmtr_str = "upsilon_c nu = " + nu_str

                self.upsilon_c_val.rename(name_str, prmtr_str)
                self.file_results.write(self.upsilon_c_val, self.t_val,
                                        encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_Upsilon_c_mesh"]:
            Upsilon_c_chunk_val = self.Upsilon_c_ufl_fenics_mesh_func()
            local_project(Upsilon_c_chunk_val, self.V_DG_scalar,
                          self.Upsilon_c_val)
            self.Upsilon_c_val.rename("Average chain survival", "Upsilon_c")
            self.file_results.write(self.Upsilon_c_val, self.t_val,
                                    encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_d_c_mesh"]:
            for nu_chunk_indx in range(self.nu_chunks_num):
                nu_indx = self.nu_chunks_indx_in_nu_list[nu_chunk_indx]
                d_c___nu_chunk_val = self.d_c_ufl_fenics_mesh_func(nu_indx)
                local_project(d_c___nu_chunk_val, self.V_DG_scalar, self.d_c_val)

                nu_str = str(self.nu_list[nu_indx])
                name_str  = "Chain damage nu = " + nu_str
                prmtr_str = "d_c nu = " + nu_str

                self.d_c_val.rename(name_str, prmtr_str)
                self.file_results.write(self.d_c_val, self.t_val,
                                        encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_D_c_mesh"]:
            D_c_chunk_val = self.D_c_ufl_fenics_mesh_func()
            local_project(D_c_chunk_val, self.V_DG_scalar, self.D_c_val)
            self.D_c_val.rename("Average chain damage", "D_c")
            self.file_results.write(self.D_c_val, self.t_val,
                                    encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_u_mesh"]:
            self.u.rename("Displacement", "u")
            self.file_results.write(self.u, self.t_val,
                                    encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_F_mesh"]:
            local_project(self.F, self.V_DG_tensor, self.F_val)
            self.F_val.rename("Deformation gradient", "F")
            self.file_results.write(self.F_val, self.t_val,
                                    encoding=XDMFFile.Encoding.ASCII)
        
        if ppp["save_sigma_mesh"]:
            sigma_chunk_val = self.cauchy_stress_ufl_fenics_mesh_func()
            local_project(sigma_chunk_val, self.V_DG_tensor, self.sigma_val)
            self.sigma_val.rename("Normalized Cauchy stress", "sigma")
            self.file_results.write(self.sigma_val, self.t_val,
                                    encoding=XDMFFile.Encoding.ASCII)

    def user_post_processing(self):
        """
        User post-processing
        """
        pass

    def finalization(self):
        """
        Plot the chunked results from the evolution problem
        """
        pass
    
    def first_pk_stress_ufl_fenics_mesh_func(self):
        """
        Nondimensional first Piola-Kirchhoff stress tensor
        """
        network = self.material

        first_pk_stress_val = Constant(0.0) * self.I
        for nu_indx in range(self.nu_num):
            nu_val = network.nu_list[nu_indx]
            P_nu___nu_val = network.P_nu_list[nu_indx]
            A_nu___nu_val = network.A_nu_list[nu_indx]
            lmbda_c_eq___nu_val = self.lmbda_c * A_nu___nu_val
            lmbda_nu___nu_val = (
                network.composite_ufjc_ufl_fenics_list[nu_indx].lmbda_nu_ufl_fenics_func(lmbda_c_eq___nu_val)
            )
            xi_c___nu_val = (
                network.composite_ufjc_ufl_fenics_list[nu_indx].xi_c_ufl_fenics_func(lmbda_nu___nu_val, lmbda_c_eq___nu_val)
            )
            upsilon_c___nu_val = self.upsilon_c_ufl_fenics_mesh_func(nu_indx)
            first_pk_stress_val += (
                upsilon_c___nu_val * P_nu___nu_val * nu_val * A_nu___nu_val
                * xi_c___nu_val / (3.*self.lmbda_c) * self.F
            )
        first_pk_stress_val += (
            self.first_pk_stress_penalty_term_ufl_fenics_mesh_func()
        )
        return first_pk_stress_val
    
    def first_pk_stress_penalty_term_ufl_fenics_mesh_func(self):
        """
        Penalty term associated with near-incompressibility in the
        nondimensional first Piola-Kirchhoff stress tensor
        """
        dp = self.parameters["deformation"]
        return (
            self.Upsilon_c_cubed_ufl_fenics_mesh_func() * dp["K__G"]
            * (self.J-1) * self.J * self.F_inv.T
        )

    def cauchy_stress_ufl_fenics_mesh_func(self):
        """
        Nondimensional Cauchy stress tensor
        """
        return (
            self.first_pk_stress_ufl_fenics_mesh_func() / self.J * self.F.T
        )
    
    def cauchy_stress_penalty_term_ufl_fenics_mesh_func(self):
        """
        Penalty term associated with near-incompressibility in the
        nondimensional Cauchy stress tensor
        """
        return (
            self.first_pk_stress_penalty_term_ufl_fenics_mesh_func()
            / self.J * self.F.T
        )
    
    def lmbda_c_eq_tilde_ufl_fenics_mesh_func(self, nu_indx):
        """
        Non-local equilibrium chain stretch for chains with a particular
        segment number
        """
        network = self.material

        A_nu___nu_val = network.A_nu_list[nu_indx]
        return self.lmbda_c_tilde * A_nu___nu_val
    
    def lmbda_nu_tilde_ufl_fenics_mesh_func(self, nu_indx):
        """
        Non-local segment stretch for chains with a particular segment
        number
        """
        network = self.material

        lmbda_c_eq_tilde___nu_val = (
            self.lmbda_c_eq_tilde_ufl_fenics_mesh_func(nu_indx)
        )
        return (
            network.composite_ufjc_ufl_fenics_list[nu_indx].lmbda_nu_ufl_fenics_func(lmbda_c_eq_tilde___nu_val)
        )
    
    def lmbda_c_eq_tilde_max_ufl_fenics_mesh_func(self, nu_indx):
        """
        Maximum non-local equilibrium chain stretch for chains with a
        particular segment number
        """
        network = self.material

        A_nu___nu_val = network.A_nu_list[nu_indx]
        return self.lmbda_c_tilde_max * A_nu___nu_val
    
    def lmbda_nu_tilde_max_ufl_fenics_mesh_func(self, nu_indx):
        """
        Maximum non-local segment stretch for chains with a particular
        segment number
        """
        network = self.material

        lmbda_c_eq_tilde_max___nu_val = (
            self.lmbda_c_eq_tilde_max_ufl_fenics_mesh_func(nu_indx)
        )
        return (
            network.composite_ufjc_ufl_fenics_list[nu_indx].lmbda_nu_ufl_fenics_func(lmbda_c_eq_tilde_max___nu_val)
        )
    
    def upsilon_c_ufl_fenics_mesh_func(self, nu_indx):
        """
        Chain degradation for chains with a particular segment number
        """
        dp = self.parameters["deformation"]

        d_c_val = self.d_c_ufl_fenics_mesh_func(nu_indx)
        A_val = (1.-d_c_val)**2 / (1.+d_c_val+4.*d_c_val**2)
        upsilon_c_val = (1.-dp["k_cond_val"]) * A_val + dp["k_cond_val"]
        return conditional(lt(upsilon_c_val, 0.0), 0.0, upsilon_c_val)
    
    def d_c_ufl_fenics_mesh_func(self, nu_indx):
        """
        Chain damage for chains with a particular segment number
        """
        mp = self.parameters["material"]
        lmbda_nu_tilde_max___nu_val = self.lmbda_nu_tilde_max_ufl_fenics_mesh_func(nu_indx)
        x = (
            (lmbda_nu_tilde_max___nu_val-mp["d_c_lmbda_nu_crit_min"])
            / (mp["d_c_lmbda_nu_crit_max"]-mp["d_c_lmbda_nu_crit_min"])
        )
        d_c_val = 70 * x**9 - 315 * x**8 + 540 * x**7 - 420 * x**6 + 126 * x**5
        d_c_val = (
            conditional(gt(lmbda_nu_tilde_max___nu_val, mp["d_c_lmbda_nu_crit_min"]),
                        d_c_val, 0.)
        )
        d_c_val = (
            conditional(lt(lmbda_nu_tilde_max___nu_val, mp["d_c_lmbda_nu_crit_max"]),
                        d_c_val, 1.)
        )
        return d_c_val
    
    def Upsilon_c_ufl_fenics_mesh_func(self):
        """
        Average network chain degradation
        """
        network = self.material

        Upsilon_c_val = Constant(0.0)
        for nu_indx in range(self.nu_num):
            P_nu___nu_val = network.P_nu_list[nu_indx]
            upsilon_c___nu_val = self.upsilon_c_ufl_fenics_mesh_func(nu_indx)
            Upsilon_c_val += P_nu___nu_val * upsilon_c___nu_val
        return Upsilon_c_val / network.P_nu_sum
    
    def Upsilon_c_squared_ufl_fenics_mesh_func(self):
        """
        Average network squared chain degradation
        """
        network = self.material

        Upsilon_c_val = Constant(0.0)
        for nu_indx in range(self.nu_num):
            P_nu___nu_val = network.P_nu_list[nu_indx]
            upsilon_c___nu_val = self.upsilon_c_ufl_fenics_mesh_func(nu_indx)
            Upsilon_c_val += P_nu___nu_val * upsilon_c___nu_val**2
        return Upsilon_c_val / network.P_nu_sum
    
    def Upsilon_c_cubed_ufl_fenics_mesh_func(self):
        """
        Average network cubed chain degradation
        """
        network = self.material

        Upsilon_c_val = Constant(0.0)
        for nu_indx in range(self.nu_num):
            P_nu___nu_val = network.P_nu_list[nu_indx]
            upsilon_c___nu_val = self.upsilon_c_ufl_fenics_mesh_func(nu_indx)
            Upsilon_c_val += P_nu___nu_val * upsilon_c___nu_val**3
        return Upsilon_c_val / network.P_nu_sum
    
    def D_c_ufl_fenics_mesh_func(self):
        """
        Average network chain damage
        """
        network = self.material

        D_c_val = Constant(0.0)
        for nu_indx in range(self.nu_num):
            P_nu___nu_val = network.P_nu_list[nu_indx]
            d_c___nu_val = self.d_c_ufl_fenics_mesh_func(nu_indx)
            D_c_val += P_nu___nu_val * d_c___nu_val
        return D_c_val / network.P_nu_sum
    
    def D_c_squared_ufl_fenics_mesh_func(self):
        """
        Average network squared chain damage
        """
        network = self.material

        D_c_val = Constant(0.0)
        for nu_indx in range(self.nu_num):
            P_nu___nu_val = network.P_nu_list[nu_indx]
            d_c___nu_val = self.d_c_ufl_fenics_mesh_func(nu_indx)
            D_c_val += P_nu___nu_val * d_c___nu_val**2
        return D_c_val / network.P_nu_sum
    
    def D_c_cubed_ufl_fenics_mesh_func(self):
        """
        Average network cubed chain damage
        """
        network = self.material

        D_c_val = Constant(0.0)
        for nu_indx in range(self.nu_num):
            P_nu___nu_val = network.P_nu_list[nu_indx]
            d_c___nu_val = self.d_c_ufl_fenics_mesh_func(nu_indx)
            D_c_val += P_nu___nu_val * d_c___nu_val**3
        return D_c_val / network.P_nu_sum
    
    def lmbda_c_eq_ufl_fenics_mesh_func(self, nu_indx):
        """
        Equilibrium chain stretch for chains with a particular segment
        number
        """
        network = self.material

        A_nu___nu_val = network.A_nu_list[nu_indx]
        return self.lmbda_c * A_nu___nu_val
    
    def lmbda_nu_ufl_fenics_mesh_func(self, nu_indx):
        """
        Segment stretch for chains with a particular segment number
        """
        network = self.material

        lmbda_c_eq___nu_val = self.lmbda_c_eq_ufl_fenics_mesh_func(nu_indx)
        return (
            network.composite_ufjc_ufl_fenics_list[nu_indx].lmbda_nu_ufl_fenics_func(lmbda_c_eq___nu_val)
        )
    
    def g_ufl_fenics_mesh_func(self):
        """
        Gradient activity
        """
        dp = self.parameters["deformation"]
        mp = self.parameters["material"]

        D_c_val = self.D_c_ufl_fenics_mesh_func()
        D_c_val = conditional(gt(D_c_val, 1.0), 1.0, D_c_val)
        n = Constant(mp["n"])
        g_val = (
            0.5 * (cos(DOLFIN_PI*D_c_val**(1/n))+1.) * (1.-dp["k_g_cond_val"])
            + dp["k_g_cond_val"]
        )
        return conditional(lt(g_val, 0.0), 0.0, g_val)
    
    # def H_tilde_ufl_fenics_mesh_func(self):
    #     """
    #     Normalized anisotropic interaction tensor
    #     """
    #     sigma_val = self.cauchy_stress_ufl_fenics_mesh_func()
    #     I_sigma_val = tr(sigma_val)
    #     III_sigma_val = det(sigma_val)

    #     sigma_1_val = (
    #         (I_sigma_val+sqrt(I_sigma_val**2-4.*III_sigma_val+DOLFIN_EPS)) / 2.
    #     )
    #     sigma_2_val = (
    #         (I_sigma_val-sqrt(I_sigma_val**2-4.*III_sigma_val+DOLFIN_EPS)) / 2.
    #     )

    #     sigma_max_val = max_ufl_fenics_mesh_func(sigma_1_val, sigma_2_val)

    #     Z_1_val_dnmntr = sigma_1_val - sigma_2_val
    #     dolfin_eps_sgn = (
    #         conditional(ge(Z_1_val_dnmntr, 0), DOLFIN_EPS, -1*DOLFIN_EPS)
    #     )
    #     Z_1_val_dnmntr = (
    #         conditional(ge(abs(Z_1_val_dnmntr), DOLFIN_EPS),
    #                     Z_1_val_dnmntr, dolfin_eps_sgn)
    #     )
    #     Z_1_val = (sigma_val-sigma_2_val*self.I) / Z_1_val_dnmntr

    #     Z_2_val_dnmntr = sigma_2_val - sigma_1_val
    #     dolfin_eps_sgn = (
    #         conditional(ge(Z_2_val_dnmntr, 0), DOLFIN_EPS, -1*DOLFIN_EPS)
    #     )
    #     Z_2_val_dnmntr = (
    #         conditional(ge(abs(Z_2_val_dnmntr), DOLFIN_EPS),
    #                     Z_2_val_dnmntr, dolfin_eps_sgn)
    #     )
    #     Z_2_val = (sigma_val-sigma_1_val*self.I) / Z_2_val_dnmntr

    #     H_tilde_val = (
    #         (sigma_1_val/sigma_max_val)**2 * Z_1_val
    #         + (sigma_2_val/sigma_max_val)**2 * Z_2_val
    #     )
    #     return (
    #         conditional(ge(abs(Z_1_val_dnmntr), DOLFIN_EPS), H_tilde_val, self.I)
    #     )