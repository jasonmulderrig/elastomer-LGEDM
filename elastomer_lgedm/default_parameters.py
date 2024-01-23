# import necessary libraries
from dolfin import *
import numpy as np
from scipy import constants

def default_parameters():

    p = Parameters("user_parameters")
    subset_list = [
        "pre_processing",
        "problem",
        "material",
        "fem",
        "deformation",
        "post_processing"
    ]
    for subparset in subset_list:
        subparset_is = eval("default_"+subparset+"_parameters()")
        p.add(subparset_is)
    return p

def default_pre_processing_parameters():
    
    pre_processing = Parameters("pre_processing")

    pre_processing.add("form_compiler_optimize", True)
    pre_processing.add("form_compiler_cpp_optimize", True)
    pre_processing.add("form_compiler_representation", "uflacs")
    pre_processing.add("form_compiler_quadrature_degree", 4)

    return pre_processing

def default_problem_parameters():
    
    problem = Parameters("problem")

    problem.add("meshtype", "crack")

    return problem

def default_material_parameters():
    
    material = Parameters("material")

    # Fundamental material constants
    k_B = constants.value(u"Boltzmann constant") # J/K
    N_A = constants.value(u"Avogadro constant") # 1/mol
    h = constants.value(u"Planck constant") # J/Hz
    hbar = h / (2*np.pi) # J*sec
    T = 298 # absolute room temperature, K
    beta = 1. / (k_B*T) # 1/J
    omega_0 = 1. / (beta*hbar) # J/(J*sec) = 1/sec
    zeta_nu_char = 300
    kappa_nu = 2300
    nu_b = "None"
    zeta_b_char = "None"
    kappa_b = "None"

    material.add("k_B", k_B)
    material.add("N_A", N_A)
    material.add("h", h)
    material.add("hbar", hbar)
    material.add("T", T)
    material.add("beta", beta)
    material.add("omega_0", omega_0)
    material.add("zeta_nu_char", zeta_nu_char)
    material.add("kappa_nu", kappa_nu)
    material.add("nu_b", nu_b)
    material.add("zeta_b_char", zeta_b_char)
    material.add("kappa_b", kappa_b)

    # Network-level damage
    d_c_lmbda_nu_crit_min = 1.0001
    d_c_lmbda_nu_crit_max = 1.005

    material.add("d_c_lmbda_nu_crit_min", d_c_lmbda_nu_crit_min)
    material.add("d_c_lmbda_nu_crit_max", d_c_lmbda_nu_crit_max)

    # Non-local interaction length scale
    material.add("l_nl", 0.01)

    # Interaction intensity
    material.add("n", 1)

    # Define the chain segment number statistics in the network
    material.add("nu_distribution", "itskov")

    nu_list = [nu for nu in range(5, 16)] # nu = 5 -> nu = 15
    nu_min = min(nu_list)
    nu_max = max(nu_list)
    nu_num = len(nu_list)
    nu_bar = 8
    Delta_nu = nu_bar - nu_min

    for nu_indx in range(nu_num):
        material.add("nu_indx_"+str(nu_indx)+"_nu_val", nu_list[nu_indx])
    material.add("nu_min", nu_min)
    material.add("nu_max", nu_max)
    material.add("nu_num", nu_num)
    material.add("nu_bar", nu_bar)
    material.add("Delta_nu", Delta_nu)

    # Define chain segment numbers to chunk during deformation
    nu_chunks_list = nu_list[::2] # nu = 5, nu = 7, ..., nu = 15
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
    nu_chunks_color_list = [
        'orange', 'blue', 'green', 'red', 'purple', 'brown'
    ]

    for nu_chunk_indx in range(nu_chunks_num):
        material.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_nu_val",
                     nu_chunks_list[nu_chunk_indx])
        material.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_indx_val_in_nu_list",
                     nu_chunks_indx_in_nu_list[nu_chunk_indx])
        material.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_label",
                     nu_chunks_label_list[nu_chunk_indx])
        material.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_color",
                     nu_chunks_color_list[nu_chunk_indx])

    material.add("nu_chunks_num", nu_chunks_num)

    return material

def default_fem_parameters():
    
    fem = Parameters("fem")

    fem.add("u_degree", 2)
    fem.add("scalar_prmtr_degree", 1)
    fem.add("tensor_prmtr_degree", 0)
    fem.add("quadrature_degree", 4)
    
    return fem

def default_deformation_parameters():

    deformation = Parameters("deformation")

    deformation.add("deformation_type", "uniaxial")

    # general network deformation parameters
    deformation.add("K__G", 100)
    
    # conditioning parameters    
    deformation.add("k_cond_val", 1e-4)
    deformation.add("k_g_cond_val", 1e-4)

    # timing parameters
    deformation.add("strain_rate", 0.1) # 1/sec
    deformation.add("t_min", 0.) # sec
    deformation.add("t_max", 15.) # sec
    deformation.add("t_step", 0.1) # sec
    deformation.add("t_step_chunk_num", 1)

    return deformation

def default_post_processing_parameters():
    
    post_processing = Parameters("post_processing")

    ext = "xdmf"
    post_processing.add("ext", ext)
    post_processing.add("file_results", "results"+"."+ext)

    post_processing.add("rewrite_function_mesh", False)
    post_processing.add("flush_output", True)
    post_processing.add("functions_share_mesh", True)

    post_processing.add("save_lmbda_c_mesh", True)
    post_processing.add("save_lmbda_c_eq_mesh", False)
    post_processing.add("save_lmbda_nu_mesh", False)
    post_processing.add("save_lmbda_c_tilde_mesh", True)
    post_processing.add("save_lmbda_c_eq_tilde_mesh", False)
    post_processing.add("save_lmbda_nu_tilde_mesh", False)
    post_processing.add("save_lmbda_c_tilde_max_mesh", True)
    post_processing.add("save_lmbda_c_eq_tilde_max_mesh", False)
    post_processing.add("save_lmbda_nu_tilde_max_mesh", False)
    post_processing.add("save_g_mesh", False)
    post_processing.add("save_upsilon_c_mesh", False)
    post_processing.add("save_Upsilon_c_mesh", True)
    post_processing.add("save_d_c_mesh", False)
    post_processing.add("save_D_c_mesh", True)
    
    post_processing.add("save_u_mesh", True)

    post_processing.add("save_F_mesh", True)
    post_processing.add("save_sigma_mesh", True)

    post_processing.add("axes_linewidth", 1.0)
    post_processing.add("font_family", "sans-serif")
    post_processing.add("text_usetex", True)
    post_processing.add("ytick_right", True)
    post_processing.add("ytick_direction", "in")
    post_processing.add("xtick_top", True)
    post_processing.add("xtick_direction", "in")
    post_processing.add("xtick_minor_visible", True)

    return post_processing
