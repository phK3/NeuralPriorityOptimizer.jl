module NeuralPriorityOptimizer

using NeuralVerification, LazySets, Parameters, DataStructures, LinearAlgebra, HDF5, VnnlibParser
using NeuralVerification: compute_output, get_activation, TOL,
                            AbstractSymbolicIntervalBounds, init_symbolic_interval_bounds, # added by me
                            init_symbolic_interval_fv, domain, split_symbolic_interval_bounds, # added by me
                            init_symbolic_interval_heur, merge_into_network, NetworkNegPosIdx, # added by me
                            AsymESIP, init_symbolic_interval_fvheur, split_symbolic_interval_fv_heur,
                            SymbolicIntervalFVHeur, maximizer, minimizer
using Convex, Mosek, MosekTools, JuMP, Gurobi
import JuMP.MOI.OPTIMAL, JuMP.MOI.INFEASIBLE, JuMP.MOI.INFEASIBLE_OR_UNBOUNDED

const NV = NeuralVerification

# From https://githubmemory.com/repo/jump-dev/Gurobi.jl/issues/388
const GUROBI_ENV = Ref{Gurobi.Env}()
function __init__()
    global GUROBI_ENV[] = Gurobi.Env()
end

include("utils.jl")
include("optimization_core.jl")
include("optimization_wrappers.jl")
include("additional_optimizers.jl")

# added by me
include("optimization_deep_poly_bounds.jl")
include("verify_vnnlib.jl")


export general_priority_optimization,
       PriorityOptimizerParameters,
       project_onto_range,
       optimize_linear,
       optimize_linear_gradient_split,
       contained_within_polytope,
       reaches_polytope,
       reaches_polytope_binary,
       reaches_obtuse_polytope,
       range_is_psd,
       max_network_difference,
       optimize_convex_program,
       fgsm,
       pgd,
       repeated_pgd,
       hookes_jeeves,
       get_acas_sets,
       mip_linear_value_only,
       mip_linear_uniform_split,
       optimize_linear_dpb, # added by me
       optimize_linear_dpfv, # added by me
       optimize_linear_deep_poly, # added by me
       contained_within_polytope_deep_poly,  # added by me
       reaches_polytope_deep_poly,  # added by me
       verify_vnnlib # added by me
end # module
