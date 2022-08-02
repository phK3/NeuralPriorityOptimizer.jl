
# Include the optimizer as well as supporting packages
using NeuralPriorityOptimizer
using NeuralVerification
using LazySets
using LinearAlgebra

const NV = NeuralVerification

@assert Threads.nthreads()==1 "for benchmarking threads must be 1"
LinearAlgebra.BLAS.set_num_threads(1)


"""
    test_acas_network(index1, index2, params, solver; split, concrete_sample)

Optimize ACAS network index1-index2 on property 1 with solver parameters given by params.
"""
function optimize_acas_network_deep_poly(index1, index2, params, solver; split=split_largest_interval, concrete_sample=:Center)
    network_name = string("ACASXU_experimental_v2a_", index1, "_", index2, ".nnet")
    # Read in the network. Named CAS so as not to confuse with the official ACAS Xu tables.
    network_file = string(@__DIR__, "/../networks/CAS/", network_name)
    network = read_nnet(network_file)

    # Define your input and output sets
    input_set, output_set = get_acas_sets(1)

    # maximize output 1
    coeffs = zeros(5)
    coeffs[1] = 1.

    time = @elapsed x_star, lower_bound, upper_bound, steps = optimize_linear_deep_poly(network, input_set, coeffs, params, 
                                                                                        solver=solver, split=split, concrete_sample=concrete_sample)

    # Print your results
    println("Elapsed time: ", time)
    println("Interval: ", [lower_bound, upper_bound])
    println("Steps: ", steps)

    return lower_bound, upper_bound, time, steps
end


"""
    test_acas_network(index1, index2, params, solver; split, concrete_sample)

Optimize ACAS network index1-index2 on property 1 with solver parameters given by params.
"""
function optimize_acas_network_ai2z(index1, index2, params)
    network_name = string("ACASXU_experimental_v2a_", index1, "_", index2, ".nnet")
    # Read in the network. Named CAS so as not to confuse with the official ACAS Xu tables.
    network_file = string(@__DIR__, "/../networks/CAS/", network_name)
    network = read_nnet(network_file)

    # Define your input and output sets
    input_set, output_set = get_acas_sets(1)

    # maximize output 1
    coeffs = zeros(5)
    coeffs[1] = 1.

    time = @elapsed x_star, lower_bound, upper_bound, steps = optimize_linear(network, input_set, coeffs, params, solver=Ai2z())

    # Print your results
    println("Elapsed time: ", time)
    println("Interval: ", [lower_bound, upper_bound])
    println("Steps: ", steps)

    return lower_bound, upper_bound, time, steps
end


function print_results(lower_bounds, upper_bounds, times, steps, max_index_1, max_index_2, stop_gap)
    # Nicely formatted printout of the tests
    for i = 1:max_index_1
        for j = 1:max_index_2
            println("Network: ", (i, j))
            println("    bounds: ", (lower_bounds[i, j], upper_bounds[i, j]), "  time: ", times[i, j])
            println("    gap: ", upper_bounds[i, j] - lower_bounds[i, j])
            println("    steps: ", steps[i, j])
        end
    end
end


# Each line will look like:
# property_number, network_index_1-network_index_2, SAT UNSAT or Inconclusive, lower_bound, upper_bound, time, steps
# this will overwrite a file if one already exists
function write_results(filename, lower_bounds, upper_bounds, times, steps, max_index_1, max_index_2, stop_gap)
    open(filename, "w") do f
        println(f, "network,gap,lower_bound,upper_bound,time,steps")
        # k, j, i at the end of the line to iterate like a for loop with the outermost i
        [writeline(f, j, k, lower_bounds[j, k], upper_bounds[j, k], times[j, k], steps[j, k], stop_gap) for k=1:max_index_2, j=1:max_index_1]
    end
end

# Write an individual line
function writeline(file, index_1, index_2, lower_bound, upper_bound, time, steps, stop_gap)
    gap = upper_bound - lower_bound
    println(file, string(index_1, "-", index_2, ",", gap, ",", lower_bound, ",", upper_bound, ",", time, ",", steps))
end


function run_acas_optimization_deep_poly(solver, max_index_1, max_index_2, filename;
                                max_steps=200000, timeout=60., split=NV.split_important_interval,
                                concrete_sample=:BoundsMaximizer, verbosity=0, stop_gap=1e-4)

    params = PriorityOptimizerParameters(max_steps=max_steps, verbosity=verbosity,
                                            timeout=timeout, stop_gap=stop_gap, stop_frequency=1) # added by me

    full_time = @elapsed begin
        lower_bounds = Array{Float64, 2}(undef, 5, 9)
        upper_bounds = Array{Float64, 2}(undef, 5, 9)
        times = Array{Float64, 2}(undef, 5, 9)
        steps = Array{Integer, 2}(undef, 5, 9)
        for i = 1:max_index_1
            for j = 1:max_index_2
                println("Network ", i, "-", j)
                lower_bounds[i, j], upper_bounds[i, j], times[i, j], steps[i, j] = optimize_acas_network_deep_poly(i, j, params, solver, split=split, concrete_sample=concrete_sample)
                println()
            end
        end
    end

    println("Max steps: ", max_steps)
    println("Full time: ", full_time)

    print_results(lower_bounds, upper_bounds, times, steps, max_index_1, max_index_2, params.stop_gap)
    write_results(filename, lower_bounds, upper_bounds, times, steps, max_index_1, max_index_2, params.stop_gap)
end


function run_acas_optimization_ai2z(max_index_1, max_index_2, filename;
    max_steps=200000, timeout=60., verbosity=0, stop_gap=1e-4)

    params = PriorityOptimizerParameters(max_steps=max_steps, verbosity=verbosity,
                    timeout=timeout, stop_gap=stop_gap, stop_frequency=100) # added by me

    full_time = @elapsed begin
        lower_bounds = Array{Float64, 2}(undef, 5, 9)
        upper_bounds = Array{Float64, 2}(undef, 5, 9)
        times = Array{Float64, 2}(undef, 5, 9)
        steps = Array{Integer, 2}(undef, 5, 9)
        for i = 1:max_index_1
            for j = 1:max_index_2
                println("Network ", i, "-", j)
                lower_bounds[i, j], upper_bounds[i, j], times[i, j], steps[i, j] = optimize_acas_network_ai2z(i, j, params)
                println()
            end
        end
    end

    println("Max steps: ", max_steps)
    println("Full time: ", full_time)

    print_results(lower_bounds, upper_bounds, times, steps, max_index_1, max_index_2, params.stop_gap)
    write_results(filename, lower_bounds, upper_bounds, times, steps, max_index_1, max_index_2, params.stop_gap)
end



###
# Setup your parameters and then run the tests
###
filename_deep_poly=string(@__DIR__, "/../results/CAS/acas_fullrun_optimization_onethread_deep_poly_gap_1e-2.csv")
filename_zope=string(@__DIR__, "/../results/CAS/acas_fullrun_optimization_onethread_zope_gap_1e-2.csv")

# was commented out before, but printing doesn't work without it
max_steps = 2000000
timeout = 60.
max_index_1 = 5
max_index_2 = 9
#max_index_1 = 2
#max_index_2 = 2
stop_gap = 1e-2

solver = DPNeurifyFV(max_vars=15, method=:DeepPolyRelax)
#solver = DPNeurifyFV(max_vars=0, method=:DeepPolyRelax)
split = NV.split_important_interval
concrete_sample = :BoundsMaximizer

#### just for precompilation
println("precompilation ...")
params = PriorityOptimizerParameters(max_steps=5, timeout=10.)
optimize_acas_network_deep_poly(1, 1, params, solver, split=split, concrete_sample=concrete_sample)
optimize_acas_network_ai2z(1, 2, params)

println("starting evaluation")
println("-- ZoPE --")
run_acas_optimization_ai2z(max_index_1, max_index_2, filename_zope, max_steps=max_steps, timeout=timeout, stop_gap=stop_gap)

println("-- DeepPoly --")
run_acas_optimization_deep_poly(solver, max_index_1, max_index_2, filename_deep_poly, max_steps=max_steps, timeout=timeout, split=split,
                                concrete_sample=concrete_sample, stop_gap=stop_gap)