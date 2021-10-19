
using NeuralPriorityOptimizer
using NeuralVerification
using LazySets


function optimize_acas(index1, index2, property_index, output_index;
                        stop_gap=0.001, stop_frequency=1, max_steps=200000, printing=true)

    network_name = string("ACASXU_experimental_v2a_", index1, "_", index2, ".nnet")
    # Read in the network. Named CAS so as not to confuse with the official ACAS Xu tables.
    network_file = string(@__DIR__, "/../networks/CAS/", network_name)
    acas = read_nnet(network_file)

    # Define your input and output sets
    input_set, output_set = get_acas_sets(property_index)

    params = PriorityOptimizerParameters(max_steps=max_steps, stop_frequency=stop_frequency,
                                verbosity=0, stop_gap=stop_gap)

    coeffs = zeros(5)
    coeffs[output_index] = 1.

    if printing
        println("Optimizing output $output_index of ACAS $index1 $index2 until gap is $stop_gap")
    end

    time = @elapsed x_star, lb, ub, steps = optimize_linear(acas, input_set, coeffs, params)

    if printing
        println("Elapsed time: $time")
        println("Interval: [$lb, $ub]")
        println("Steps: $steps")
    end

    return time, x_star, lb, ub, steps
end


function optimize_gan_controller(lbs, ubs; stop_gap=0.001, stop_frequency=1,
                                max_steps=200000, printing=true)

    # Read in the network
    network_file = string(@__DIR__, "/../networks/GANControl/full_big_uniform.nnet")
    gan = read_nnet(network_file)

    # Define the coefficients for a linear objective
    #coeffs = [-0.74; -0.44]

    # just maximize the single output
    coeffs = [1.; 0]

    if isnothing(lbs)
        lbs = [-1.0, -1.0, -0.93141591135858504, -0.928987424967730113]
    end

    if isnothing(ubs)
        ubs = [1.0, 1.0, -0.9, -0.9]
    end

    input_set = Hyperrectangle(low=lbs, high=ubs)
    maximize = true

    params = PriorityOptimizerParameters(max_steps=max_steps, stop_frequency=stop_frequency,
                                verbosity=0, stop_gap=stop_gap)

    if printing
        println("Optimizing GAN_Controller until gap is $stop_gap")
    end

    time = @elapsed x_star, lb, ub, steps = optimize_linear(gan, input_set, coeffs, params)

    if printing
        println("Elapsed time: $time")
        println("Interval: [$lb, $ub]")
        println("Steps: $steps")
    end

    return time, x_star, lb, ub, steps
end
