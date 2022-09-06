
## Generate specification for NeuralPriorityOptimizer

"""
Generates specification for NeuralPriorityOptimizer from result value from vnnlib parser.

Returns a list [(input_set, output_set)], where input_set is a Hyperrectangle and
output_set is either of
- HPolytope (for a disjunctive constraint like adversarial robustness)
    -> want to use contained_within_polytope(), if it is violated, there is an adversarial example
- Complement(HPolytope) (for a conjunctive constraint)
    -> want to use reaches_polytope(), if it is satisfied, we have reached the unsafe state

Mixed disjunctive and conjunctive constraints are not fully supported!

"""
function generate_specs(rv)
    # returns list of specs
    # (input_set, output_set)
    specs = []
    for rv_tuple in rv
        l, u, output_specs = rv_tuple

        input_set = Hyperrectangle(low=l, high=u)

        if length(output_specs) == 1
            # a single polytope
            A, b = output_specs[1]

            # spec is SAT, if we reach the polytope
            # we want to use
            output_set = Complement(HPolytope(A, b))
            push!(specs, (input_set, output_set))
        elseif all([length(b) for (A, b) in output_specs] .== ones(Integer, length(output_specs)))
            # disjunction of halfspaces
            A₁, b₁ = output_specs[1]
            Â = zeros(length(output_specs), length(A₁))
            b̂ = zeros(length(output_specs))

            for (i, (A, b)) in enumerate(output_specs)
                Â[i, :] .= vec(A)
                b̂[i] = b[1]
            end

            # we want to use contained within polytope
            # spec is SAT, if we are not contained -> violation is greater than 0
            output_set = HPolytope(-Â, -b̂)

            push!(specs, (input_set, output_set))
        else
            # disjunction of conjunction of halfspaces
            println("WARNING: No efficient verification for disjunction of conjunctions of halfspaces implemented yet!\n
                     Creating multiple sub-problems.\n
                     Check yourself, wether any element of the disjunction is satisfied!")

            for (A, b) in output_specs
                output_set = Complement(HPolytope(A, b))
                push!(specs, (input_set, output_set))
            end
        end

    end

    return specs
end


## directly use specification for verification of property

function verify_vnnlib(network, vnnlib_file, params; solver=nothing, split=NV.split_important_interval, concrete_sample=:BoundsMaximizer, max_vars=20, method=:DeepPolyRelax)
    solver = isnothing(solver) ? DPNeurifyFV(method=method, max_vars=max_vars) : solver

    n_in = size(network.layers[1].weights, 2)
    n_out = length(network.layers[end].bias)

    rv = read_vnnlib_simple(vnnlib_file, n_in, n_out)
    specs = generate_specs(rv)

    for (input_set, output_set) in specs

        if output_set isa AbstractPolytope
            println("Checking if contained within polytope")

            # contained_within_polytope maximizes violation of polytope's constraints
            x_star, lower_bound, upper_bound, steps = contained_within_polytope_deep_poly(network, input_set, output_set, params; solver=solver,
                                                                split=split, concrete_sample=concrete_sample)

            if lower_bound > 0
                # at least one constraint of the polytope is violated! Have found a counterexample
                println("SAT")
            elseif upper_bound <= params.stop_gap
                println("UNSAT")
            else
                println("inconclusive")
            end
        elseif output_set isa Complement{<:Number, <:AbstractPolytope}
            println("Checking if polytope can be reached")

            # reaches_polytope minimizes distance to polytope
            x_star, lower_bound, upper_bound, steps = reaches_polytope_deep_poly(network, input_set, output_set.X, params; solver=solver,
                                                                split=split, concrete_sample=concrete_sample)
            if upper_bound <= params.stop_gap
                # distance to polytope is smaller than allowed! Have found a counterexample
                # TODO: could return x_star as counterexample
                println("SAT")
            elseif lower_bound > 0
                # distance to polytope is larger than 0 (and stop_gap), property is proven
                println("UNSAT")
            else
                println("inconclusive")
            end
        else
            @assert false "No implementation for output_set = $(output_set)"
        end
    end

end
