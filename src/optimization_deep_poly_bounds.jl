

"""
overestimate_cell : cell -> (val, cell_out) returns overestimate of value as well as cell propagated through the nn
"""
function general_priority_optimization(start_cell::AbstractSymbolicIntervalBounds, overestimate_cell, achievable_value, params::PriorityOptimizerParameters, lower_bound_threshold, upper_bound_threshold, split)
    # set timer
    t_start = time()

    initial_cells = split_multiple_times(start_cell, params.initial_splits)
    # Create your queue, then add your original new_cells
    cells = PriorityQueue(Base.Order.Reverse) # pop off largest first

    # add with priority
    for cell in initial_cells
        res = overestimate_cell(cell)
        #println(res)
        val, sym_out = res
        enqueue!(cells, sym_out, val)
    end

    best_lower_bound = -Inf
    best_x = nothing
    # For n_steps dequeue a cell, split it, and then
    for i = 1:params.max_steps
        cell, value = peek(cells) # peek instead of dequeue to get value, is there a better way?
        @assert value + TOL[] >= best_lower_bound string("Our highest upper bound must be greater than the highest achieved value. Upper bound: ", value, " achieved value: ", best_lower_bound)
        dequeue!(cells)

        # We've passed some threshold on our upper bound that is satisfactory, so we return
        if value < upper_bound_threshold
            println("Returning early because of upper bound threshold")
            return best_x, best_lower_bound, value, i
        end

        # Early stopping
        if params.early_stop
            t_now = time()
            elapsed_time = t_now - t_start

            if i % params.stop_frequency == 0
                input_in_cell, lower_bound = achievable_value(cell)
                if lower_bound > best_lower_bound
                    best_lower_bound = lower_bound
                    best_x = input_in_cell
                end

                if ((value .- lower_bound) <= params.stop_gap
                    || lower_bound > lower_bound_threshold
                    || elapsed_time >= params.timeout)

                    print_progress(params.verbosity, i, lower_bound, best_lower_bound,
                                    lower_bound_threshold, value, cell, elapsed_time)

                    return best_x, best_lower_bound, value, i
                end
            end

            if params.verbosity >= 1 && i % params.print_frequency == 0
                print_progress(params.verbosity, i, lower_bound, best_lower_bound,
                                lower_bound_threshold, value, cell, elapsed_time)
            end

            if params.plotting && i % params.plot_frequency == 0
                push!(params.history_vals, value)
                push!(params.history_lbs, best_lower_bound)
                push!(params.history_ts, elapsed_time)
            end
        end

        new_cells = split(cell)
        # Enqueue each of the new cells
        for new_cell in new_cells
            # If you've made the max objective cell tiny
            # break (otherwise we end up with zero radius cells)
            if radius(new_cell) < NeuralVerification.TOL[]
                # Return a concrete value and the upper bound from the parent cell
                # that was just dequeued, as it must have higher value than all other cells
                # that were on the queue, and they constitute a tiling of the space
                input_in_cell, lower_bound = achievable_value(cell)
                if lower_bound > best_lower_bound
                    best_lower_bound = lower_bound
                    best_x = input_in_cell
                end
                return best_x, best_lower_bound, value, i
            end
            new_value, new_sym_out = overestimate_cell(new_cell)
            enqueue!(cells, new_sym_out, new_value)
        end
    end
    # The largest value in our queue is the approximate optimum
    cell, value = peek(cells)
    input_in_cell, lower_bound = achievable_value(cell)
    if lower_bound > best_lower_bound
        best_lower_bound = lower_bound
        best_x = input_in_cell
    end
    return best_x, best_lower_bound, value, params.max_steps
end


function general_priority_optimization(start_cell::AbstractSymbolicIntervalBounds, relaxed_optimize_cell,
                                       evaluate_objective, params::PriorityOptimizerParameters,
                                       maximize; bound_threshold_realizable=(maximize ? Inf : -Inf),
                                       bound_threshold_approximate=(maximize ? -Inf : Inf),
                                       split=split_largest_interval)
    if maximize
        return general_priority_optimization(start_cell, relaxed_optimize_cell, evaluate_objective,
                                             params, bound_threshold_realizable,
                                             bound_threshold_approximate, split)
    else
        overestimate_cell = cell -> -relaxed_optimize_cell(cell)
        neg_evaluate_objective = cell -> begin
            input, result = evaluate_objective(cell)
            return input, -result
        end
        x, lower, upper, steps = general_priority_optimization(start_cell, overestimate_cell,
                                                               neg_evaluate_objective, params,
                                                               -bound_threshold_realizable,
                                                               -bound_threshold_approximate, split)
        return x, -upper, -lower, steps
    end
end


function optimize_linear_dpb(network, input_set, coeffs, params; maximize=true, solver=DeepPolyBounds())
    min_sign_flip = maximize ? 1.0 : -1.0

    initial_sym = init_symbolic_interval_bounds(network, input_set)

    function approximate_optimize_cell(cell)
        out_cell = forward_network(solver, network, cell)
        val = min_sign_flip * ρ(min_sign_flip .* coeffs, out_cell)
        return val, out_cell
    end

    achievable_value = cell -> (domain(cell).center, compute_linear_objective(network, domain(cell).center, coeffs))
    return general_priority_optimization(initial_sym, approximate_optimize_cell, achievable_value, params, maximize)
end


function optimize_linear_dpfv(network, input_set, coeffs, params; maximize=true, solver=DeepPolyFreshVars(),
                              split=split_largest_interval)
    min_sign_flip = maximize ? 1.0 : -1.0

    initial_sym = init_symbolic_interval_fv(network, input_set, max_vars=solver.max_vars)

    function approximate_optimize_cell(cell)
        out_cell = forward_network(solver, network, cell)
        val = min_sign_flip * ρ(min_sign_flip .* coeffs, out_cell)
        return val, out_cell
    end

    achievable_value = cell -> (domain(cell).center, compute_linear_objective(network, domain(cell).center, coeffs))
    return general_priority_optimization(initial_sym, approximate_optimize_cell, achievable_value, params, maximize, split=split)
end


function get_initial_symbolic_interval(network, input_set, solver::DeepPolyBounds)
    return init_symbolic_interval_bounds(network, input_set)
end

function get_initial_symbolic_interval(network, input_set, solver::DeepPolyFreshVars)
    return init_symbolic_interval_fv(network, input_set, max_vars=solver.max_vars)
end

function get_initial_symbolic_interval(network, input_set, solver::DeepPolyHeuristic)
    return init_symbolic_interval_heur(network, input_set, max_vars=solver.max_vars)
end

function optimize_linear_deep_poly(network, input_set, coeffs, params; maximize=true, solver=DeepPolyHeuristic(),
                              split=split_largest_interval)
    min_sign_flip = maximize ? 1.0 : -1.0

    initial_sym = get_initial_symbolic_interval(network, input_set, solver)

    function approximate_optimize_cell(cell)
        out_cell = forward_network(solver, network, cell)
        val = min_sign_flip * ρ(min_sign_flip .* coeffs, out_cell)
        return val, out_cell
    end

    achievable_value = cell -> (domain(cell).center, compute_linear_objective(network, domain(cell).center, coeffs))
    return general_priority_optimization(initial_sym, approximate_optimize_cell, achievable_value, params, maximize, split=split)
end


function split_largest_interval(s::AbstractSymbolicIntervalBounds)
    largest_dimension = argmax(high(domain(s)) - low(domain(s)))
    return split_symbolic_interval_bounds(s, largest_dimension)
end

function split_multiple_times(cell::AbstractSymbolicIntervalBounds, n; split=split_largest_interval)
    q = Queue{AbstractSymbolicIntervalBounds}()
    enqueue!(q, cell)
    for i in 1:n
        new_cells = split(dequeue!(q))
        enqueue!(q, new_cells[1])
        enqueue!(q, new_cells[2])
    end
    return q
end
