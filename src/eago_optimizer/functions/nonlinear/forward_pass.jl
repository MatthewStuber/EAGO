# Copyright (c) 2018: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#############################################################################
# EAGO
# A development environment for robust and global optimization
# See https://github.com/PSORLab/EAGO.jl
#############################################################################
# src/eago_optimizer/evaluator/passes.jl
# Functions used to compute forward pass of nonlinear functions.
#############################################################################

"""
$(FUNCTIONNAME)

Post process set_value operator. By default, performs the affine interval cut on
a MC structure.
"""
function set_value_post(x_values::Vector{Float64}, val::MC{N,T}, lower_variable_bounds::Vector{Float64},
                        upper_variable_bounds::Vector{Float64}) where {N, T<:RelaxTag}

    lower = val.cv
    upper = val.cc
    lower_refinement = true
    upper_refinement = true

    for i = 1:N

        x_val = @inbounds x_values[i]
        cv_val = @inbounds val.cv_grad[i]
        cc_val = @inbounds val.cc_grad[i]

        lower_bound = @inbounds lower_variable_bounds[i]
        upper_bound = @inbounds upper_variable_bounds[i]

        if lower_refinement
            if cv_val > 0.0
                if isinf(lower_bound)
                    !upper_refinement && break
                    lower_refinement = false
                else
                    lower += cv_val*(lower_bound - x_val)
                end
            else
                if isinf(upper_bound)
                    !upper_refinement && break
                    lower_refinement = false
                else
                    lower += cv_val*(upper_bound - x_val)
                end
            end
        end

        if upper_refinement
            if cc_val > 0.0
                if isinf(lower_bound)
                    !lower_refinement && break
                    upper_refinement = false
                else
                    upper += cc_val*(upper_bound - x_val)
                end
            else
                if isinf(upper_bound)
                    !lower_refinement && break
                    upper_refinement = false
                else
                    upper += cc_val*(lower_bound - x_val)
                end
            end
        end
    end

    lower = lower_refinement ? max(lower, val.Intv.lo) : val.Intv.lo
    upper = upper_refinement ? min(upper, val.Intv.hi) : val.Intv.hi

    return MC{N,T}(val.cv, val.cc, Interval{Float64}(lower, upper), val.cv_grad, val.cc_grad, val.cnst)
end

"""
$(FUNCTIONNAME)

Intersects the new set valued operator with the prior and performing `set_value_post` if the flag is...
"""
function overwrite_or_intersect(xMC::MC{N,T}, past_xMC::MC{N,T}, x::Vector{Float64}, lbd::Vector{Float64},
                                ubd::Vector{Float64}, is_post::Bool, is_intersect::Bool)

    if is_post && is_intersect
        return set_value_post(x, xMC ∩ past_xMC, lbd, ubd)
    elseif is_post && !is_intersect
        return set_value_post(x, xMC, lbd, ubd)
    elseif !is_post && is_intersect
        return x ∩ past_xMC
    end
    return xMC
end

function forward_plus_binary!(k::Int64, children_arr::Vector{Int64}, children_idx::UnitRange{Int64},
                              numvalued::Vector{Bool}, numberstorage::Vector{Float64}, setstorage::Vector{V},
                              x::Vector{Float64}, lbd::Vector{Float64}, ubd::Vector{Float64},
                              is_post::Bool, is_intersect::Bool, is_first_eval::Bool) where V

    # get row indices
    idx1 = first(children_idx)
    idx2 = last(children_idx)

    # extract values for argument 1
    arg1_index = @inbounds children_arr[idx1]
    arg1_is_number = @inbounds numvalued[arg1_index]
    if arg1_is_number
        set1 = zero(MC{N,T})
        num1 = @inbounds numberstorage[arg1_index]
    else
        num1 = 0.0
        set1 = @inbounds setstorage[arg1_index]
    end

    # extract values for argument 2
    arg2_index = @inbounds children_arr[idx2]
    arg2_is_number = @inbounds numvalued[arg2_index]
    if arg2_is_number
        num2 = @inbounds numberstorage[arg2_index]
        set2 = zero(MC{N,T})
    else
        set2 = @inbounds setstorage[arg2_index]
        num2 = 0.0
    end

    output_is_number = arg1_is_number && arg2_is_number

    # a + b
    if output_is_number
        @inbounds numberstorage[k] = num1 + num2

    # x + b
    elseif !arg1_is_number && arg2_is_number
        outset = is_first_eval ? (set1 + num2) : plus_kernel(set1, num2, setstorage[k].Intv)

    # a + y
    elseif arg1_is_number && !arg2_is_number
        outset = is_first_eval ? (num1 + set2) : plus_kernel(num1, set2, setstorage[k].Intv)

    # x + y
    else
        outset = is_first_eval ? (set1 + set2) : plus_kernel(set1, set2, setstorage[k].Intv)

    end

    @inbounds numvalued[k] = output_is_number
    if !isnum
        @inbounds setstorage[k] = overwrite_or_intersect(outset, setstorage[k], x, lbd, ubd, is_post, is_intersect)
    end

    return nothing
end

function forward_plus_narity!(k::Int64, children_arr::Vector{Int64}, children_idx::UnitRange{Int64},
                              numvalued::Vector{Bool}, numberstorage::Vector{Float64}, setstorage::Vector{V},
                              x::Vector{Float64}, lbd::Vector{Float64}, ubd::Vector{Float64},
                              is_post::Bool, is_intersect::Bool) where V


    # get row indices
    idx = first(children_idx)

    # extract values for argument 1
    arg_index = @inbounds children_arr[idx]
    output_is_number = @inbounds numvalued[arg1_index]
    if output_is_number
       tmp_set = zero(MC{N,T})
       tmp_num = @inbounds numberstorage[arg_index]
    else
       tmp_num = 0.0
       tmp_set = @inbounds setstorage[arg_index]
    end

    output_is_number = true

    for idx = 2:length(children_idx)
        arg_index = @inbounds children_arr[idx]
        arg_is_number = @inbounds numvalued[arg_index]
        if arg_is_number
            tmp_num += @inbounds numberstorage[arg_index]
        else
            tmp_set += @inbounds setstorage[arg_index]
        end
        output_is_number &= arg_is_number
    end

    @inbounds numvalued[k] = output_is_number
    if output_is_number
        @inbounds numberstorage[k] = tmp_num
    else
        tmp_set += tmp_num
        setstorage[k] = overwrite_or_intersect(tmp_set, setstorage[k], x, lbd, ubd, is_post, is_intersect)
    end

    return nothing
end

function forward_multiply_binary!(k::Int64, children_arr::Vector{Int64}, children_idx::UnitRange{Int64},
                                  numvalued::Vector{Bool}, numberstorage::Vector{Float64}, setstorage::Vector{V},
                                  x::Vector{Float64}, lbd::Vector{Float64}, ubd::Vector{Float64},
                                  is_post::Bool, is_intersect::Bool, is_first_eval::Bool) where V
    # get row indices
    idx1 = first(children_idx)
    idx2 = last(children_idx)

    # extract values for argument 1
    arg1_index = @inbounds children_arr[idx1]
    arg1_is_number = @inbounds numvalued[arg1_index]
    if arg1_is_number
        set1 = zero(MC{N,T})
        num1 = @inbounds numberstorage[arg1_index]
    else
        num1 = 0.0
        set1 = @inbounds setstorage[arg1_index]
    end

    # extract values for argument 2
    arg2_index = @inbounds children_arr[idx2]
    arg2_is_number = @inbounds numvalued[arg2_index]
    if arg2_is_number
        num2 = @inbounds numberstorage[arg2_index]
        set2 = zero(MC{N,T})
    else
        set2 = @inbounds setstorage[arg2_index]
        num2 = 0.0
    end

    output_is_number = arg1_is_number && arg2_is_number

    # a * b
    if output_is_number
        @inbounds numberstorage[k] = num1 * num2

    # x * b
    elseif !arg1_is_number && arg2_is_number
        outset = is_first_eval ? (set1 * num2) : mult_kernel(set1, num2, setstorage[k].Intv)

    # a * y
    elseif arg1_is_number && !arg2_is_number
        outset = is_first_eval ? (num1 * set2) : mult_kernel(num1, set2, setstorage[k].Intv)

    # x * y
    else
        outset = is_first_eval ? (set1 * set2) : mult_kernel(set1, set2, setstorage[k].Intv)

    end

    @inbounds numvalued[k] = output_is_number
    if !isnum
        inbounds setstorage[k] = overwrite_or_intersect(outset, setstorage[k], x_values, lbd, ubd, is_post, is_intersect)
    end

    return nothing
end

function forward_multiply_narity!(k::Int64, children_arr::Vector{Int64}, children_idx::UnitRange{Int64},
                                  numvalued::Vector{Bool}, numberstorage::Vector{Float64}, setstorage::Vector{V},
                                  x::Vector{Float64}, lbd::Vector{Float64}, ubd::Vector{Float64},
                                  is_post::Bool, is_intersect::Bool) where V
    # get row indices
    idx = first(children_idx)

    # extract values for argument 1
    arg_index = @inbounds children_arr[idx]
    output_is_number = @inbounds numvalued[arg1_index]
    if output_is_number
        tmp_set = zero(MC{N,T})
        tmp_num = @inbounds numberstorage[arg_index]
    else
        tmp_num = 0.0
        tmp_set = @inbounds setstorage[arg_index]
    end

    output_is_number = true

    for idx = 2:length(children_idx)
        arg_index = @inbounds children_arr[idx]
        arg_is_number = @inbounds numvalued[arg_index]
        if arg_is_number
            tmp_num *= @inbounds numberstorage[arg_index]
        else
            tmp_set *= @inbounds setstorage[arg_index]
        end
        output_is_number &= arg_is_number
    end

    @inbounds numvalued[k] = output_is_number
    if isnum
        @inbounds numberstorage[k] = tmp_num
    else
       tmp_set *= tmp_num
       setstorage[k] = overwrite_or_intersect(tmp_set, setstorage[k], x_values, lbd, ubd, is_post, is_intersect)
    end

    return nothing
end

function forward_minus!(k::Int64, children_arr::Vector{Int64}, children_idx::UnitRange{Int64},
                        numvalued::Vector{Bool}, numberstorage::Vector{Float64}, setstorage::Vector{V},
                        x::Vector{Float64}, lbd::Vector{Float64}, ubd::Vector{Float64},
                        is_post::Bool, is_intersect::Bool, is_first_eval::Bool) where V

    # get row indices
    idx1 = first(children_idx)
    idx2 = last(children_idx)

    # extract values for argument 1
    arg1_index = @inbounds children_arr[idx1]
    arg1_is_number = @inbounds numvalued[arg1_index]
    if arg1_is_number
        set1 = zero(MC{N,T})
        num1 = @inbounds numberstorage[arg1_index]
    else
        num1 = 0.0
        set1 = @inbounds setstorage[arg1_index]
    end

    # extract values for argument 2
    arg2_index = @inbounds children_arr[idx2]
    arg2_is_number = @inbounds numvalued[arg2_index]
    if arg2_is_number
        num2 = @inbounds numberstorage[arg2_index]
        set2 = zero(MC{N,T})
    else
        set2 = @inbounds setstorage[arg2_index]
        num2 = 0.0
    end

    output_is_number = arg1_is_number && arg2_is_number

    # a - b
    if output_is_number
        @inbounds numberstorage[k] = num1 - num2

    # x - b
    elseif !arg1_is_number && arg2_is_number
        outset = is_first_eval ? (set1 - num2) : minus_kernel(set1, num2, setstorage[k].Intv)

    # a - y
    elseif arg1_is_number && !arg2_is_number
        outset = is_first_eval ? (num1 - set2) : minus_kernel(num1, set2, setstorage[k].Intv)

    # x - y
    else
        outset = is_first_eval ? (set1 - set2) : minus_kernel(set1, set2, setstorage[k].Intv)

    end

    @inbounds numvalued[k] = output_is_number
    if ~isnum
        inbounds setstorage[k] = overwrite_or_intersect(outset, setstorage[k], x_values, lbd, ubd, is_post, is_intersect)
    end

    return nothing
end

function forward_power!(k::Int64, children_arr::Vector{Int64}, children_idx::UnitRange{Int64},
                        numvalued::Vector{Bool}, numberstorage::Vector{Float64}, setstorage::Vector{V},
                        x::Vector{Float64}, lbd::Vector{Float64}, ubd::Vector{Float64},
                        is_post::Bool, is_intersect::Bool, is_first_eval::Bool, ctx::GuardCtx) where V

    # get row indices
    idx1 = first(children_idx)
    idx2 = last(children_idx)

    # extract values for argument 1
    arg1_index = @inbounds children_arr[idx1]
    arg1_is_number = @inbounds numvalued[arg1_index]
    if arg1_is_number
        set1 = zero(MC{N,T})
        num1 = @inbounds numberstorage[arg1_index]
    else
        num1 = 0.0
        set1 = @inbounds setstorage[arg1_index]
    end

    # extract values for argument 2
    arg2_index = @inbounds children_arr[idx2]
    arg2_is_number = @inbounds numvalued[arg2_index]
    if arg2_is_number
        num2 = @inbounds numberstorage[arg2_index]
        set2 = zero(MC{N,T})
    else
        set2 = @inbounds setstorage[arg2_index]
        num2 = 0.0
    end

    # is output a number (by closure of the reals)?
    output_is_number = arg1_is_number && arg2_is_number
    @inbounds numvalued[k] = output_is_number

    # x^1 = x
    if num2 === 1.0
        if arg1_is_number
            @inbounds numberstorage[k] = num1
        else
            @inbounds setstorage[k] = set1
        end
        return nothing

    # x^0 = 1
    elseif num2 === 0.0
        if arg1_is_number
            @inbounds numberstorage[k] = 1.0
        else
            @inbounds setstorage[k] = zero(MC{N,T})
        end
        return nothing

    else
        # a^b
        if arg1_is_number && arg2_is_number
            @inbounds numberstorage[k] = num1^num2

        # x^b
        elseif !arg1_is_number && arg2_is_number
            outset = is_first_eval ? pow(set1, num2) : ^(set1, num2, setstorage[k].Intv)

        # a^y
        elseif arg1_is_number  && !arg2_is_number
            outset = is_first_eval ? overdub(ctx, pow, num1, set2) : overdub(ctx, ^, num1, set2, setstorage[k].Intv)

        # x^y
        elseif !arg1_is_number && !arg2_is_number
            outset = is_first_eval ? overdub(ctx, pow, set1, set2) : overdub(ctx, ^, set1, set2, setstorage[k].Intv)

        end
    end

    if !output_is_number
        setstorage[k] = overwrite_or_intersect(outset, setstorage[k], x_values, lbd, ubd, is_post, is_intersect)
    end

    return nothing
end

function forward_divide!(k::Int64, children_arr::Vector{Int64}, children_idx::UnitRange{Int64},
                         numvalued::Vector{Bool}, numberstorage::Vector{Float64}, setstorage::Vector{V},
                         x::Vector{Float64}, lbd::Vector{Float64}, ubd::Vector{Float64},
                         is_post::Bool, is_intersect::Bool, is_first_eval::Bool, ctx::GuardCtx) where V

    # get row indices
    idx1 = first(children_idx)
    idx2 = last(children_idx)

    # extract values for argument 1
    arg1_index = @inbounds children_arr[idx1]
    arg1_is_number = @inbounds numvalued[arg1_index]
    if arg1_is_number
        set1 = zero(MC{N,T})
        num1 = @inbounds numberstorage[arg1_index]
    else
        num1 = 0.0
        set1 = @inbounds setstorage[arg1_index]
    end

    # extract values for argument 2
    arg2_index = @inbounds children_arr[idx2]
    arg2_is_number = @inbounds numvalued[arg2_index]
    if arg2_is_number
        num2 = @inbounds numberstorage[arg2_index]
        set2 = zero(MC{N,T})
    else
        set2 = @inbounds setstorage[arg2_index]
        num2 = 0.0
    end

    # is output a number (by closure of the reals)?
    output_is_number = arg1_is_number && arg2_is_number
    @inbounds numvalued[k] = output_is_number

    # a/b
    if output_is_number
        @inbounds numberstorage[k] = num1/num2

    # x/b
    elseif !arg1_is_number && arg2_is_number
        outset = is_first_eval ? set1/num2 : div_kernel(set1, num2, setstorage[k].Intv)

    # a/y
    elseif arg1_is_number && !arg2_is_number
        outset = is_first_eval ? num1/set2 : div_kernel(num1, set2, setstorage[k].Intv)

    # x/y
    else
        outset = is_first_eval ? set1/set2 : div_kernel(set1, set2, setstorage[k].Intv)

    end

    @inbounds setstorage[k] = overwrite_or_intersect(outset, setstorage[k], x_values, lbd, ubd, is_post, is_intersect)

    return nothing
end

function forward_user_multivariate!(k::Int64, children_arr::Vector{Int64}, children_idx::UnitRange{Int64},
                                    numvalued::Vector{Bool}, numberstorage::Vector{Float64}, setstorage::Vector{MC{N,T}},
                                    x::Vector{Float64}, lbd::Vector{Float64}, ubd::Vector{Float64},
                                    is_post::Bool, is_intersect::Bool, ctx::GuardCtx,
                                    user_operators::JuMP._Derivatives.UserOperatorRegistry,
                                    set_mv_buffer::Vector{MC{N,T}}, num_mv_buffer::Vector{Float64}) where {N, T<:RelaxTag}

        n = length(children_idx)
        evaluator = user_operators.multivariate_operator_evaluator[op - JuMP._Derivatives.USER_OPERATOR_ID_START + 1]
        set_input = view(set_mv_buffer, 1:n)
        num_input = view(num_mv_buffer, 1:n)
        fill!(num_input, -Inf)

        buffer_count = 1
        output_is_number = true
        for c_idx in children_idx
            arg_index = @inbounds children_arr[c_idx]
            arg_is_number = @inbounds numvalued[arg_index]
            if arg_is_number
                @inbounds num_input[buffer_count] = numberstorage[arg_index]
            else
                @inbounds set_input[buffer_count] = setstorage[arg_index]
            end
            buffer_count += 1
        end

        if output_is_number
            numberstorage[k] = MOI.eval_objective(evaluator, num_input)
        else
            for i = 1:(buffer_count - 1)
                if !isinf(@inbounds num_input[i])
                    @inbounds set_input[buffer_count] = V(num_input[buffer_count])
                end
            end
            outset = Cassette.overdub(ctx, MOI.eval_objective, evaluator, set_input)
            setstorage[k] = overwrite_or_intersect(outset, setstorage[k], x, lbd, ubd, is_post, is_intersect)
        end
        @inbounds numvalued[k] = output_is_number
    else
        error("Unsupported operation $(operators[op])")
    end

    return nothing
end

function forward_univariate_number!(k::Int64, op::Int64, numvalued::Vector{Bool}, numberstorage::Vector{Float64})

    tmp_num = @inbounds numberstorage[child_idx]
    outnum = eval_univariate_set(op, tmp_num)

    @inbounds numberstorage[k] = outnum
    @inbounds numvalued[k] = true

    return nothing
end

function forward_univariate_tiepnt_1!(k::Int64, child_idx::Int64, setstorage::Vector{V},
                                      x::Vector{Float64}, lbd::Vector{Float64}, ubd::Vector{Float64},
                                      tpdict::Dict{Int64, Tuple{Int64,Int64,Int64,Int64}},
                                      tp1storage::Vector{Float64}, tp2storage::Vector{Float64},
                                      is_post::Bool, is_intersect::Bool, is_first_eval::Bool, ctx::GuardCtx) where V

    tmp_set = @inbounds setstorage[child_idx]

    tidx1, tidx2 = tpdict[k]
    tp1 = @inbounds tp1storage[tindx1]
    tp2 = @inbounds tp2storage[tindx2]
    new_tie_points = tp1 === Inf

    outset, tp1, tp2 = Cassette.overdub(ctx, single_tp_set, op, tmp_set, setstorage[k], tp1, tp2, first_eval_flag)

    if new_tie_points
        @inbounds tp1storage[tindx] = tp1
        @inbounds tp1storage[tindx] = tp2
    end

    setstorage[k] = overwrite_or_intersect(outset, setstorage[k], x, lbd, ubd, is_post, is_intersect)
    return nothing
end

function forward_univariate_tiepnt_2!(k::Int64, child_idx::Int64, setstorage::Vector{V},
                                      x::Vector{Float64}, lbd::Vector{Float64}, ubd::Vector{Float64},
                                      tpdict::Dict{Int64, Tuple{Int64,Int64,Int64,Int64}},
                                      tp1storage::Vector{Float64}, tp2storage::Vector{Float64},
                                      tp3storage::Vector{Float64}, tp4storage::Vector{Float64},
                                      is_post::Bool, is_intersect::Bool, is_first_eval::Bool, ctx::GuardCtx) where V

    tmp_set = @inbounds setstorage[child_idx]

    # retreive previously calculated tie-points
    # These are re-initialize to Inf for each box
    tidx1, tidx2, tidx3, tidx4 = tpdict[k]
    tp1 = @inbounds tp1storage[tidx1]
    tp2 = @inbounds tp2storage[tidx2]
    tp3 = @inbounds tp3storage[tidx3]
    tp4 = @inbounds tp4storage[tidx4]

    new_tie_points = tp1 === Inf

    # Perform an evaluation of the univariate function overdubbed with Cassette.jl
    outset, tp1, tp2, tp3, tp4 = Cassette.overdub(ctx, double_tp_set, op, tmp_set, setstorage[k], tp1, tp2, tp3, tp4, first_eval_flag)

    # Store new tiepoints if new evaluation
    if new_tie_points
        @inbounds tp1storage[tidx1] = tp1
        @inbounds tp2storage[tidx2] = tp2
        @inbounds tp3storage[tidx3] = tp3
        @inbounds tp4storage[tidx4] = tp4
    end

    setstorage[k] = overwrite_or_intersect(outset, setstorage[k], x_values, lbd, ubd, is_post, is_intersect)
    return nothing
end

function forward_univariate_user!(k::Int64, op::Int64, child_idx::Int64, setstorage::Vector{V},
                                  x::Vector{Float64}, lbd::Vector{Float64}, ubd::Vector{Float64},
                                  is_post::Bool, is_intersect::Bool, is_first_eval::Bool, ctx::GuardCtx,
                                  user_operators) where V

    userop = op - JuMP._Derivatives.USER_UNIVAR_OPERATOR_ID_START + 1
    @inbounds f = user_operators.univariate_operator_f[userop]

    if arg_is_number
        tmp_num = @inbounds setstorage[child_idx]
        outnum = f(tmp_num)
        @inbounds numberstorage[k] = outnum
    else
        tmp_set = @inbounds setstorage[child_idx]
        outnum = Cassette.overdub(ctx, f, tmp_set)
        setstorage[k] = overwrite_or_intersect(outnum, setstorage[k], x_values, lbd, ubd, is_post, is_intersect)
    end

    return nothing
end

function forward_univariate_other!(k::Int64, op::Int64, child_idx::Int64, setstorage::Vector{V},
                                   x::Vector{Float64}, lbd::Vector{Float64}, ubd::Vector{Float64},
                                   is_post::Bool, is_intersect::Bool, is_first_eval::Bool, ctx::GuardCtx) where V

    tmp_set = @inbounds setstorage[child_idx]
    outset = Cassette.overdub(ctx, eval_univariate_set, op, tmp_set)
    setstorage[k] = overwrite_or_intersect(outset, setstorage[k], x_values, lbd, ubd, is_post, is_intersect)

    return nothing
end

const id_to_operator = Dict(value => key for (key, value) in JuMP.univariate_operator_to_id)
function forward_pass_kernel!(nd::Vector{JuMP.NodeData}, adj::SparseMatrixCSC{Bool,Int64}, x::Vector{Float64},
                              lbd::Vector{Float64}, ubd::Vector{Float64},
                              setstorage::Vector{MC{N,T}}, numberstorage::Vector{Float64}, numvalued::Vector{Bool},
                              tpdict::Dict{Int64,Tuple{Int64,Int64,Int64,Int64}}, tp1storage::Vector{Float64},
                              tp2storage::Vector{Float64}, tp3storage::Vector{Float64}, tp4storage::Vector{Float64},
                              user_operators::JuMP._Derivatives.UserOperatorRegistry, subexpression_isnum::Vector{Bool},
                              subexpr_values_flt::Vector{Float64}, subexpr_values_set, num_mv_buffer::Vector{Float64},
                              set_mv_buffer::Vector{MC{N,T}}, ctx::GuardCtx, is_post::Bool, is_intersect::Bool,
                              is_first_eval::Bool, cv_grad_buffer::Vector{Float64}, cc_grad_buffer::Vector{Float64}) where {N, T<:RelaxTag}

    children_arr = rowvals(adj)

    for k = length(nd):-1:1
        nod = @inbounds nd[k]
        op = nod.index

        if nod.nodetype == JuMP._Derivatives.VALUE
            @inbounds numvalued[k] = true

        elseif nod.nodetype == JuMP._Derivatives.PARAMETER
            @inbounds numvalued[k] = true

        elseif nod.nodetype == JuMP._Derivatives.VARIABLE
            isa_number = @inbounds xtype[op]
            @inbounds numvalued[k] = isa_number
            if isa_number
                @inbounds numberstorage[k] = x[op]
            else
                xMC = MC{N,T}(x[op], Interval{Float64}(lbd[op], ubd[op]), Val{N}())
                @inbounds setstorage[k] = is_first_eval ? xMC : (xMC ∩ setstorage[k])
            end

        elseif nod.nodetype == JuMP._Derivatives.SUBEXPRESSION
            isa_number = @inbounds subexpression_is_number[op]
            if isa_number
                @inbounds numberstorage[k] = subexpr_values_flt[op]
            else
                copy_subexpression_value!(k, op, setstorage, subexpr_values_set, cv_grad_buffer, cc_grad_buffer)
            end
            @inbounds numvalued[k] = isa_number

        elseif nod.nodetype == JuMP._Derivatives.CALL

            @inbounds children_idx = nzrange(adj, k)
            n_children = length(children_idx)

            # :+ with arity two or greater
            if op === 1
                n = length(children_idx)
                if n === 2
                    forward_plus_binary!(k, children_arr, children_idx, numvalued, numberstorage,
                                         setstorage, x, lbd, ubd, is_post, is_intersect, is_first_eval)
                else
                    forward_plus_narity!(k, children_arr, children_idx, numvalued, numberstorage,
                                         setstorage, x, lbd, ubd, is_post, is_intersect)
                end

            # :- with arity two
            elseif op === 2
                forward_minus!(k, children_arr, children_idx, numvalued, numberstorage,
                               setstorage, x, lbd, ubd, is_post, is_intersect, is_first_eval)

            # :* with arity two or greater
            elseif op === 3
                n = length(children_idx)
                if n === 2
                    forward_multiply_binary!(k, children_arr, children_idx, numvalued,
                                             numberstorage, setstorage, x, lbd, ubd, is_post,
                                             is_intersect, is_first_eval)
                else
                    forward_multiply_narity!(k, children_arr, children_idx, numvalued,
                                             numberstorage, setstorage, x, lbd, ubd,
                                             is_post, is_intersect)
                end

            # :^
            elseif op === 4
                forward_power!(k, children_arr, children_idx, numvalued, numberstorage,
                               setstorage, x, lbd, ubd, is_post, is_intersect, is_first_eval, ctx)

            # :/
            elseif op === 5
                forward_divide!(k, children_arr, children_idx, numvalued, numberstorage,
                                setstorage, x, lbd, ubd, is_post, is_intersect, is_first_eval, ctx)

            # user multivariate function
            elseif op >= JuMP._Derivatives.USER_OPERATOR_ID_START
                forward_user_multivariate!(k, children_arr, children_idx, numvalued, numberstorage,
                                           setstorage, x, lbd, ubd, is_post, is_intersect, ctx,
                                           user_operators, set_mv_buffer, num_mv_buffer)

        elseif nod.nodetype == JuMP._Derivatives.CALLUNIVAR

            # checks to see if operator is a number
            @inbounds arg_idx = children_arr[adj.colptr[k]]
            arg_is_number = @inbounds numvalued[arg_idx]
            @inbounds numvalued[k] = arg_is_number

            # performs univariate operators on number valued inputs
            if op >= JuMP._Derivatives.USER_UNIVAR_OPERATOR_ID_START
                forward_univariate_user!(k, op, child_idx, setstorage, x, lbd, ubd, is_post,
                                         is_intersect, is_first_eval, ctx, user_operators)

            elseif arg_is_number
                forward_univariate_number!(k, op, numvalued, numberstorage)

            # performs set valued operators that require a single tiepoint calculation
            elseif single_tp(op)
                forward_univariate_tiepnt_1!(k, child_idx, setstorage, x, lbd, ubd, tpdict,
                                             tp1storage, tp2storage, is_post, is_intersect,
                                             is_first_eval, ctx)

            # performs set valued operators that require two tiepoint calculations
            elseif double_tp(op)
                forward_univariate_tiepnt_2!(k, child_idx, setstorage, x, lbd, ubd, tpdict,
                                             tp1storage, tp2storage, tp3storage, tp4storage,
                                             is_post, is_intersect, is_first_eval, ctx)

            # performs set valued operator on other functions in base library
            else
                forward_univariate_other!(k, op, child_idx, setstorage, x, lbd, ubd, is_post,
                                          is_intersect, is_first_eval, ctx)

            end

        else
            error("Unrecognized node type $(nod.nodetype).")

        end
    end

    return nothing
end