abstract type AbstractSIPAlgo end

abstract type AbstractSubproblemType end
struct LowerLevel1 <: AbstractSubproblemType end
struct LowerLevel2 <: AbstractSubproblemType end
struct LowerLevel3 <: AbstractSubproblemType end
struct LowerProblem <: AbstractSubproblemType end
struct UpperProblem <: AbstractSubproblemType end
struct ResProblem <: AbstractSubproblemType end

"""
    SIPResult

Structure storing the results of the SIPres algorithm.
"""
mutable struct SIPResult
    iteration_number::Int64
    upper_bound::Float64
    lower_bound::Float64
    feasibility::Bool
    xsol::Vector{Float64}
    psol::Vector{Float64}
    solution_time::Float64
end
SIPResult() = SIPResult(1, Inf, -Inf, true, Float64[], Float64[], 0.0)
SIPResult(nx::Int, np::Int) = SIPResult(1, Inf, -Inf, true, zeros(nx), zeros(np), 0.0)


"""
      SIPProblem

Structure storing problem information for the solution routine.
"""
mutable struct SIPProblem
    x_l::Vector{Float64}
    x_u::Vector{Float64}
    p_l::Vector{Float64}
    p_u::Vector{Float64}

    np::Int64
    nSIP::Int64
    nx::Int64
    sense::Symbol

    init_lower_disc::Vector{Vector{Vector{Float64}}}
    init_upper_disc::Vector{Vector{Vector{Float64}}}

    absolute_tolerance::Float64
    constraint_tolerance::Float64
    iteration_limit::Int64
    initial_eps_g::Float64
    initial_r::Float64

    return_hist::Bool
    header_interval::Int64
    print_interval::Int64
    verbosity::Int64

    local_solver::Bool

    #polyhedral_uncertainty_set
    #ellipsodial_uncertainty_set

    optimizer
    kwargs
end

get_sip_kwargs(s::LowerLevel1, p::SIPProblem) = p.kwargs_llp1
get_sip_kwargs(s::LowerLevel2, p::SIPProblem) = p.kwargs_llp2
get_sip_kwargs(s::LowerProblem, p::SIPProblem) = p.kwargs_lbd
get_sip_kwargs(s::UpperProblem, p::SIPProblem) = p.kwargs_ubd
get_sip_kwargs(s::AdaptiveRes, p::SIPProblem) = p.kwargs_res

function SIPProblem(x_l::Vector{Float64}, x_u::Vector{Float64},
                    p_l::Vector{Float64}, p_u::Vector{Float64},
                    gSIP, optimizer, kwargs)

    initial_eps_g = haskey(kwargs, :sip_initial_eps_g) ? kwargs[:sip_initial_eps_g] : 1.0
    initial_r = haskey(kwargs, :sip_initial_r) ? kwargs[:sip_initial_r] : 2.0

    (initial_r <= 1.0) && error("initial_r must be greater than 1")
    (initial_eps_g <= 0.0) && error("eps_g must be greater than 0")

    absolute_tolerance = haskey(kwargs, :sip_absolute_tolerance) ? kwargs[:sip_absolute_tolerance] : 1E-3
    constraint_tolerance = haskey(kwargs, :sip_constraint_tolerance) ? kwargs[:sip_constraint_tolerance] : 1E-3
    iteration_limit = haskey(kwargs, :sip_iteration_limit) ? kwargs[:sip_iteration_limit] : 100
    return_hist = haskey(kwargs, :sip_return_hist) ? kwargs[:sip_return_hist] : false
    header_interval = haskey(kwargs, :sip_header_interval) ? kwargs[:sip_header_interval] : 20
    print_interval = haskey(kwargs, :sip_print_interval) ? kwargs[:sip_print_interval] : 1
    verbosity = haskey(kwargs, :sip_verbosity) ? kwargs[:sip_verbosity] : 1
    local_solver = haskey(kwargs, :sip_local_solver) ?  kwargs[:sip_local_solver] : false

    np = length(p_l)
    nx = length(x_l)

    sense = haskey(kwargs, :sip_sense) ? kwargs[:sip_sense] : :min
    init_lower_disc = haskey(kwargs, :sip_init_lower_disc) ? kwargs[:sip_init_lower_disc] : Vector{Vector{Float64}}[]
    init_upper_disc = haskey(kwargs, :sip_init_upper_disc) ? kwargs[:sip_init_upper_disc] : Vector{Vector{Float64}}[]

    opt_dict = Dict{Symbol,Any}()
    for key in keys(kwargs)
        string_key = String(key)
        if string_key[1:3] !== "sip"
            opt_dict[key] = kwargs[key]
        end
    end

    # polyhedral_uncertainty_set = nothing
    # ellipsodial_uncertainty_set = nothing
    # conic_uncertainty_set = nothing
    # convex_uncertainty_set = nothing

    nSIP = length(gSIP)

    SIPProblem(x_l, x_u, p_l, p_u, np, nSIP, nx, sense, init_lower_disc,
               init_upper_disc, absolute_tolerance, constraint_tolerance,
               iteration_limit,
               initial_eps_g, initial_r, return_hist, header_interval,
               print_interval, verbosity, local_solver,
               #polyhedral_uncertainty_set,
               #ellipsodial_uncertainty_set, conic_uncertainty_set,
               #convex_uncertainty_set,
               optimizer, opt_dict)
end

struct SIPCallback
    f
    gSIP
end

"""
    SIPBuffer

Hold objective value, solution, discretization set, and feasibility status of
each subproblem encountered by SIP algorithm.
"""
@Base.kwdef mutable struct SIPSubResult
    lbd_x::Vector{Float64} = Float64[]
    ubd_x::Vector{Float64} = Float64[]
    res_x::Vector{Float64} = Float64[]
    llp1_p::Vector{Float64} = Float64[]
    llp2_p::Vector{Float64} = Float64[]
    llp3_p::Vector{Float64} = Float64[]
    obj_value_lbd::Float64 = 0.0
    obj_value_ubd::Float64 = 0.0
    obj_value_res::Float64 = 0.0
    obj_value_llp1::Float64 = 0.0
    obj_value_llp2::Float64 = 0.0
    obj_value_llp3::Float64 = 0.0
    is_feasible_lbd::Bool = false
    is_feasible_ubd::Bool = false
    is_feasible_res::Bool = false
    is_feasible_llp1::Bool = false
    is_feasible_llp2::Bool = false
    is_feasible_llp3::Bool = false
    llp1_abs_tol::Float64 = 1E-3
    llp2_abs_tol::Float64 = 1E-3
    llp3_abs_tol::Float64 = 1E-3
    lbd_disc::Vector{Vector{Float64}} = Vector{Float64}[]
    ubd_disc::Vector{Vector{Float64}} = Vector{Float64}[]
    res_disc::Vector{Vector{Float64}} = Vector{Float64}[]
end
function SIPSubResult(nx::Int, np::Int, ng::Int, tol::Float64)
    buffer = SIPSubResult()
    buffer.llp1_abs_tol = tol
    buffer.llp2_abs_tol = tol
    buffer.llp3_abs_tol = tol
    append!(buffer.lbd_x, zeros(nx))
    append!(buffer.ubd_x, zeros(nx))
    append!(buffer.res_x, zeros(nx))
    append!(buffer.llp1_p, zeros(np))
    append!(buffer.llp2_p, zeros(np))
    append!(buffer.llp3_p, zeros(np))
    for _ in 1:ng
        push!(buffer.lbd_disc, zeros(np))
        push!(buffer.ubd_disc, zeros(np))
        push!(buffer.res_disc, zeros(np))
    end
    return buffer
end

function load!(::LowerProblem, subresult::SIPSubResult, feas::Bool,
                               objval::Float64, x::Vector{Float64})
    subresult.is_feasible_lbd = feas
    subresult.obj_value_lbd = objval
    subresult.lbd_x .= x
    return nothing
end
function load!(::UpperProblem, subresult::SIPSubResult, feas::Bool,
                               objval::Float64, x::Vector{Float64})
    subresult.is_feasible_ubd = feas
    subresult.obj_value_ubd = objval
    subresult.ubd_x .= x
    return nothing
end
function load!(::ResProblem, subresult::SIPSubResult, feas::Bool,
                             objval::Float64, x::Vector{Float64})
    subresult.is_feasible_res = feas
    subresult.obj_value_res = objval
    subresult.res_x .= x
    return nothing
end
function load!(::LowerLevel1, subresult::SIPSubResult, feas::Bool,
                              objval::Float64, p::Vector{Float64})
    subresult.is_feasible_llp1 = feas
    subresult.obj_value_llp1 = objval
    subresult.llp1_p .= p
    return nothing
end
function load!(::LowerLevel2, subresult::SIPSubResult, feas::Bool,
                              objval::Float64, p::Vector{Float64})
    subresult.is_feasible_llp2 = feas
    subresult.obj_value_llp2 = objval
    subresult.llp2_p .= p
    return nothing
end
function load!(::LowerLevel3, subresult::SIPSubResult, feas::Bool,
                              objval::Float64, p::Vector{Float64})
    subresult.is_feasible_llp3 = feas
    subresult.obj_value_llp3 = objval
    subresult.llp3_p .= p
    return nothing
end

get_disc_set(s::LowerProblem, prob::SIPProblem) = prob.lbd_disc
get_disc_set(s::UpperProblem, prob::SIPProblem) = prob.ubd_disc
get_disc_set(s::ResProblem, prob::SIPProblem) = prob.res_disc
