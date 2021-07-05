# Copyright (c) 2018: Matthew Wilhelm & Matthew Stuber.
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# EAGO
# A development environment for robust and global optimization
# See https://github.com/PSORLab/EAGO.jl
#############################################################################
# # src/eago_semiinfinite/types.jl
# Defines intermediate types used by SIP routines.
#############################################################################

abstract type AbstractSIPAlgo end

abstract type AbstractSubproblemType end
struct LowerLevel1 <: AbstractSubproblemType end
struct LowerLevel2 <: AbstractSubproblemType end
struct LowerLevel3 <: AbstractSubproblemType end
struct LowerProblem <: AbstractSubproblemType end
struct UpperProblem <: AbstractSubproblemType end
struct ResProblem <: AbstractSubproblemType end

for ptype in (LowerLevel1, LowerLevel2, LowerLevel3,
                  LowerProblem, UpperProblem, ResProblem)
    pstring = String(Symbol(ptype))[6:end]
    @eval Base.show(io::IO, x::$ptype) = print(io, $pstring)
    @eval Base.show(io::IO, m::MIME"text/plain", x::$ptype) = print(io, $pstring)
end

"""
    SIPResult

Structure storing the results of the SIPres algorithm.
"""
mutable struct SIPResult
    iteration_number::Int64
    res_iteration_number::Int64
    upper_bound::Float64
    lower_bound::Float64
    feasibility::Bool
    xsol::Vector{Float64}
    psol::Vector{Float64}
    solution_time::Float64
end
SIPResult() = SIPResult(1, 1, Inf, -Inf, true, Float64[], Float64[], 0.0)
SIPResult(nx::Int, np::Int) = SIPResult(1, 1, Inf, -Inf, true, zeros(nx), zeros(np), 0.0)


"""
      SIPProblem

Structure storing problem information for the solution routine.
"""
Base.@kwdef mutable struct SIPProblem
    x_l::Vector{Float64}            = Float64[]
    x_u::Vector{Float64}            = Float64[]
    p_l::Vector{Float64}            = Float64[]
    p_u::Vector{Float64}            = Float64[]
    np::Int                         = 0
    nSIP::Int                       = 0
    nx::Int                         = 0

    abs_tolerance::Float64     = 1E-3
    cons_tolerance::Float64   = 1E-3
    iteration_limit::Int            = 100
    res_iteration_limit::Int        = 3
    initial_eps_g::Float64          = 1.0
    initial_r::Float64              = 2.0
    local_solver::Bool              = false

    return_hist::Bool               = false
    header_interval::Int            = 20
    print_interval::Int             = 1
    verbosity::Int                  = 1

    kwargs_llp1::Dict{String,Any}   = Dict{String,Any}()
    kwargs_llp2::Dict{String,Any}   = Dict{String,Any}()
    kwargs_llp3::Dict{String,Any}   = Dict{String,Any}()
    kwargs_lbd::Dict{String,Any}    = Dict{String,Any}()
    kwargs_ubd::Dict{String,Any}    = Dict{String,Any}()
    kwargs_res::Dict{String,Any}    = Dict{String,Any}()
end

get_sip_kwargs(s::LowerLevel1, p::SIPProblem) = p.kwargs_llp1
get_sip_kwargs(s::LowerLevel2, p::SIPProblem) = p.kwargs_llp2
get_sip_kwargs(s::LowerLevel3, p::SIPProblem) = p.kwargs_llp3
get_sip_kwargs(s::LowerProblem, p::SIPProblem) = p.kwargs_lbd
get_sip_kwargs(s::UpperProblem, p::SIPProblem) = p.kwargs_ubd
get_sip_kwargs(s::ResProblem, p::SIPProblem) = p.kwargs_res

function SIPProblem(x_l::Vector{Float64}, x_u::Vector{Float64},
                    p_l::Vector{Float64}, p_u::Vector{Float64},
                    gSIP, kwargs)

    prob = SIPProblem()

    for key in keys(kwargs)
        string_key = String(key)
        if string_key[1:5] === "llp1_"
            prob.kwargs_llp1[string_key[6:end]] = kwargs[key]
        elseif string_key[1:5] === "llp2_"
            prob.kwargs_llp2[string_key[6:end]] = kwargs[key]
        elseif string_key[1:5] === "llp3_"
            prob.kwargs_llp3[string_key[6:end]] = kwargs[key]
        elseif string_key[1:4] === "lbd_"
            prob.kwargs_lbd[string_key[5:end]] = kwargs[key]
        elseif string_key[1:4] === "ubd_"
            prob.kwargs_ubd[string_key[5:end]] = kwargs[key]
        elseif string_key[1:4] === "res_"
            prob.kwargs_res[string_key[5:end]] = kwargs[key]
        elseif key in fieldnames(SIPProblem)
            setfield!(prob, key, kwargs[key])
        else
            error("Keyword = `$string_key` is not supported by SIPProblem. Use
                  `llp1_$string_key` to pass this argument to the optimizer used by
                  the first lower level problem, `lbd_$string_key` to pass the
                  keyword to the optimizer used in the lower bounding problem
                  and so on...")
        end
    end

    (prob.initial_r <= 1.0) && error("initial_r must be greater than 1")
    (prob.initial_eps_g <= 0.0) && error("eps_g must be greater than 0")

    prob.np = length(p_l)
    prob.nx = length(x_l)
    prob.nSIP = length(gSIP)
    append!(prob.x_l, x_l)
    append!(prob.x_u, x_u)
    append!(prob.p_l, p_l)
    append!(prob.p_u, p_u)

    return prob
end

struct SIPCallback
    f
    gSIP
end

mutable struct SubProblemInfo
    sol::Vector{Float64}
    obj_val::Float64
    obj_bnd::Float64
    feas::Bool
    tol::Vector{Float64}
end
function SubProblemInfo(nd::Int, ng::Int, tol::Float64)
    SubProblemInfo(zeros(nd), 0.0, 0.0, false, fill(tol, ng))
end

"""
    SIPBuffer

Hold objective value, solution, discretization set, and feasibility status of
each subproblem encountered by SIP algorithm.
"""
Base.@kwdef mutable struct SIPSubResult
    fRes::Float64 = Inf
    lbd::SubProblemInfo
    ubd::SubProblemInfo
    res::SubProblemInfo
    llp1::SubProblemInfo
    llp2::SubProblemInfo
    llp3::SubProblemInfo
    r_g::Float64 = 5.0
    r_l::Float64 = 5.0
    r_u::Float64 = 5.0
    eps_g::Vector{Float64} = Float64[]
    eps_l::Vector{Float64} = Float64[]
    eps_u::Vector{Float64} = Float64[]
    disc_l_buffer::Vector{Float64} = Float64[]
    disc_u_buffer::Vector{Float64} = Float64[]
    disc_l::Vector{Vector{Vector{Float64}}} = Vector{Vector{Float64}}[]
    disc_u::Vector{Vector{Vector{Float64}}} = Vector{Vector{Float64}}[]
end
function SIPSubResult(nx::Int, np::Int, ng::Int, tol::Float64)
    buffer = SIPSubResult(lbd  = SubProblemInfo(nx, 1, tol),
                          ubd  = SubProblemInfo(nx, 1, tol),
                          res  = SubProblemInfo(nx, 1, tol),
                          llp1 = SubProblemInfo(np, ng, tol),
                          llp2 = SubProblemInfo(np, ng, tol),
                          llp3 = SubProblemInfo(np, ng, tol))
    append!(buffer.eps_g, fill(1E-3, ng))
    append!(buffer.eps_l, fill(1E-3, ng))
    append!(buffer.eps_u, fill(1E-3, ng))
    append!(buffer.disc_l_buffer, zeros(np))
    append!(buffer.disc_u_buffer, zeros(np))
    for _ in 1:ng
        push!(buffer.disc_l, Vector{Float64}[])
        push!(buffer.disc_u, Vector{Float64}[])
    end
    return buffer
end

get_eps(::LowerProblem, sr::SIPSubResult, i::Int) = 0.0
get_eps(::ResProblem, sr::SIPSubResult, i::Int) = 0.0
get_eps(::UpperProblem, sr::SIPSubResult, i::Int) = sr.eps_g[i]

const SUBPROB_SYM = Dict{Symbol,Symbol}(:LowerProblem => :lbd,
                                        :UpperProblem => :ubd,
                                        :ResProblem   => :res,
                                        :LowerLevel1  => :llp1,
                                        :LowerLevel2  => :llp2,
                                        :LowerLevel3  => :llp3)

for (typ, fd) in SUBPROB_SYM
    @eval function load!(::$typ, sr::SIPSubResult, feas::Bool, val::Float64,
                                 bnd::Float64, x::Vector{Float64})

        sr.$fd.feas = feas
        sr.$fd.obj_val = val
        sr.$fd.obj_bnd = bnd
        sr.$fd.sol .= x
        return nothing
    end
end
