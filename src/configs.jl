#=
    configs.jl

Pre-defined 1D and 2D simulation configurations for PIC.jl.

These builders return named tuples (`SimulationConfig`) containing all
parameters needed to generate initial data and run a simulation.
=#

# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

function _format_fraction(x::Real)
    isinteger(x) && return string(round(Int, x))
    for den in 1:20
        num = round(Int, x * den)
        if num > 0 && abs(num / den - x) < 1e-12
            return "$(num)d$(den)"
        end
    end
    return replace(string(x), "." => "d")
end

function _data_name(prefix, args...)
    name = prefix
    for a in args
        name *= string(a)
    end
    return name
end

# ---------------------------------------------------------------------------
# Pre-defined 2D simulation configurations
# ---------------------------------------------------------------------------

"""
    SimulationConfig

Named tuple returned by the config builders below. Contains all parameters
needed to generate initial data and run a simulation.
"""
const SimulationConfig = NamedTuple

"""
    thermal_rel_config(; N_exp=5, J=(50,50), box_bounds=(0.0,1.0,0.0,1.0),
                        nm=[1,1], alpha_exp=1, theta_exp=3)

Relativistic thermal distribution with a cosine spatial perturbation.
"""
function thermal_rel_config(;
        N_exp::Int=5,
        J::NTuple{2,Int}=(50, 50),
        box_bounds::NTuple{4,Float64}=(0.0, 1.0, 0.0, 1.0),
        nm::Vector{Int}=[1, 1],
        alpha_exp::Int=1,
        theta_exp::Int=3,
    )
    N = 10^N_exp
    box = Box(box_bounds)
    alpha = 10.0^(-alpha_exp)
    theta = 10.0^(-theta_exp)
    p_max = sqrt((1 + 10 * theta)^2 - 1)
    box_p = Box((-p_max, p_max, -p_max, p_max))
    par_f_x = (alpha, nm, box)
    par_f_p = (theta, 2)
    data_name = "par_dis_rel_thermal_nm_[$(nm[1]),$(nm[2])]_alp$(alpha_exp)_N$(N_exp)_Th$(theta_exp)"
    run_name = "thermal_rel_Box_$(extent(box,1))x$(extent(box,2))_J_$(J[1])x$(J[2])_nm_[$(nm[1]),$(nm[2])]_Th$(theta_exp)_alp$(alpha_exp)"
    return (;
        name = :thermal,
        N, J, box, box_bounds, nm,
        alpha_exp, theta_exp, alpha, theta,
        f_p = f_p_thermal,
        f_p_max = f_p_thermal_max,
        par_f_p, box_p,
        f_x = f_x,
        f_x_max = f_x_max,
        par_f_x,
        data_name,
        run_name,
    )
end

"""
    weibel_norel_config(; N_exp=5, J=(100,100), box_bounds=(0.0,1.0,0.0,1.0),
                         Ax=1/2, theta_exp=3)

Non-relativistic Weibel anisotropic distribution.
"""
function weibel_norel_config(;
        N_exp::Int=5,
        J::NTuple{2,Int}=(100, 100),
        box_bounds::NTuple{4,Float64}=(0.0, 1.0, 0.0, 1.0),
        Ax::Float64=0.5,
        theta_exp::Int=3,
    )
    N = 10^N_exp
    box = Box(box_bounds)
    alpha_exp = 8
    alpha = 10.0^(-alpha_exp)
    theta = 10.0^(-theta_exp)
    p_max = sqrt((1 + 10 * 10 * theta)^2 - 1)
    box_p = Box((-p_max, p_max, -p_max, p_max))
    par_f_x = (alpha, [0.0, 0.0], box)
    par_f_p = (theta, 2, Ax)
    ax_label = "Ax_($(_format_fraction(Ax)))_"
    data_name = "par_dis_norel_weibel_$(ax_label)alp$(alpha_exp)_N$(N_exp)_Th$(theta_exp)"
    run_name = "weibel_norel_Box_$(extent(box,1))x$(extent(box,2))_J_$(J[1])x$(J[2])_$(ax_label)Th$(theta_exp)_alp$(alpha_exp)"
    return (;
        name = :weibel_norel,
        N, J, box, box_bounds,
        Ax, alpha_exp, theta_exp, alpha, theta,
        f_p = f_p_weibel_norel,
        f_p_max = f_p_weibel_norel_max,
        par_f_p, box_p,
        f_x = f_x,
        f_x_max = f_x_max,
        par_f_x,
        data_name,
        run_name,
    )
end

"""
    weibel_rel_config(; N_exp=5, J=(128,128), box_bounds=(0.0,15.0,0.0,15.0),
                       Ax=25, theta_exp=3, v_stream=[0.5,0.0])

Relativistic two-counter-stream Weibel distribution (Morozov-Nielsen style).
"""
function weibel_rel_config(;
        N_exp::Int=5,
        J::NTuple{2,Int}=(128, 128),
        box_bounds::NTuple{4,Float64}=(0.0, 15.0, 0.0, 15.0),
        Ax::Float64=25.0,
        theta_exp::Int=3,
        v_stream::Vector{Float64}=[0.5, 0.0],
        integration_resolution::Vector{Int}=[20000, 20000],
    )
    N = 10^N_exp
    box = Box(box_bounds)
    alpha_exp = 8
    alpha = 10.0^(-alpha_exp)
    theta1 = 10.0^(-theta_exp)
    theta2 = Ax * theta1
    p_max = 10 * theta2
    box_p = Box((-p_max, p_max, -p_max, p_max))
    par_f_x = (alpha, [0.0, 0.0], box)
    par_f_p_unnorm = (theta1, theta2, v_stream, 1.0, 2)
    norm = int_mid_point_f(f_p_two_particle_distribution, par_f_p_unnorm, integration_resolution, box_p)
    par_f_p = (theta1, theta2, v_stream, norm, 2)
    ax_label = "Ax_$(_format_fraction(Ax))_"
    data_name = "par_dis_rel_weibel_MN_$(ax_label)alp$(alpha_exp)_N$(N_exp)_Th$(theta_exp)"
    run_name = "weibel_rel_Box_$(extent(box,1))x$(extent(box,2))_J_$(J[1])x$(J[2])_$(ax_label)Th$(theta_exp)_alp$(alpha_exp)"
    return (;
        name = :weibel_rel,
        N, J, box, box_bounds,
        Ax, alpha_exp, theta_exp, theta1, theta2,
        v_stream,
        f_p = f_p_two_particle_distribution,
        f_p_max = f_p_two_particle_distribution_max,
        par_f_p, box_p,
        f_x = f_x,
        f_x_max = f_x_max,
        par_f_x,
        data_name,
        run_name,
    )
end

"""
    make_config(name::Symbol; kwargs...)

Convenience dispatcher for the predefined 2D configurations.
Supported names: `:thermal`, `:weibel_norel`, `:weibel_rel`.
(The damped Landau case is 1D; see `make_config_1d(:damped)`.)
"""
function make_config(name::Symbol; kwargs...)
    if name == :thermal
        return thermal_rel_config(; kwargs...)
    elseif name == :weibel_norel
        return weibel_norel_config(; kwargs...)
    elseif name == :weibel_rel
        return weibel_rel_config(; kwargs...)
    else
        error("unknown configuration name: $name (use :thermal, :weibel_norel, or :weibel_rel)")
    end
end

# ---------------------------------------------------------------------------
# Pre-defined 1D simulation configurations (Landau-damping family)
# ---------------------------------------------------------------------------

function _landau_1d_config(kind::String, L::Real, m::Int;
                           N_exp::Int=6,
                           J::Union{Int,NTuple{1,Int}}=128,
                           alpha_exp::Int=2,
                           theta_exp::Int=3,
                           integration_resolution::Vector{Int}=[20000])
    N = 8 * 10^N_exp
    box_x = Box((0.0, Float64(L)))
    box = box_x
    J1 = J isa Int ? (J,) : J
    alpha = 10.0^(-alpha_exp)
    theta = 10.0^(-theta_exp)
    p_max = sqrt((1 + 10 * theta)^2 - 1)
    box_p = Box((-p_max, p_max))
    par_f_x = (alpha, [m], box)
    par_f_p_unnorm = (theta, 1.0)
    norm = int_mid_point_f(f_p_rel, par_f_p_unnorm, integration_resolution, box_p)
    par_f_p = (theta, norm)
    data_name = "par_dis_landau_norm_norel_$(kind)_$(m)_alp$(alpha_exp)_8$(N_exp)_Th$(theta_exp)"
    run_name = "landau_1d_$(kind)_L_$(L)_J_$(J1[1])_m_$(m)_Th$(theta_exp)_alp$(alpha_exp)"
    return (; name = Symbol(:landau_1d_, kind),
              N, L, m, J=J1, box, box_x,
              alpha_exp, theta_exp, alpha, theta,
              f_p = f_p_rel,
              f_p_max = f_p_rel_max,
              par_f_p, box_p,
              f_x = f_x,
              f_x_max = f_x_max,
              par_f_x, data_name, run_name)
end

"""
    undamped_l_config(; N_exp=6, alpha_exp=2, theta_exp=3)

Long-wavelength undamped 1D Landau mode: `L = 39.738`, `m = 2`.
"""
undamped_l_config(; kwargs...) = _landau_1d_config("undamped_l", 39.738, 2; kwargs...)

"""
    undamped_s_config(; N_exp=6, alpha_exp=2, theta_exp=3)

Short-domain undamped 1D Landau mode: `L = 4.0`, `m = 2`.
"""
undamped_s_config(; kwargs...) = _landau_1d_config("undamped_s", 4.0, 2; kwargs...)

"""
    vlasov_exp_200_config(; N_exp=6, alpha_exp=2, theta_exp=3)

Large 1D Vlasov box: `L = 200.0`, `m = 10`.
"""
vlasov_exp_200_config(; kwargs...) = _landau_1d_config("vla_200", 200.0, 10; kwargs...)

"""
    vlasov_exp_40_config(; N_exp=6, alpha_exp=2, theta_exp=3)

Medium 1D Vlasov box: `L = 40.0`, `m = 2`.
"""
vlasov_exp_40_config(; kwargs...) = _landau_1d_config("vla_40", 40.0, 2; kwargs...)

"""
    landau_damped_1d_config(; N_exp=6, m=15, alpha_exp=2, theta_exp=3)

Standard damped 1D Landau mode: `L = 7.455`, default `m = 15`.
Other common choices from PIC-1D are `m = 12` or `m = 19`.
"""
landau_damped_1d_config(; m::Int=15, kwargs...) = _landau_1d_config("damped", 7.455, m; kwargs...)

"""
    make_config_1d(name::Symbol; kwargs...)

Convenience dispatcher for the predefined 1D Landau-damping configurations.
Supported names: `:undamped_l`, `:undamped_s`, `:vlasov_exp_200`,
`:vlasov_exp_40`, `:damped`.
"""
function make_config_1d(name::Symbol; kwargs...)
    if name == :undamped_l
        return undamped_l_config(; kwargs...)
    elseif name == :undamped_s
        return undamped_s_config(; kwargs...)
    elseif name == :vlasov_exp_200
        return vlasov_exp_200_config(; kwargs...)
    elseif name == :vlasov_exp_40
        return vlasov_exp_40_config(; kwargs...)
    elseif name == :damped
        return landau_damped_1d_config(; kwargs...)
    else
        error("unknown 1D configuration name: $name (use :undamped_l, :undamped_s, :vlasov_exp_200, :vlasov_exp_40, or :damped)")
    end
end