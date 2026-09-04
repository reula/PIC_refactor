module PIC
using Base.Threads
using StaticArrays
using LinearAlgebra
using FFTW
using Printf
using Statistics

# Core generic D-dimensional PIC library.
# Supports D = 1, 2, 3.

export
    # grids & particles
    Box, PICGrid, particle_dof, field_dof, state_length, particle_count, set_fields!, get_fields,
    # particle utils
    get_positions, get_momenta, p2v, v2p, γ_p, make_periodic!, coordinate_test,
    # shapes
    Shape, W_shape, static_bound,
    # grid utils
    volume, differentials, extent,
    # index computation
    get_index_and_y, get_indices_and_y!,
    # deposition
    DepositDensity, DepositCurrent, deposit!, reset!,
    # field solvers
    solve_poisson!, field_divergence, MaxwellSolver, NoMaxwell, SpectralMaxwell, SBPMaxwell,
    # interpolation
    interpolate_field!,
    # RHS & stepping
    rhs!, RHSParams, RHSWorkspace, StepWorkspace, RK4_step!, make_state, get_fields,
    # energies / diagnostics
    kinetic_energy, field_energy, total_energy, temperature_rel, temperature,
    total_momentum, total_charge, check_constraints, check_constraints_weak,
    # IO helpers
    save_averages, load_averages, save_snapshot,
    # initial data
    build_initial_data_D, retrieve_initial_data_D,
    f_p_rel, f_p_rel_max, f_p_thermal, f_p_thermal_max,
    f_p_weibel_norel, f_p_weibel_norel_max,
    f_p_two_particle_distribution, f_p_two_particle_distribution_max,
    f_x, f_x_max, random_sampling_from_distribution_D

const DMAX = 3

include("types.jl")
include("shapes.jl")
include("grid.jl")
include("particles.jl")
include("deposition.jl")
include("fields.jl")
include("interpolation.jl")
include("rhs.jl")
include("diagnostics.jl")
include("io.jl")
include("initial_data.jl")

end # module
