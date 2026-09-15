cd(@__DIR__)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath("..","src","PIC.jl"))
using .PIC
using Statistics
using FFTW
using FileIO, JLD2
using Printf

# Use the existing spectral test output (which has 3 snapshots) as if it were partial
out_file = joinpath("..", "Results",
    "thermal_rel_spectral_J50x50_N5_Th3_alp1_o5_test.jld2")

isfile(out_file) || error("Output file not found: run test_nb2_spectral.jl first")

data = load(out_file)
N, J, box, order = data["par_grid"]
t_i, t_f_target, M, M_g_target, dt = data["par_evolv"]
g = PICGrid(box, J, order)

# Detect how many snapshots were actually saved
tiempos = [k for k in 1:M_g_target if haskey(data, @sprintf("n_%05d", k))]
M_g = length(tiempos)
println("Requested M_g = ", M_g_target, ", actually saved = ", M_g)

out_every = div(M - 1, M_g_target - 1)
t_series = t_i .+ dt .* out_every .* (tiempos .- 1)

avg = load_averages(out_file, g, M_g; indices=tiempos,
    fields=(:n, :S, :E, :B, :Energy_K, :Energy_E, :T, :Gauss_max, :Gauss_l2))

@assert size(avg[:n]) == (J[1], J[2], M_g)
@assert size(avg[:S]) == (2, J[1], J[2], M_g)
@assert size(avg[:E]) == (2, J[1], J[2], M_g)
@assert size(avg[:B]) == (J[1], J[2], M_g)
@assert length(avg[:Energy_K]) == M_g
@assert length(avg[:Energy_E]) == M_g
@assert length(avg[:T]) == M_g
@assert length(avg[:Gauss_max]) == M_g
@assert length(avg[:Gauss_l2]) == M_g

println("t_series = ", t_series)
println("final t = ", t_series[end])
println("Notebook 3 partial-load test passed")
