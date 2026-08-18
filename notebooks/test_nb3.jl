cd(@__DIR__)
include(joinpath("..","src","PIC.jl"))
using .PIC
using Statistics
using FFTW
using FileIO, JLD2

out_file = joinpath("..", "Results",
    "thermal_rel_newmodule_J50x50_N5_Th3_alp1_o5_test.jld2")

isfile(out_file) || error("Output file not found: run test_nb2.jl first")

data = load(out_file)
N, J, box, order = data["par_grid"]
t_i, t_f, M, M_g, dt = data["par_evolv"]
g = PICGrid(box, J, order)

@show data["run_name"]

avg = load_averages(out_file, g, M_g;
    fields=(:n, :S, :E, :B, :Energy_K, :Energy_E, :T))

@assert size(avg[:n]) == (J[1], J[2], M_g)
@assert size(avg[:S]) == (2, J[1], J[2], M_g)
@assert size(avg[:E]) == (2, J[1], J[2], M_g)
@assert size(avg[:B]) == (J[1], J[2], M_g)
@assert length(avg[:Energy_K]) == M_g
@assert length(avg[:Energy_E]) == M_g
@assert length(avg[:T]) == M_g

println("Notebook 3 load test passed")
