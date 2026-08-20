cd(@__DIR__)
include(joinpath("..","src","PIC.jl"))
using .PIC
using Statistics

box = Box((0.0, 1.0, 0.0, 1.0))
g = PICGrid(box, [50, 50], 1)

# rho = cos(2π x) cos(2π y)
rho = [cos(2π * (j-1)*g.dx[1]) * cos(2π * (l-1)*g.dx[2]) for j in 1:50, l in 1:50]
rho .-= mean(rho)

E = zeros(2, 50, 50)
solve_poisson!(E, rho, g)

# Compute divE spectrally
spec = PIC.SpectralMaxwell(g)
divE = zeros(50, 50)
divE .+= PIC._spectral_derivative(E[1,:,:], spec, 1)
divE .+= PIC._spectral_derivative(E[2,:,:], spec, 2)

println("rho range: ", extrema(rho))
println("divE range: ", extrema(divE))
println("max |divE - rho|: ", maximum(abs.(divE .- rho)))
println("max |divE + rho|: ", maximum(abs.(divE .+ rho)))
