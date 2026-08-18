include(joinpath(@__DIR__, "src", "PIC.jl"))
using .PIC
using LinearAlgebra
using Statistics

println("Threads: ", Threads.nthreads())

# --- 1D test ---
let
    g = PICGrid(Box((0.0, 1.0)), [32], 5)
    N = 32
    u = make_state(g, N)
    # place particles at grid points, zero momentum
    for i in 1:N
        u[(i-1)*2 + 1] = (i-1) * g.dx[1]
    end
    n = zeros(size(g))
    dep = DepositDensity(g, N)
    deposit!(dep, u, g, n)
    @show mean(n), std(n)
    @assert isapprox(mean(n), 1.0, atol=1e-10)

    # constant E field
    E = ones(1, size(g)...)
    set_fields!(u, g, N, E)
    Fi = zeros(N, 1)
    interpolate_field!(Fi, u, g)
    @show mean(Fi), std(Fi)
    @assert isapprox(mean(Fi), 1.0, atol=1e-10)

    # Poisson solver: rho = cos(2πx) -> E = -sin(2πx)/(2π)? Actually E' = rho -> E = sin(2πx)/(2π)
    rho = [cos(2π * (j-1) * g.dx[1]) for j in 1:size(g,1)]
    E2 = zeros(1, size(g)...)
    solve_poisson!(E2, rho, g)
    # mean should be zero
    @assert isapprox(sum(E2), 0.0, atol=1e-10)

    println("1D tests passed")
end

# --- 2D test ---
let
    g = PICGrid(Box((0.0, 1.0, 0.0, 1.0)), [16, 16], 5)
    N = 16*16
    u = make_state(g, N)
    # random positions, zero momentum
    for i in 1:N
        u[(i-1)*4 + 1] = rand()
        u[(i-1)*4 + 2] = rand()
    end
    n = zeros(size(g))
    dep = DepositDensity(g, N)
    deposit!(dep, u, g, n)
    Q = PIC.total_charge(n, g)
    @show Q, volume(g)
    @assert isapprox(Q, volume(g), rtol=1e-2)

    # constant E and B
    E = ones(2, size(g)...)
    B = ones(size(g))
    set_fields!(u, g, N, E, B)
    Fi = zeros(N, 2)
    interpolate_field!(Fi, u, g)
    # zero velocity => force = E => (1,1)
    @show mean(Fi[:,1]), mean(Fi[:,2])
    @assert isapprox(mean(Fi[:,1]), 1.0, atol=1e-10)
    @assert isapprox(mean(Fi[:,2]), 1.0, atol=1e-10)

    # Poisson solver (2D): rho = cos(2π x) cos(2π y), zero mode removed
    rho2 = [cos(2π * (j-1) * g.dx[1]) * cos(2π * (l-1) * g.dx[2])
            for j in 1:size(g,1), l in 1:size(g,2)]
    rho2 .-= mean(rho2)
    E2 = zeros(2, size(g)...)
    solve_poisson!(E2, rho2, g)
    @assert size(E2) == (2, size(g)...)
    @assert isapprox(sum(E2), 0.0, atol=1e-10)

    # RHS shape
    p = RHSParams(g, N; maxwell=NoMaxwell{2}())
    du = similar(u)
    rhs!(du, u, 0.0, p)
    @assert length(du) == length(u)
    println("2D tests passed")
end

# --- 3D test (small) ---
let
    g = PICGrid(Box((0.0, 1.0, 0.0, 1.0, 0.0, 1.0)), [8, 8, 8], 3)
    N = 8*8*2
    u = make_state(g, N)
    for i in 1:N
        u[(i-1)*6 + 1] = rand()
        u[(i-1)*6 + 2] = rand()
        u[(i-1)*6 + 3] = rand()
    end
    n = zeros(size(g))
    dep = DepositDensity(g, N)
    deposit!(dep, u, g, n)
    Q = PIC.total_charge(n, g)
    @show Q, volume(g)
    @assert isapprox(Q, volume(g), rtol=5e-2)

    # constant E and B, zero velocity -> force = E
    E = ones(3, size(g)...)
    B = ones(3, size(g)...)
    set_fields!(u, g, N, E, B)
    Fi = zeros(N, 3)
    interpolate_field!(Fi, u, g)
    @show mean(Fi[:,1]), mean(Fi[:,2]), mean(Fi[:,3])
    @assert isapprox(mean(Fi[:,1]), 1.0, atol=1e-10)

    # RHS with spectral Maxwell (shape only)
    p = RHSParams(g, N; maxwell=SpectralMaxwell(g))
    du = similar(u)
    rhs!(du, u, 0.0, p)
    @assert length(du) == length(u)
    println("3D tests passed")
end

println("All basic tests passed")
