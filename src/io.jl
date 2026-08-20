"""
Lightw eight I/O helpers built on JLD2.
"""

using FileIO
using JLD2

function save_averages(file_name::String, j::Int, u::AbstractVector, g::PICGrid{D}, N::Int;
                       m::Float64=1.0, dep::Union{DepositDensity{D},Nothing}=nothing,
                       factor::Int=1) where {D}
    tiempo = @sprintf("%05d", j)
    n = zeros(size(g))
    S = zeros(D, size(g)...)
    dep_n = DepositDensity(g, N)
    dep_S = DepositCurrent(g, N)
    deposit!(dep_n, u, g, n)
    deposit!(dep_S, u, g, S; m=m)
    E, B = get_fields(u, g, N)
    EK, EE = total_energy(u, g, N; m=m)
    T = temperature_rel(u, g, N; m=m)
    P = total_momentum(u, g, N)
    Q = total_charge(n, g)

    jldopen(file_name, "a+") do file
        file["n_$(tiempo)"] = n
        file["S_$(tiempo)"] = S
        file["E_$(tiempo)"] = E
        file["B_$(tiempo)"] = B
        file["Energy_K_$(tiempo)"] = EK
        file["Energy_E_$(tiempo)"] = EE
        file["p_T_$(tiempo)"] = P
        file["Q_T_$(tiempo)"] = Q
        file["T_$(tiempo)"] = T
        if dep !== nothing
            err_max, err_l2 = check_constraints(u, g, N, dep, factor)
            file["Gauss_max_$(tiempo)"] = err_max
            file["Gauss_l2_$(tiempo)"] = err_l2
        end
    end
end

function save_snapshot(file_name::String, j::Int, u::AbstractVector)
    tiempo = @sprintf("%05d", j)
    jldopen(file_name, "a+") do file
        file["u_$(tiempo)"] = u
    end
end

function load_averages(file_name::String, g::PICGrid{D}, M_g::Int; fields=(:n, :S, :Energy_K, :Energy_E, :T), indices=1:M_g) where {D}
    data = load(file_name)
    out = Dict{Symbol,Any}()
    for key in fields
        out[key] = _load_field(data, key, g, M_g; indices=indices)
    end
    return out
end

function _load_field(data, key::Symbol, g::PICGrid{D}, M_g::Int; indices=1:M_g) where {D}
    sz = size(g)
    nidx = length(indices)
    if key in (:Energy_K, :Energy_E, :T, :Q, :Gauss_max, :Gauss_l2)
        arr = zeros(Float64, nidx)
        for (k, j) in enumerate(indices)
            tiempo = @sprintf("%05d", j)
            arr[k] = data["$(key)_$(tiempo)"]
        end
        return arr
    elseif key == :p_T
        arr = zeros(Float64, D, nidx)
        for (k, j) in enumerate(indices)
            tiempo = @sprintf("%05d", j)
            arr[:, k] = data["p_T_$(tiempo)"]
        end
        return arr
    elseif key in (:S, :E) || (key == :B && D == 3)
        T = (D, sz...)
    elseif key == :B
        T = sz
    else
        T = sz
    end
    arr = zeros(Float64, T..., nidx)
    colons = ntuple(_ -> :, ndims(arr) - 1)
    for (k, j) in enumerate(indices)
        tiempo = @sprintf("%05d", j)
        arr[colons..., k] = data["$(key)_$(tiempo)"]
    end
    return arr
end
