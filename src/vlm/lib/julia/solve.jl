# wing.jl
using LinearAlgebra
using SparseArrays
using IterativeSolvers
using IncompleteLU 

function solve_vlm(P_ij, w_i)

    P_ij

    gammas = P_ij \ w_i

    return gammas
end

