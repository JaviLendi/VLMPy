# Author: Javier Lendínez Castillo
# SPDX-License-Identifier: AGPL-3.0-only
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

