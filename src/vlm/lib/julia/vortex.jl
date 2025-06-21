# wing.jl
using LinearAlgebra
using SparseArrays
using IterativeSolvers
using IncompleteLU 

function calculate_vortex_terms_batch(r1, r2, r0, r1_, r2_)
    n = size(r1, 1)
    V_AB = zeros(n, n, 3)
    V_AInf = zeros(n, n, 3)
    V_BInf = zeros(n, n, 3)

    for i in 1:n
        for j in 1:n

            r1_norm             = norm(r1[i,j,:])
            r2_norm             = norm(r2[i,j,:])

            cross_r1_r2         = cross(r1[i,j,:], r2[i,j,:])
            cross_r1_r2_norm    = norm(cross_r1_r2)

            r0r1                = dot(r0[i,j,:], r1[i,j,:])
            r0r2                = dot(r0[i,j,:], r2[i,j,:])

            psi                 = cross_r1_r2 / cross_r1_r2_norm^2 
            omega               = r0r1 / r1_norm - r0r2 / r2_norm

            V_AB[i,j,:]         = (psi * omega) / (4 * pi)

            V_AInf[i,j,2]       = (r1[i,j,3] / (r1[i,j,3]^2+r1_[i,j,2]^2)) * (1 + r1[i,j,1] / r1_norm) / (4 * pi) 
            V_AInf[i,j,3]       = (r1_[i,j,2] / (r1[i,j,3]^2+r1_[i,j,2]^2)) * (1 + r1[i,j,1] / r1_norm) / (4 * pi)

            V_BInf[i,j,2]       = -(r2[i,j,3] / (r2[i,j,3]^2+r2_[i,j,2]^2)) * (1 + r2[i,j,1] / r2_norm) / (4 * pi) 
            V_BInf[i,j,3]       = -(r2_[i,j,2] / (r2[i,j,3]^2+r2_[i,j,2]^2)) * (1 + r2[i,j,1] / r2_norm) / (4 * pi)

        end
    end
    return V_AB, V_AInf, V_BInf
end