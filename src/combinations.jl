

module Combinations

using ACE, JuLIP, ForwardDiff, LinearAlgebra
using JuLIP.MLIPs: IPBasis
using ACE.RPI: RPIBasis


import JuLIP.Potentials: zlist, z2i, i2z, numz,
                         evaluate!, evaluate_d!,
                         alloc_temp, alloc_temp_d,
                         evaluate, evaluate_d,
                         cutoff

import JuLIP: fltype

export FitCombiPotential,
       num_features



@doc raw"""
`mutable struct FitCombiPotential`: Defines a site potential of the form
```math
\begin{aligned}
   E_i &= F(\rho_1, \dots, \rho_P), \\
   \rho_p &= \sum_k \theta_{pk} B_k
\end{aligned}
where $$B_k$$ are the basis functions and $$\theta_{pk}$$ the parameters.
```

### Fields
* `basis` : any IPBasis is allowed for now
* `params` : the parameters, a matrix; cf above.
* `F` : the nonlinear outer function.

It is assumed that the output of F is also `::T`
"""
mutable struct FitCombiPotential{T, TP, TF, NZ} <: SitePotential
   basis::RPIBasis{T}
   params::NTuple{NZ, Matrix{TP}}
   F::TF
end

zlist(V::FitCombiPotential) = zlist(V.basis)
# z2i(V::FitCombiPotential, z) = z2i(V.basis, z)
# i2z(V::FitCombiPotential, i) = i2z(V.basis, i)
# numz(V::FitCombiPotential) = length(zlist(V.basis))

fltype(V::FitCombiPotential{T}) where {T} = T

cutoff(V::FitCombiPotential) = cutoff(V.basis)

function FitCombiPotential(basis, F, numfeatures)
   T = fltype(basis)
   NZ = length(zlist(basis))
   params = ntuple( iz -> zeros(T, (numfeatures, length(basis, iz))), NZ )
   return FitCombiPotential(basis, params, F)
end

# -------------- Parameter management ...


num_features(V::FitCombiPotential) = size(V.params, 1)

get_params(V::FitCombiPotential) = vcat(vec.(V.params)...)

function set_params!(V::FitCombiPotential, θ)
   idx = 0
   len = length(V.params[1])
   for iz = 1:numz(V)
      V.params[iz][:] .= θ[(idx+1):(idx+len)]
      idx += len
   end
   return V
end

function set_params(V::FitCombiPotential, p::Vector{TP}) where {TP}
   new_params = ntuple( iz -> zeros(TP, (num_features(V), length(V.basis, iz))), numz(V) )
   Vnew = FitCombiPotential(V.basis, new_params, V.F)
   idx = 0
   len = length(Vnew.params[1])
   for iz = 1:numz(V)
      Vnew.params[iz][:] .= p[(idx+1):(idx+len)]
      idx += len
   end
   return Vnew
end


# -------------- Evaluation Code


function evaluate!(tmp, V::FitCombiPotential,
                   Rs::AbstractVector{JVec{T}},
                   Zs::AbstractVector{<:AtomicNumber},
                   z0::AtomicNumber) where {T}
   iz0 = z2i(V, z0)
   # Evaluate the inner basis
   #        TODO: replace with non-allocating version!
   B = evaluate(V.basis, Rs, Zs, z0)
   # only  get the part of the basis vector that matters to this site
   B_z0 = @view B[V.basis.Bz0inds[iz0]]
   # apply the parameters to get the densities
   ρ = V.params[iz0] * B_z0
   # return the nonlinear function of the densities.
   return V.F(ρ)
end


function evaluate_d!(dEs, tmpd, V::FitCombiPotential,
                   Rs::AbstractVector{JVec{T}},
                   Zs::AbstractVector{<:AtomicNumber},
                   z0::AtomicNumber) where {T}
   iz0 = z2i(V, z0)
   # Evaluate the densities (cf evaluate! for comments)
   B = evaluate(V.basis, Rs, Zs, z0)
   B_z0 = @view B[V.basis.Bz0inds[iz0]]
   ρ = V.params[iz0] * B_z0
   # derivative of F ---  TODO : get rid of enforcing AD here!
   dF = ForwardDiff.gradient(V.F, ρ)
   # get the collapsed coefficients
   #    ∂_{rj} Ei = ∑_p ∂_{ρp} F * ∂_{rj} ρp
   #              = ∑_k [ ∑_p ∂_{ρp} F * θ_{pk} ] * ∂_{rj} B_k
   c = dF' * V.params[iz0]   # TODO: this is ridiculous ...
   dB = evaluate_d(V.basis, Rs, Zs, z0)
   dB_z0 = @view dB[V.basis.Bz0inds[iz0], :]
   dEi = (UniformScaling.(c) * dB_z0)[:]
   return dEi
end

# --------------- Parameter Gradient







# SECOND STEP: fast evaluator! -> TODO later
# mutable struct CombiPotential{TF} <: AbstractCalculator
#    ace::PIPotential
#    F::TF
# end


end
