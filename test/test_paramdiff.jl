

#---

using ACE, JuLIP, Test, ForwardDiff, LinearAlgebra
using ACEexperimental
using ACEexperimental.Combinations: get_params, set_params!, set_params
using JuLIP.Potentials: evaluate, evaluate_d
using ForwardDiff
using ForwardDiff: Dual

#---

basis = ACE.Utils.ace_basis(; species = :W, N = 4, maxdeg = 6)
Ffun = ρ -> ρ[1] + exp(-ρ[2]^2)
V = FitCombiPotential(basis, Ffun, 2)
set_params!(V, rand(length(get_params(V))))

#---

at = rattle!( bulk(:W, cubic=true, pbc = false) * 2, 0.1 )
energy(V, at)
forces(V, at)

#small change
p0 = get_params(V)
# write the energy  as a vector containing a single value
Efun = p -> [ energy(set_params(V, p), at) ]
Efun(p0)
# we want to use the jacobian
ForwardDiff.jacobian(Efun, p0)

# write forces (vector of vectors) as a single long vector
Ffun = p -> mat(forces(set_params(V, p), at))[:]
Ffun(p0)
ForwardDiff.jacobian(Ffun, p0)
