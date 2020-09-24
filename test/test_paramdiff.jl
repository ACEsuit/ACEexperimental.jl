

#---

using ACE, JuLIP, Test, ForwardDiff, LinearAlgebra
using ACEexperimental
using ACEexperimental.Combinations: get_params, set_params!, set_params
using JuLIP.Potentials: evaluate, evaluate_d


#---

basis = ACE.Utils.ace_basis(; species = :W, N = 4, maxdeg = 6)
Ffun = ρ -> ρ[1] + exp(-ρ[2])
V = FitCombiPotential(basis, Ffun, 2)
set_params!(V, rand(length(get_params(V))))

#---

at = rattle!( bulk(:W, cubic=true, pbc = false) * 2, 0.1 )
energy(V, at)
forces(V, at)

p0 = get_params(V)
Efun = p -> energy(set_params(V, p), at)
print("bew")
print(Efun(p0))

ForwardDiff.gradient(Efun, p0)
