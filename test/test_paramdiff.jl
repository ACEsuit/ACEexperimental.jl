

#---

using ACE, JuLIP, Test, ForwardDiff, LinearAlgebra
using ACEexperimental
using ACEexperimental.Combinations: get_params, set_params!, set_params
using JuLIP.Potentials: evaluate, evaluate_d
using ForwardDiff
using ForwardDiff: Dual
using Plots
@show pwd()

#---
#reference model
r0 = rnn(:Al)

pot = let A = 4.0, r0 = r0
   @analytic r -> 6.0 * exp(- A * (r/r0 - 1.0)) - A * (r0/r)^6
end
pot = pot * SplineCutoff(2.1 * r0, 3.5 * r0)

#---

#multi-body potential
basis = ACE.Utils.ace_basis(; species = :W, N = 4, maxdeg = 6)
Ffun = ρ -> ρ[1] + exp(-ρ[2]^2)

V = FitCombiPotential(basis, Ffun, 2)
set_params!(V, rand(length(get_params(V))))

#---
#singular atoms
at = rattle!( bulk(:W, cubic=true, pbc = false) * 2, 0.1 )
#training set of length N_train
N_train = 4
train = [rattle!( bulk(:W, cubic=true, pbc = false) * 2, 0.1 ) for _ = 1:N_train ]

#weights
w_RE = 1
w_RF = 1
#quadratic cost function
#J(V,train) = sum([w_RE^2 * abs(energy(V,R) - energy(pot, R))^2 + w_RF^2 * norm(forces(V, R) - forces(pot, R))^2 for R in train])
J(V,train) = sum([w_RE^2 * abs(energy(V,R) - energy(pot, R))^2 for R in train])

@show J(V, train)

p0 = get_params(V)
@show p0
# write the energy  as a vector containing a single value
Efun = p -> [ energy(set_params(V, p), at) ]

# we want to use the jacobian
ForwardDiff.jacobian(Efun, p0)

# write forces (vector of vectors) as a single long vector
#Ffun = p -> mat(forces(set_params(V, p), at))[:]
#Ffun(p0)
#ForwardDiff.jacobian(Ffun, p0)

#now the cost function
Jfun = p -> [J(set_params(V, p), train) ]
Jac = ForwardDiff.jacobian(Jfun, p0)
p = p0
while norm(Jac) > 0.01
   @show "2"
   p = p - transpose(Jac)*Jac \ transpose(Jac)*J(set_params!(V,p),train) 
   @show "2"
   Jac = ForwardDiff.jacobian(Jfun, p)
   
   hmj
end 

@show "end"

  