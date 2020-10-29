using ACE, JuLIP, Test, ForwardDiff, LinearAlgebra, ACEexperimental
using ACEexperimental.Combinations: get_params, set_params!, set_params
using ACEexperimental.Solvers: steepestgradient, adam
using ACEexperimental.Problems: lls_basis_prob, combinations_prob

using Plots

#---
#reference model
r0 = rnn(:Al)

MixPot = let A = 4.0, r0 = r0
   @analytic r -> 6.0 * exp(- A * (r/r0 - 1.0)) - A * (r0/r)^6
end
MixPot = MixPot * SplineCutoff(2.1 * r0, 3.5 * r0)

LJPot = let r0 = r0
    @analytic r -> (r0/r)^12 - 2*(r0/r)^6 #lennard jhones
 end
LJPot = LJPot * SplineCutoff(2.1 * r0, 3.5 * r0) #this was random, need to think about it

MorsePot = let A = 1, r0 = r0 #optimize for A?
    @analytic r -> exp(- 2 * A * (r/r0 - 1.0)) - 2*exp(- A * (r/r0 - 1.0)) #Morse
 end
MorsePot = MorsePot * SplineCutoff(2.1 * r0, 3.5 * r0) #this was random, need to think about it


#---
#training set
#the training set consists of several "atom packages" it contains 
#for t in train t.at = atoms = rand_config(Vref)

#only used to "normalize energy"
function get_V0(train)
    # get list of atomic numbers
    Zs = AtomicNumber[]
    for (at, E) in train
       Zs = unique( [Zs; at.Z] )
    end
 
    # setup lsq system for E0s
    A = zeros(length(train), length(Zs))
    y = zeros(length(train))
    for (it, (at, E)) in enumerate(train)
       y[it] = E
       for (iz, z) in enumerate(Zs)
          A[it, iz] = length(findall(at.Z .== z))
       end
    end
    E0s = A \ y
    #@info("  E0s = $(E0s)")
    syms = chemical_symbol.(Zs)
    return JuLIP.OneBody([syms[i] => E0s[i] for i = 1:length(Zs)]...)
 end
 
function rand_config(V; rattle = 0.2, nrepeat = 3)
    at = bulk(:W, cubic=true, pbc=false) * nrepeat
    return rattle!(at, rattle)
 end

 function trainset(Vref, Ntrain; kwargs...)
    train = []
    for n = 1:Ntrain
       at = rand_config(Vref)
       push!(train, (at = at, E = energy(Vref, at), F = forces(Vref, at)))
    end
    V0 = get_V0(train)
    #@show V0
    train = [ (at = at, E = E - energy(V0, at), F = F)
                for (at, E,  F) in train ]
    return train
 end

 #potential and number of configurations or atoms?
 train = trainset(LJPot, 10)

 L1,c = lls_basis_prob(train)
 @show L1(c)
 @show L1(c*1.1)


 #R^n -> R^n
W1 = rand()
b1 = rand()
layer1(ρ) = W1 .* ρ .+ b1
#R^n -> R^n
W2 = rand()
b2 = rand()
layer2(ρ) = W2 .* ρ .+ b2
#R^n -> R
layer3(ρ) = sum(ρ)

neurNet = ρ -> layer3(layer2(layer1(ρ)))

 L2, V = combinations_prob(train, neurNet)

 
 norm_grad, θ = steepestgradient(L2, get_params(V), iterations=10 , param_sd= 1/10^4 , h=0.01 , quad = true , initstep = true , initminimizer = true , termination = 1/10^4)

 plot(norm_grad, yaxis=:log, xlab="iterations", ylab="norm_inf( nabla f(x) )", label="SD t, t, t, P1, lj")
