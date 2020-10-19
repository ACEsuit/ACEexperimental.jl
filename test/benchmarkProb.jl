using ACE, JuLIP, Test, ForwardDiff, LinearAlgebra
using ACEexperimental
using ACEexperimental.Combinations: get_params, set_params!, set_params
using JuLIP.Potentials: evaluate, evaluate_d
using ForwardDiff
using ForwardDiff: Dual
using Plots

@show pwd()
@show "1"

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

#choose a potential
pot = LJPot


#---
#training set
#the training set consists of several "atom packages" it contains 
#for t in train t.at = atoms = rand_config(Vref)

#do we need this one?
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
    @info("  E0s = $(E0s)")
    syms = chemical_symbol.(Zs)
    return JuLIP.OneBody([syms[i] => E0s[i] for i = 1:length(Zs)]...)
 end
 

function rand_config(V; rattle = 0.2, nrepeat = 3)
    at = bulk(:W, cubic=true, pbc=false) * nrepeat
    return rattle!(at, rattle)
 end

 #decide if we want or need E-energy(v0,at)
 function trainset(Vref, Ntrain; kwargs...)
    train = []
    for n = 1:Ntrain
       at = rand_config(Vref)
       push!(train, (at = at, E = energy(Vref, at), F = forces(Vref, at)))
    end
    V0 = get_V0(train)
    @show V0
    train = [ (at = at, E = E - energy(V0, at), F = F)
                for (at, E,  F) in train ]
    return train
 end

 train = trainset(pot, 1000)

@show length(train)
at,e,f = train[1]
@show e
@show train[1][2]

@show "prelims"



#--
#1st problem 

#Linear least squares with monomial basis. exact solution exists
#given a training set and a potential to fit to we use LSQ with QR to fit to it
#we are looking for something to minimize L, and it should equal to our c since 
#this are the parameters that solve the linear system

#use this for more freedom on the transform
function pair_basis_manual(; species = :X,
    # transform parameters
    r0 = 2.5,
    trans = PolyTransform(2, r0),
    # degree parameters
    maxdeg = 8,
    # radial basis parameters
    rcut = 5.0,
    rin = 0.5 * r0,
    pcut = 2,
    pin = 0,
    rbasis = transformed_jacobi(maxdeg, trans, rcut, rin; pcut=pcut, pin=pin))

 return PolyPairBasis(rbasis, species)
end

function get_2b_basis(species; maxdeg = 10, rcut = 8.0 )
    basis = pair_basis_manual(species = species,
       r0 = rnn(Symbol(species)),
       maxdeg = maxdeg,
       rcut = rcut)
    return basis
 end

#poly pair basis
#tutorial or something
#pair basis and monkey patching

function lsq(train, basis; verbose=true, wE = 1.0, wF = 1.0)
    @info("lsq info")
    nobs = sum( 1+3*length(t.at) for t in train )
    @info("  nobs = $(nobs); nbasis = $(length(basis))")
    A = zeros(nobs, length(basis))
    y = zeros(nobs)
    irow = 0
    for (at, E, F) in train
       # add energy to the lsq system
       irow += 1
       y[irow] = wE * E / length(at)
       A[irow, :] = wE * energy(basis, at) / length(at)
 
       # add forces to the lsq system
       nf = 3*length(at)
       y[(irow+1):(irow+nf)] = wF * mat(F)[:]
       Fb = forces(basis, at)
       for ib = 1:length(basis)
          A[(irow+1):(irow+nf), ib] = wF * mat(Fb[ib])[:]
       end
       irow += nf
    end
    qrF = qr(A)
    c = qrF \ y
    relrmse = norm(A * c - y) / norm(y)
    @info("   cond(R) = $(cond(qrF.R)); relrmse = $(relrmse)")
    return A,c,y
 end

#choose a basis
basis = get_2b_basis(:W; maxdeg = 14, rcut = 6.0)

#Ac = y problem
A,c,y = lsq(train, basis, wE=30.0, wF=1.0)

#what will be our theta not, do we even need one?
L(θ) =  norm(A * θ - y) / norm(y)

θ_0 = zeros(14)
@show L(θ_0)

@show "1st prob"

#---
#2nd problem

#multi-body potential, this is the new problem with added non linearities
#Ffun has the nonlinearity which is embeded in the potential
#both problems seek to find theta to fit the potential given. 
#why don't we use a basis here as well?
#this is not really a basis, its kind of used only to make the potential
#we are fitting. this potential has the nonlinearity, while before it had 
#e^ something right? and thats why we could use the basis? ???????
basis = ACE.Utils.ace_basis(; species = :W, N = 4, maxdeg = 6)

#here we can play with nonlinearities
Ffun = ρ -> ρ[1] + sqrt(ρ[2])
Ffun = ρ -> ρ[1] + exp(-ρ[2]^2)
#number of parameters, although is it really?
N = 3
α = [1,2,3]
Ffun = ρ -> sum([abs(ρ[n])^(α[n]) for n in 1:N]) 
Ffun = ρ -> ρ[1] + exp(-ρ[2]^2)

#the 2 here is it the number of parameters?
#if so how are we going to know this before hand
#and we need to fix the code with it
V = FitCombiPotential(basis, Ffun, 2)
@show V
set_params!(V, rand(length(get_params(V))))

#need to properly incorporate forces and decide what to do with the weights
#weights
w_RE = 1
w_RF = 1
#quadratic cost function
#J(V,train) = sum([w_RE^2 * abs(energy(V,R) - energy(pot, R))^2 + w_RF^2 * norm(forces(V, R) - forces(pot, R))^2 for R in1 train])
θ_0 = get_params(V)

#fix the energy, it was already calculated. instead of energy(pot, t.at) do t.E y t.F for force
L(θ) = sum([w_RE^2 * abs(energy(set_params!(V,θ),t.at) - t.E)^2 + w_RF^2 * norm(forces(set_params!(V,θ), t.at) - t.F)^2 for t in train])
#@show L(θ_0)
@show "2nd prog"

#---
#3rd problem
#here we draft the problem where V is the composition of other functions

using Flux

W1 = rand(3, 5)
b1 = rand(3)
layer1(x) = W1 * x .+ b1

W2 = rand(2, 3)
b2 = rand(2)
layer2(x) = W2 * x .+ b2

Ffun(x) = layer2(σ.(layer1(x)))
V = FitCombiPotential(basis, Ffun, 2)