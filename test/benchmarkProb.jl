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
#training set of length N_train
N_train = 4
train = [rattle!( bulk(:W, cubic=true, pbc = false) * 2, 0.1 ) for _ = 1:N_train ]






#--
#1st problem 

#Linear least squares with monomial basis. exact solution exists
#given a training set and a potential to fit to we use LSQ with QR to fit to it
#we are looking for something to minimize L, and it should equal to our c since 
#this are the parameters that solve the linear system

#choose a potential
pot = LJPot

#find out how to do this with ACE
struct MonomialBasis{T} <: PairBasis
    alpha::T
    nparams::Int    
end

Base.length(B::PairBasis) = B.nparams

(B::MonomialBasis)(r) = [ exp( - B.alpha * (n-1) * (r - 1))  for n = 1:length(B) ] * fc(r)

#cheby
#implement cheby basis here

#choose a basis
basis = MonomialBasis(1.0, 8)

#cA = b problem
function LLSbasis(train, pot, basis)
    A = zeros(length(train), length(basis))
    b = zeros(length(train))
    for (iX, X) in enumerate(train) #need to figure out how to iterate train this naive might work
        E = energy(pot,X)
        w = (1.0 + max(0, E/length(X) - 1.0))^(-4)
        b[iX] = w * energy(pot,X)
        A[iX, :] = w * energy(basis,X)
    end 
    F = qr(A)
    c = F \ b 
    return(A,b,c)
end
#c is the array of parameters c=theta
A,b,c = LLSbasis(train, pot, basis)
#what will be our theta not, do we even need one?
L(θ) =  norm(A*θ - b) / norm(b)

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
n = 3
α = [1,2,3]
Ffun = ρ -> sum([abs(ρ[n])^(α[n]) for n in 1:N]) 


V = FitCombiPotential(basis, Ffun, 2)
set_params!(V, rand(length(get_params(V))))

#need to properly incorporate forces and decide what to do with the weights
#weights
w_RE = 1
w_RF = 1
#quadratic cost function
#J(V,train) = sum([w_RE^2 * abs(energy(V,R) - energy(pot, R))^2 + w_RF^2 * norm(forces(V, R) - forces(pot, R))^2 for R in train])
θ_o = get_params(V)
L(θ) = sum([w_RE^2 * abs(energy(set_params!(V,θ),R) - energy(pot, R))^2 for R in train])

#---
#3rd problem
#here we draft the problem where V is the composition of other functions