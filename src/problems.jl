module Problems

using LinearAlgebra
using ACEexperimental
using ACE, JuLIP, Test, ForwardDiff, LinearAlgebra
using ACEexperimental.Combinations: get_params, set_params!, set_params
"""

Linear least squares with Poly transform basis.

Eact solution exists given a training set and a potential to fit to we use LSQ 
with QR to fit to it we are looking for something to minimize L, and it should 
equal to our c since this are the parameters that solve the linear system

"""

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


 function lsq(train, basis; verbose=false, wE = 1.0, wF = 1.0)
    #@info("lsq info")
    nobs = sum( 1+3*length(t.at) for t in train )
    #@info("  nobs = $(nobs); nbasis = $(length(basis))")
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
    #@info("   cond(R) = $(cond(qrF.R)); relrmse = $(relrmse)")
    return A,c,y
 end

"""
linear least squares problem function

takes: number of parameters, training set, rcut, wE and wF 
returns: loss func, QR solution
"""

function lls_basis_prob(train; maxdeg=10, rcut=6.0, wE=30.0, wF=1.0)
    species = :W #train[1].at.Z[1] #is there a case with mixed at num
    #choose a basis
    basis = get_2b_basis(species; maxdeg = maxdeg, rcut = rcut)
    
    #Ac = y problem
    A,c,y = lsq(train, basis, wE=30.0, wF=1.0)
    
    L(θ) =  norm(A * θ - y) / norm(y)

    return(L, c)
end





"""

#multi-body potential, this is the new problem with added non linearities
#Ffun has the nonlinearity which is embeded in the potential
#both problems seek to find theta to fit the potential given. 
#why don't we use a basis here as well?
#this is not really a basis, its kind of used only to make the potential
#we are fitting. this potential has the nonlinearity, while before it had 
#e^ something right? and thats why we could use the basis? ???????


#ideas for non linearities
#Ffun = ρ -> ρ[1] + sqrt(ρ[2])
#Ffun = ρ -> ρ[1] + exp(-ρ[2]^2)
#N = 3
#α = [1,2,3]
#Ffun = ρ -> sum([abs(ρ[n])^(α[n]) for n in 1:N]) 

takes: train data, nonlinearity (R^n->R^1); some weird numbers
N, maxdeg and V num params and the weights
return: loss and potential fitted with basis
"""

function combinations_prob(train, F_nonlin; N=4, maxdeg=6, V_numparams=2, wE=1, wF=1)

    species = :W #train[1].at.Z[1] #is there a case with mixed at numtrain[1].at.Z[1] #is there a case with mixed at num
    basis = ACE.Utils.ace_basis(; species = species, N = N, maxdeg = maxdeg)

    #what is the real number of parameters?
    V = FitCombiPotential(basis, F_nonlin, V_numparams)
    #need to figure out the forces at some point
    L(θ) = sum([wE^2 * abs(energy(set_params(V,θ),t.at) - t.E)^2 for t in train])
    return(L, V)

end



end