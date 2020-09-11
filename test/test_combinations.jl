

@testset "Combinations" begin
#---

using ACE, JuLIP, Test, ForwardDiff
using ACEexperimental
using ACEexperimental.Combinations: get_params, set_params!
using JuLIP.Potentials: evaluate, evaluate_d


#---

basis = ACE.Utils.ace_basis(; species = :W, N = 4, maxdeg = 6)
V = FitCombiPotential(basis, ρ -> ρ[1]+sqrt(ρ[2]), 2)

#--- parameter test s
p1 = V.params
p = get_params(V)
p[:] .= rand(length(p)) .- 0.5
set_params!(V, p)
m1 = V.params[1]
p1 = get_params(V)
println(@test p1 ≈ p)
set_params!(V, m1[:])
println(@test V.params[1] ≈ m1)


#--- Evaluation test
Rs, Zs, z0 = ACE.Random.rand_nhd(12, basis.pibasis.basis1p.J, :W)
B = evaluate(basis, Rs, Zs, z0)
ρ = V.params[1] * B
v_man = V.F(ρ)
v = evaluate(V, Rs, Zs, z0)
println(@test v_man ≈ v)

dB = evaluate_d(basis, Rs, Zs, z0)
dF = ForwardDiff.gradient(V.F, ρ)
c = dF' * V.params[1]
dv_man = (UniformScaling.(c) * dB )[:]
dv = evaluate_d(V, Rs, Zs, z0)
println(@test(dv_man ≈ dv))

# dv_lin = (V.params[1] * dB)[:]
# dv_lin ≈ dv

#--- Finite different test

at = rattle!( bulk(:W, cubic=true, pbc = false) * 2, 0.1 )
JuLIP.Testing.fdtest(V, at)

#---

U = rand(JVecF, length(Rs))
dB_dU = [ dot(dB[n, :][:], U) for n = 1:length(B) ]
for h in 0.1.^(2:10)
   Bh = evaluate(basis, Rs+h * U, Zs, z0)
   dBh_dU = (Bh - B) / h
   @show norm(dB_dU - dBh_dU, Inf)
end

#---

U = rand(JVecF, length(Rs))
v = evaluate(V, Rs, Zs, z0)
dV_dU = dot(dv, U)
for h in 0.1.^(2:10)
   Vh = evaluate(V, Rs+h * U, Zs, z0)
   dVh_dU = (Vh - v) / h
   @show norm(dV_dU - dVh_dU, Inf)
end

#---


end
