

module Combinations

using ACE, JuLIP



mutable struct CombiRPIPotential{TF} <: AbstractCalculator
   acebasis::RPIBasis
   params::Matrix
   F::TF
end


mutable struct CombiPIPotential{TF} <: AbstractCalculator
   ace::PIPotential
   F::TF
end


end
