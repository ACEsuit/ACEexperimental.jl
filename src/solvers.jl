
module Solvers

using ForwardDiff, LinearAlgebra

"""
quadratice line search

line search with backtracking and quadratic interpolation. The armijo condition is used.
"""
function quadlinesearch(x, f, grad_f, param_sd, α)
    s = -grad_f(x) #p_k in the textbook
    #p = 2 #p norm to calculate below
    cnt = 0
    while(f(x+α.*s) > f(x) + (param_sd .* α .* transpose(grad_f(x)) * s)[1])
        #the minimum of the quadratic interpolation
        temp_al = α
        α = -((transpose(grad_f(x)) * s .* α^2)[1] /
            (2 * (f(x + α.*s) - f(x) - (transpose(grad_f(x)) * s .* α)[1])))
        cnt += 1
        if(α > temp_al)
            print("  alpha increased  ")
            @show α
            @show temp_al
            @show cnt
            @show 5.0 + [2,3]
        end
        if(cnt > 1000)
            print("  infinite loop  ")
            @show 5.0 + [2,3]
        end
    end
    return α
end;

initialstep(grad_loss,theta,k,alpha_prev) = 
    alpha_prev * (transpose(grad_loss(theta[k-1])) * (-grad_loss(theta[k-1])) / 
        (transpose(grad_loss(theta[k])) * (-grad_loss(theta[k]))));
initialstep(grad_loss,theta,k) = 
        2 * ((loss(theta[k]) - loss(theta[k-1])) / 
            (transpose(grad_loss(theta[k])) * (-grad_loss(theta[k]))));

"""
steepest descent algorithm

theta_not is the initial guess, it's only hardcoded because of morsepotential
quad = true uses quadratic linear search, otherwise it keeps h fixed
initstep = true uses the initial step algorithm
initminimizer is to choose which of the two equations to use, minimizer is the one that minimizes quadratic data
param_sd is the parameter in the quadratic linear search
"""

function steepestgradient(loss, theta_not ; iterations::Int=1000000 , param_sd::Real= 1/10^4 , h=0.0025 , quad = false , initstep = false , initminimizer = false , termination::Real = 1/10^4)
    grad_loss = theta -> ForwardDiff.gradient(loss, theta);
    theta = AbstractVector[]
    nabla_f_normed = Float64[] #variable size since we terminate at some point
    push!(theta, theta_not)
    #the first iteration. perfromed outside to avoid an if statement inside the loop

    if (quad)
        h = quadlinesearch(theta[1], loss, grad_loss, param_sd, h)
    end

   
    push!(theta, theta[1] - h * grad_loss(theta[1]))
    push!(nabla_f_normed, norm(grad_loss(theta[1]), Inf))

    #@show theta
    n = 1
    while(norm(grad_loss(theta[n]), Inf) > termination)
        n += 1
        if (initstep)
            if (initminimizer)
                h_0 = initialstep(grad_loss,theta,n,h)
            else
                h_0 = initialstep(grad_loss,theta,n)
            end
        end

        if (quad)
            h = quadlinesearch(theta[n], loss, grad_loss, param_sd, h_0)
        end
        push!(theta, theta[n] - h * grad_loss(theta[n]))
        
        push!(nabla_f_normed, norm(grad_loss(theta[n]), Inf))
        
        if(iterations < n)
            @show nabla_f_normed[n]
            print("max iterations reached")
            break
        end

    end
    return(nabla_f_normed,theta)
end;


"""
Adam

Adam algorithm implementation
"""

function adam(loss, theta_not; alpha::Float64=0.001 , beta1::Float64=0.9 , beta2::Float64=0.999 , epsilon::Float64=1/10^8 , termination::Real = 1/10^4, iterations::Int=10000)
    grad_loss = theta -> ForwardDiff.gradient(loss, theta);
    m = zeros(length(theta_not))
    v = 0
    t = 0
    nabla_f_normed = Float64[]
    theta = theta_not
    
    while(termination < norm(grad_loss(theta), Inf))
        t = t + 1
        gt = grad_loss(theta)
        m = beta1 .* m + (1-beta1) .* gt
        v = beta2 .* v + (1-beta2) * transpose(gt) * gt
        m_hat = m ./ (1-beta1^t)
        v_hat = v / (1-beta2^t)
        theta = theta - alpha .* m_hat / (sqrt(v_hat) + epsilon)

        push!(nabla_f_normed, norm(grad_loss(theta), Inf))
        if(t > iterations)
            @show norm(grad_loss(theta), Inf)
            print("max iterations reached")
            break
        end
    end
    return(nabla_f_normed,theta)
end

end