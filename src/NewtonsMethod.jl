module NewtonsMethod

using LinearAlgebra, Statistics, Compat, ForwardDiff

function newtonroot(f, f_prime; x_0, tolerance = 1E-7, maxiter = 1000)
    x_old = x_0
    error = Inf
    iter = 1
    while error > tolerance && iter <= maxiter
        if f_prime(x_old) ≈ 0.0
            return (root = nothing, error = nothing, iter = nothing)
        end
        x_new = x_old - f(x_old)/f_prime(x_old)
        error = norm(x_new - x_old)
        x_old = x_new
        iter = iter + 1
    end
    if iter == maxiter+1
        return (root = nothing, error = nothing, iter = nothing)
    else
        return (root = x_old, error = error, iter = iter)
    end
end

# Applying auto-differentiation
D(f) = x -> ForwardDiff.derivative(f, x)

newtonroot(f; x_0, tolerance = 1E-7, maxiter = 1000) = newtonroot(f, D(f), x_0 = x_0, tolerance = tolerance, maxiter = maxiter)

# Export function
export newtonroot

end
