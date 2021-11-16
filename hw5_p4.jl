using Distributions
using Plots
using ForwardDiff
using LinearAlgebra

function newton(f, theta1; γ0=1.0,
                ftol = 1e-4, xtol = 1e-4, gtol = 1e-4, maxiter = 32,
                xrange = [-10.,10.], yrange = [-7.5,7.5], animate = true)
    fold = f(theta1)
    xold = theta1
    xchange=Inf
    fchange=Inf
    iter = 0
    stuck=0

    if animate
        c = contour(range(xrange[1],xrange[2], length=200),
            range(yrange[1],yrange[2], length=200),
            (x,y) -> f([x,y]))
        anim = Animation()
    end
    
    g = ForwardDiff.gradient(f,xold)
    
    while(iter < maxiter && ((xchange>xtol) || (fchange>ftol) || (stuck>=0) || norm(g)>gtol))
        g = ForwardDiff.gradient(f,xold)
        H = ForwardDiff.hessian(f,xold)
        Δx = - inv(H)*g
        x = xold + Δx
        fnew = f(x)

        if animate
            scatter!(c, [xold[1]],[xold[2]], markercolor=:red, legend=false, 
                xlims=xrange, ylims=yrange) 
            quiver!(c, [xold[1]],[xold[2]], quiver=([Δx[1]],[Δx[2]]), legend=false,
                xlims=xrange, ylims=yrange)
            frame(anim)
        end
        
        if (fnew>=fold)
            stuck += 1
            if (stuck>10)
                break
            end
        else
            stuck = 0
        end
        xold = x
        fold = fnew
        xchange = norm(x-xold)
        fchange = abs(fnew-fold)
        iter += 1
    end
    if (iter >= maxiter)
        info = "Maximum iterations reached"
    elseif (stuck > 10)
        info = "Failed to improve for " * string(stuck) * " iterations."
    else
        info = "Convergence."
    end
    return(fold, xold, iter, info, anim) 
end

theta1 = [-5.0, -0.2]

function func()
  x -> sin(x[1]) + sin(x[2]) + (x[1]^2 + x[2]^2) / 10
end
f = func()

result = newton(f, theta1)
gif(result[5], "newton.gif", fps=6)
