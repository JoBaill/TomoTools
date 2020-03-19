include("BFGS6_gc.jl")
include("lbfgs_Jo.jl")


function subspace_iteration4(nlp,
                            x,
                            f,
                            ∇f,
                            m,
                            Z,
                            R,
                            Sₖ,
                            iterb,
                            assez_ortho;
                            normal :: Bool=true,
                            linesearch :: Function = Newarmijo_wolfe,
                            #solver :: Function = BFGS6_gc,
                            solver :: Function = Newlbfgs_Jo,
                            kwargs...)

print_with_color(:red,"iter=$iterb")

        ∇ft = Array{Float64}(length(∇f))
        M = min(iterb,m) #Sets the dimension of D for later.
        dt = Array{Float64}(length(∇f))

        #fabrication du NLPModels du probleme dans le sous-espace

        ϕ(α) = obj(nlp, x + Z*α)
        ∇ϕ(α) = Z'*grad(nlp, x + Z*α)
        function g!(α,g)
          g[:] = Z'*grad(nlp,x + Z*α)#fabrique un effet de bord
        end
println("")
print_with_color(:yellow,"size(Z)= $(size(Z))")
println("")
        D = ones(m)

        (iterb > M ) || ( D = ones(M) )

        SPR  = SimpleNLPModel(ϕ,D;g=∇ϕ,g! = g!)

        (D,DDD,Hₖ,tired, optimal,assez_ortho) = solver(SPR,Z;mem=11,
                                                        kwargs...)

        #Creation de la direction de descente
println("D = $D")
        dt = grad(nlp, x + Z * D)
        #dt = Z * DDD

        x_kp = x + Z * D
        g_kp = grad(nlp,x_kp)
println("")
print_with_color(:cyan,"norm du gradient = $(norm(∇f))")
print_with_color(:green,"Test de direction dans le subspace")
println("")

        TestDirectionSubspace(Z,g_kp,iterb,m)

println("")



        x = copy(x_kp)
        ∇f = g_kp#TEST DU X QUI A UN PEU CHANGER
        f = obj(nlp,x_kp)
        dt = -copy(g_kp)
        slope = ∇f ⋅ dt

        h = LineModel(nlp, x, dt)

println("")
print_with_color(:green,"slope du subspace = $slope")
println("")
        t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, f, slope, ∇ft; kwargs...)

println("")
print_with_color(:red,"pas du subspace = $t")
println("")
        #update de l'itere
        xt   = x + t * dt

        good_grad || (∇ft = grad!(nlp, xt, ∇ft))

        ft=obj(nlp,xt)

        yₖ    = ∇ft - ∇f

        sₖ    = xt - x
        S=norm(sₖ)
normal = true
        if !normal
          (Z,R) = MyZupdate(Sₖ,Z,R,iterb,dt,m)

        else
                  it       = mod(iterb-1,m)+1
                  Sₖ[:,it] = dt
                  Z,R=qr(Sₖ[:,1:min(iterb,m)])
        end
        iterb += 1



        return Z,R,Hₖ,xt,ft,∇ft,∇f,yₖ,sₖ,dt,iterb,assez_ortho
end     #function
