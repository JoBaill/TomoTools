function preconditionning_step4(nlp,
                            x,
                            f,
                            gt,
                            gprec,
                            m,
                            Z,
                            R,
                            S,
                            Hₖ,
                            yₖ,
                            sₖ,
                            d,
                            iterb;
                            normal :: Bool=true,
                            η:: Float64=0.4,
                            Θₖ :: Float64=1.0,
                            linesearch :: Function = Newarmijo_wolfe,
                            kwargs...)
println("dimension de Hₖ = $(size(Hₖ))")
println("Hₖ = $Hₖ")
println("dimension de Z = $(size(Z))")
println("dimension de R = $(size(R))")
println("dimension de gt = $(size(gt))")
println("dimension de gprec = $(size(gprec))")
        dt   = zeros(d)
        ĝt   = Z' * gt
        ĝ    = Z' * gprec
        ŷₖ   = ĝt - ĝ
println("rank de Z = $(rank(Z))")

        σₖ   = BB(sₖ,yₖ)

        βₖ   = σₖ * ((yₖ⋅gt - ŷₖ⋅ĝt) / (d⋅yₖ) - (norm(yₖ)^2 - norm(ŷₖ)^2) * d⋅gt / (d⋅yₖ).^2)
        ηₖ   = η * ( (sₖ⋅gprec) / (d⋅yₖ) )

        βₖᵖ  = max(βₖ, ηₖ)

        if iterb < 12
            Zi = Z[:,1:iterb-1]
println("dimension de Zi = $(size(Zi))")
            ĝti = ĝt[1:iterb-1]
            dt  = -Zi * (Hₖ - σₖ * eye(Hₖ)) * ĝti - σₖ * gt + βₖᵖ * d
        else
            dt   = -Z * (Hₖ - σₖ * eye(m,m)) * ĝt - σₖ * gt + βₖᵖ * d
        end
println("dt=$dt")

        #dt   = -Zi * (Hₖ - σₖ * eye(m,m)) * ĝt - σₖ * gt + βₖᵖ * d

        gprec = copy(gt)#pour calculer yₖ

        slope = gprec ⋅ dt
        h = LineModel(nlp, x, dt)

        t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, f, slope, gt; kwargs...)
println("ft=$ft")
println("t = $t")

        xt   = x + t * dt
println("xt = $xt")
# fft=obj(nlp,xt)

println("good_grad=$good_grad")

        good_grad || (ft = obj(nlp,xt))
        good_grad || (gt = grad!(nlp, xt, gt))
println("ft=$ft")
        yₖ    = gt - gprec
        sₖ    = xt - x
        βₖ    = (yₖ⋅gt) / (dt⋅yₖ) - Θₖ * ((yₖ⋅yₖ) .* (dt⋅gt))/(dt⋅yₖ).^2
        ηₖ    = η * ( (d⋅gprec) / (d⋅d) )
        βₖᵖ   = max(βₖ, ηₖ)

        it       = mod(iterb-1,m)+1
#println("S = $(S[1:4,1:it])")
println("it =$it")
println("m =$m")

println("iterb=$iterb")
println("S = $(S[:,1:min(iterb,m)])")

        S[:,it] = dt
println("rank de S = $(rank(S))")
println("S = $(S[:,1:min(iterb,m)])")

        Z,R=qr(S[:,1:min(iterb,m)])
print_with_color(:yellow,"Z_precon = $(size(Z))")

println("")
print_with_color(:cyan,"slope dans preconditionning_step: $slope, ft = $ft")


        iterb +=1

        return Z,R,xt,ft,gt,yₖ,sₖ,dt,βₖᵖ,iterb

end#function
