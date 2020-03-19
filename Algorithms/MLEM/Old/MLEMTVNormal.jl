function MLEMTV(nlp :: AbstractNLPModel,
                A,
                b,
                m,
                n;
                Affichage :: Bool=false,
                ShowAll :: Bool=false,
                TESTING :: Bool=false,
                MAXITER :: Int=100,
                epsilon :: Float64 = 1e-4)

    #β=0.0 #for now DSA Gus

    #β = nlp.lambda #fait la terre si beta = 0.5 avec 100 iter et epsi = 0.001;
    β = 1.0 #notre modele de TV contient deja lambda de maniere multiplicative


    length_Xₖ = m * n
    Xₖ        = (1 / length_Xₖ * sum(b)) * ones(length_Xₖ)
    #Xₖₚ       = (1/length_Xₖ * sum(b)) * ones(length_Xₖ)

    Norme = Inf
    i     = 0

    N = A' * ones(size(b))

    ng = []
    nf = []

    while i < MAXITER
    #    Xₖ        = Xₖₚ
    C  = A' * (b ./ (A * Xₖ))
    #    Xₖₚ       = (Xₖ ./ N) .* (C)
    #Xₖ = (Xₖ ./ (N + β * grad(nlp.TVP, Xₖ[:]))) .* (C + eps())
    ∇f = N - C + grad(nlp.TVP, Xₖ[:])
    Xₖ = (Xₖ ./ (N + β * grad(nlp.TVP, Xₖ[:]))) .* (C)

    Norme = norm(∇f,Inf)

    push!(ng,Norme)
    push!(nf, obj(nlp, Xₖ))

    i = i + 1

    #TESTING && (Grad = grad(nlp, Xₖ))
    if TESTING
        #println(norm(∇f - Grad))
        #println(nf[i])
    end

    if mod(i, 20) == 0
        println(i)
        println("")
        if ShowAll
            X = reshape(Xₖ,128,128);
            imshow(X)
        end#if
    end#if

    end #while
    #TODO message d'erreur si on sort avec MAXITER
    #return Xₖₚ

    if Affichage
        X = reshape(Xₖ, 128, 128);
        imshow(X)
    end
    return Xₖ, nf, ng

end
