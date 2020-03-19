function MLEMNLP(nlp :: AbstractNLPModel,A,
              b,
              m,
              n;
              Affichage :: Bool=false,
              TESTING :: Bool=false,
              ShowAll :: Bool=false,
              MAXITER :: Int=100,
              epsilon :: Float64 = 1e-4)



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
    TESTING && (grad = grad(nlp, Xₖ))
    Xₖ = (Xₖ ./ N) .* (C + eps())
    ∇f = N-C

    if TESTING
        println(norm(∇f - grad))
    end

    Norme = norm(∇f)

    push!(ng,Norme)
    push!(nf, obj(nlp, Xₖ))

    i = i + 1

    if mod(i, 5) == 0
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
