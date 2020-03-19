
function fgp_denoise_bound_init(Xobs,
                            lambda,
                            l,
                            u,
                            P1,
                            P2;
                            epsilon :: Float64=1e-3,
                            tv :: String="iso",
                            MAXITER :: Int=10,
                            M1 = [],
                            M2 = [],
                            kwargs...)

    #Define the Projection onto the box
    #voir project(x,l,u)

    m,n = size(Xobs);
    X_den = zeros(m,n);

    if isempty(P1)
        P1 = zeros((m-1), n);
        P2 = zeros(m, (n-1));
        R1 = zeros((m-1), n);
        R2 = zeros(m, (n-1));
    else
        P1 = P1;
        P2 = P2;
        R1 = P1;
        R2 = P2;
    end

    tk = 1;
    tkp1 = 1;
    count = 0;
    i = 0;


    D = zeros(m, n);
    fval = Inf;
    fun_all = [];

    while (i < MAXITER) & (count < 5)
        fold = fval;
        i += 1;
        Dold = D;

        #Computing the gradient of the objective function
        #TODO tester la differenciation automatique aussi
        Pold1 = P1;
        Pold2 = P2;
        tk = tkp1;

        D = project(Xobs - lambda * Lforward(R1, R2), l, u);

        Q1,Q2 = LtransM(D, M1, M2);

        #Taking a step towards minus of the gradient
        P1 = R1 + 1 / (8 * lambda) * Q1;
        P2 = R2 + 1 / (8 * lambda) * Q2;

        if tv == "iso"
            A = [P1;zeros(1, n)].^2 + [P2 zeros(m, 1)].^2;
            A = sqrt.(max.(A, 1));
            P1 = P1 ./ A[1:(m-1), :];
            P2 = P2 ./ A[:, 1:(n-1)];
        elseif tv == "l1"
            P1 = P1 ./ (max.(abs(P1), 1));
            P2 = P2 ./ (max.(abs(P2), 1));
        else
            error("unknown type of total variation. should be iso or l1");
        end

        tkp1 = (1 + sqrt(1 + 4 * tk^2)) / 2;

        R1 = P1 + (tk - 1) / (tkp1) * (P1 - Pold1);
        R2 = P2 + (tk - 1) / (tkp1) * (P2 - Pold2);

        re = vecnorm(D - Dold) / vecnorm(D);
        if re < epsilon
            count = count + 1;
        else
            count = 0;
        end

        C = Xobs - lambda * Lforward(P1, P2);
        PC = project(C, l, u);
        fval = -vecnorm(C - PC)^2 + vecnorm(C)^2;
        fun_all = push!(fun_all, fval);

        X_den = D;

    end

    return (X_den,i,fun_all,P1,P2)
end
