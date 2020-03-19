export Newarmijo_wolfe6

function Newarmijo_wolfe6(h :: LineModel,
                      h₀ :: Float64,
                      slope :: Float64,
                      g :: Array{Float64,1};
                      τ₀ :: Float64=1.0e-4,
                      τ₁ :: Float64=0.9999,
                      bk_max :: Int=50,
                      nbWM :: Int=50,
                      verbose :: Bool=false,
                         kwargs...)

    # Perform improved Armijo linesearch.
    nbk = 0
    nbW = 0
    t = 1.0

  # First try to increase t to satisfy loose Wolfe condition
  ht = obj(h, t)
  println("ht = $ht")
  slope_t = grad!(h, t, g)
  println("slope_t = $slope_t")
  while (slope_t < τ₁*slope) && (ht <= h₀ + τ₀ * t * slope) && (nbW < nbWM)
    t *= 5.0
    ht = obj(h, t)
  println("ht27 = $ht")
    slope_t = grad!(h, t, g)
  println("slope_t29 = $slope_t")
  println("on finit une iteration dans le while")

    nbW += 1
    verbose && @printf(" W  %4d  slope  %4d slopet %4d\n", nbW, slope, slope_t);
  end
println("on est sortit du while")
  hgoal = h₀ + slope * t * τ₀;
  fact = -0.8
  ϵ = 1e-10

  # Enrich Armijo's condition with Hager & Zhang numerical trick
  Armijo = (ht <= hgoal) || ((ht <= h₀ + ϵ * abs(h₀)) && (slope_t <= fact * slope))
  good_grad = true
  while !Armijo && (nbk < bk_max)
    #println("2e while")
    t *= 0.4
    #println("t=$t")
    #println(obj)
    ht = obj(h, t)
    #println("htDebut2eWhile = $ht")

    hgoal = h₀ + slope * t * τ₀;
    #println("hgoal=$hgoal")

    # avoids unused grad! calls
    Armijo = false
    good_grad = false
    if ht <= hgoal
      Armijo = true
    elseif ht <= h₀ + ϵ * abs(h₀)
      slope_t = grad!(h, t, g)
      good_grad = true
      if slope_t <= fact * slope
        Armijo = true
      end
    end

    nbk += 1
    verbose && @printf(" A  %4d  h0  %4e ht %4e\n", nbk, h₀, ht);
  end

  verbose && @printf("  %4d %4d %8e\n", nbk, nbW, t);
  return (t, good_grad, ht, nbk, nbW)
end
