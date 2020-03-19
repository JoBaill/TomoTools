function project(x,
                 l,
                 u)

    if (l==-Inf) .& (u==Inf)
      p = x
    elseif !isinf(l) .& (isinf(u))
      p = ((l.<x) .* x) + (l * (x.<=l))
    elseif !isinf(u) .& (isinf(l))
      p = ((x.<u) .* x) + ((x.>=u) * u)
    elseif ((!isinf(u)) .& (!isinf(l))) .& (l<u)
      p = ((l.<x) .& (x.<u)) .* x+((x.>=u) * u) + (l * (x.<=l))
    else
      error("lower and upper bound l,u should satisfy l<u");
    end

    return p;
end
