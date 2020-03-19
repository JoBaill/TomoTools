export arwheadJo
function arwheadJo(;n::Int=4)
       nlp=Model()
       @variable(nlp, x[i=1:n], start=1.0)
       @NLobjective(
           nlp,
           Min,
           sum((x[i]^2 + x[n]^2)^2 - 4 * x[i] + 3 for i=1:n-1)
       )
       return nlp
       end

#prob1=MathProgNLPModel(arwheadJo());
#nlp simple pour lancer des tests
