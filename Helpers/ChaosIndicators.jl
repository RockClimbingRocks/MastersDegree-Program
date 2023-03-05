
# To še raymisl kakao boš naredu... Nisem zdovoljen z obliko funckcih (input/output parameteri)..
module ChaosIndicators
    using SparseArrays;
    using LinearAlgebra;
    using Combinatorics

    include("./FermionAlgebra.jl");
    using .FermionAlgebra;

    include("../Hamiltonians/H2.jl");
    using .H2;

    include("../Hamiltonians/H4.jl");
    using .H4;

    


    global function InformationalEntropy(L:: Int64, q′s:: Vector{Float64}, numberOfIterations:: Int64)
        D = binomial(L,L÷2);
        η = 0.3;
        i₁ = Int((D - (D*η)÷1)÷2);
        i₂ = Int(i₁ + (D*η)÷1);


        ∑Sₘ = zeros(Float64, length(q′s));
        for (i,q) in enumerate(q′s)
            print("-",i)
            for j in 1:numberOfIterations 
                params2 = H2.Params(L);
                params4 = H4.Params(L);
                
                H₂ = H2.Ĥ(params2);
                H₄ = H4.Ĥ(params4);

                ϕ = eigvecs( Symmetric(Matrix(H₂ .+ q .* H₄)) );
                ϕ  = ϕ[:, i₁:i₂];
                Sₘ = sum( map(x -> - abs(x)^2 * log(abs(x)^2), ϕ) , dims=1);

                ∑Sₘ[i] += sum(Sₘ)/length(Sₘ);
            end 
        end

        return (∑Sₘ./numberOfIterations)./log(0.48*D);
    end


    global function LevelSpacingRatio(L, q′s, numberOfIterations)
        r̲ = zeros(Float64, length(q′s))
        D = binomial(L,L÷2);
        η = 0.2;
        i₁ = Int((D - (D*η)÷1)÷2);
        i₂ = Int(i₁ + (D*η)÷1);

        # Here we go throu every value of disorder
        for (i, q) in enumerate(q′s)
            print("-",i) 

            #---------------- In this for loop we average over more calculations of "r" for diffrent uniform distributions of disorer. 
            for j in 1:numberOfIterations
                params2 = H2.Params(L);
                params4 = H4.Params(L);
                
                H₂ = H2.Ĥ(params2)
                H₄ = H4.Ĥ(params4)

                λ = eigvals( Symmetric(Matrix(H₂ .+ q.*H₄)) )[i₁:i₂]
                δ = λ[2:end] .- λ[1:end-1]
                r = map((x,y) -> min(x,y)/max(x,y), δ[1:end-1], δ[2:end] )
                r̲[i] += sum(r) / length(r)
            end 
        end
        return r̲ ./ numberOfIterations
    end

end

# L = 8;
# S = 1 /2;
# μ = 0;
# deviation_t = 1.;
# deviation_U =1;
# mean = 0.

# q′s = [0.,0.1,0.3,0.5,1.,2,5,10,15]

# numberOfIterations = 100


# r = ChaosIndicators.LevelSpacingRatio(L, S, μ, deviation_t, deviation_U, mean, q′s, numberOfIterations)

# println(r)



