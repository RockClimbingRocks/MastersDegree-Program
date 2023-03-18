module Hgoe
    using SparseArrays;
    using Distributions;
    using LinearAlgebra;

    struct Params
        N:: Int64;
        deviation:: Real;
        mean:: Real;

        function Params(L:: Int64, mean::Real=0, deviation::Real=1/4)
            N:: Int64 = Int(2^L);
            new(N, deviation, mean)
        end
    end

    function ExactNormalization(params)
        return √(params.deviation^2*( params.N + 1 - 1/params.N));
    end

    function ApproximateNormalization(params)
        return √(params.deviation^2*( params.N + 1));
    end
    

    global function Ĥ(params:: Params) 
        N = params.N;

        A = rand(Normal(params.mean, params.deviation),(N,N));
        H̅ = @. (A + A')/√2;

        norm = ApproximateNormalization(params);
        H = H̅ / norm;
        
        return H;
    end 

end

# N=10

# params = Hgoe.Params(N)


# H = Hgoe.Ĥ(params)

# display(H)
# println()


