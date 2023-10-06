module OBSYK2
    using Distributions;
    using LinearAlgebra;
    using StaticArrays;

    include("../../Helpers/FermionAlgebra.jl");
    using .FermionAlgebra;


    struct Params
        L:: Int64;
        t:: Matrix{Real};
        S:: Real;
        μ:: Float64;
        deviation:: Real;
        mean:: Real;

        function Params(L:: Int64, t̲::Union{Matrix{Float64}, Missing} = missing , S::Real = 1/2, μ:: Real = 0., mean::Real=0, deviation::Real=1)
            
            if isequal(t̲, missing)
                t = Matrix{Float64}(undef, (L,L))
                for i=1:L, j=1:i
                    t[i,j] = rand(Normal(mean, deviation));
                    
                    if i!=j
                        t[j,i] = t[i,j];
                    end
                end
            else
                t = t̲[1:L,1:L];
            end      

            new(L, t, S, μ, deviation, mean)
        end
    end


    global function AnaliticalNormOfHamiltonianAveraged(params:: Params)
        norm = params.deviation^2 * params.L *(params.L + 1)/4 ;
        return √(norm);
    end



    global function Ĥ(params:: Params)
        H0 = AnaliticalNormOfHamiltonianAveraged(params);
        H  = @. params.t / H0;
        return H;
    end 

    global function Ĥ!(params:: Params) 
        H0 = AnaliticalNormOfHamiltonianAveraged(params);
        map!(x -> x/H0, H,  params.t );
    end 

end


