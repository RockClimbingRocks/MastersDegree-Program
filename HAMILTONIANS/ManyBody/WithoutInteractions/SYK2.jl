module SYK2
    using Distributions;

    include("../../FermionAlgebra.jl");
    using .FermionAlgebra;


    struct Params
        L:: Int64;
        t:: Matrix{Float64};
        S:: Float64;

        function Params(L:: Int64, t̲::Union{Matrix{Float64}, Missing} = missing , S::Float64 = 1/2)
            
            if isequal(t̲, missing)
                t = Matrix{Float64}(undef, (L,L))
                for i=1:L, j=1:i
                    t[i,j] = rand(Normal(0, 1));
                    
                    if i!=j
                        t[j,i] = t[i,j];
                    end
                end
            else
                t = t̲[1:L,1:L];
            end      

            new(L, t, S)
        end
    end


    function AnaliticalNormOfHamiltonian(params:: Params)
        function AnaliticalExpressionForAverageOfSqueredHamiltonian(params)
            stateNumbers = FermionAlgebra.IndecesOfSubBlock(params.L) .- 1;
            states = FermionAlgebra.WriteStateInFockSpace.(stateNumbers, params.L, params.S);

            t = params.t
            D = binomial(params.L, Int(params.L/2));
            norm = 0.

            # Covers first term of equation
            for state in states
                positionsOfParticles = findall(x -> x==1 ,state);

                for i in positionsOfParticles
                    for k in positionsOfParticles
                        norm  += t[i,i]*t[k,k]
                    end
                end
            end 

            # Covers second term of a equation
            for state in states
                positionsOfParticles = findall(x -> x==1 ,state);

                for i in positionsOfParticles
                    for j in 1:params.L
                        if i != j
                            norm  += t[i,j]*t[j,i]
                        end
                    end
                end
            end


            # Covers third term of a equation
            for state in states
                positionsOfParticles = findall(x -> x==1 ,state);

                for i in positionsOfParticles
                    for j in positionsOfParticles
                        if i != j
                            norm  -= t[i,j]*t[j,i]
                        end
                    end
                end
            end 

            return norm / (D*params.L);
        end

        function AnaliticalExpressionForSquaredAverageOfHamiiltonian(params)
            stateNumbers = FermionAlgebra.IndecesOfSubBlock(params.L) .- 1;
            states = FermionAlgebra.WriteStateInFockSpace.(stateNumbers, params.L, params.S);

            t = params.t
            D = binomial(params.L, Int(params.L/2));
            norm = 0.

            for state in states
                positionsOfParticles = findall(x -> x==1 ,state);

                for i in positionsOfParticles
                    norm += t[i,i]
                end
            end

            return ( norm / (D*√params.L) )^2
        end

        H²_avg =  AnaliticalExpressionForAverageOfSqueredHamiltonian(params);
        H_avg² =  AnaliticalExpressionForSquaredAverageOfHamiiltonian(params);

        # println("AnaliticalNormOfHamiltonian: ", params.L, " ", params.deviation)
        # println("   ",H²_avg, " - ", H_avg²)
        return H²_avg - H_avg²;
    end


    function AnaliticalNormOfHamiltonianAveraged(params:: Params, N::Int64 = Int(params.L÷2))
        if N==0 || N==params.L
            return 1.
        end
        norm = params.deviation^2 * N * (params.L - N + 1 - N/params.L);
        # norm = params.deviation^2 * params.L *(params.L + 1)/4 ;
        return √(norm);
    end

end