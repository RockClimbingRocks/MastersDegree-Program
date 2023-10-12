module SYK2LOC
    using Distributions;

    struct Params
        L:: Int64;
        a:: Float64;
        b:: Float64;
        t:: Matrix{Float64};
        S:: Float64;

        function Params(L:: Int64, a::Float64, b::Float64, t̲::Union{Matrix{Float64}, Missing} = missing, S::Float64 = 1/2)
            
            if isequal(t̲, missing)
                t = Matrix{Float64}(undef, (L,L))
                for i=1:L, j=1:i
                    t[i,j] = rand(Normal(0, 1)) / (1 + (abs(i-j)/b)^(2*a))^1/2;
                    
                    if i!=j
                        t[j,i] = t[i,j];
                    end
                end
            else
                t = t̲[1:L,1:L];
            end      

            new(L, a, b, t, S)
        end
    end

end