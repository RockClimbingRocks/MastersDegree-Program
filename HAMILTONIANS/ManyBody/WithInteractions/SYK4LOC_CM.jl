module SYK4LOC_CM
    using Distributions;

    struct Params
        L:: Int64;
        a:: Float64;
        b:: Float64;
        U:: Array{Float64, 4}; 
        S:: Float64;

        function Params(L:: Int64, a::Float64, b::Float64, U̲::Union{Array{Float64}, Missing} = missing, S::Float64=1/2)
            
            if isequal(U̲, missing)
                U = zeros(Float64, (L,L,L,L))

                for i=1:L, j=i+1:L, k=1:L, l=k+1:L
                    if U[i,j,k,l] == 0 && i!=j && k!=l
                        rcm = (i+j+k+l)/4;
                        r̄ = (abs(rcm-i) + abs(rcm-j) + abs(rcm-k) + abs(rcm-l))/4;

                        U[i,j,k,l] = rand(Normal(0, 1)) / (1 + (r̄/b)^(2*a))^1/2;;
                        U[j,i,l,k] = U[k,l,i,j] = U[l,k,j,i] = U[i,j,k,l];
                        U[i,j,l,k] = U[j,i,k,l] = U[k,l,j,i] = U[l,k,i,j] = -U[i,j,k,l];
                    end
                end
            else
                U = U̲[1:L, 1:L, 1:L, 1:L]
            end

            new(L, a, b, U, S)
        end
    end

end