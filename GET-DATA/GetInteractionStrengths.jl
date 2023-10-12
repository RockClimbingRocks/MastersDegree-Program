module IntStrength
    function λ̂′s()
        λMinMax::Vector{Tuple{Int64,Int64}} = [(-4,-3), (-3,-2), (-2,-1), (-1,0)];
        Nλ′s:: Vector{Int64} = [5,8,12,8];
        
        length(Nλ′s) != length(λMinMax) ? throw(error("Not the same size!!!")) : nothing;
        λ′s = Vector{Float64}();
        
        for (i, Nλ) in enumerate(Nλ′s)
            λmin, λmax = λMinMax[i];
            x = LinRange(λmin, λmax, Nλ);
            λ′s_i = round.(10 .^(x), digits=5);
        
            append!(λ′s, λ′s_i[1:end-1]);
            i == length(Nλ′s) ? append!(λ′s, [λ′s_i[end]]) : nothing;
        end

        return λ′s
    end
end
