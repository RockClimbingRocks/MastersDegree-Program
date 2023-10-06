using PyPlot;
using LinearAlgebra;
using JLD2;
using LaTeXStrings;
using Statistics;
using CurveFit;

include("../../Hamiltonians/H2.jl");
using .H2;
include("../../Hamiltonians/H4.jl");
using .H4;
include("../../Helpers/ChaosIndicators/ChaosIndicators.jl");
using .ChaosIndicators;


include("../../Helpers/ChaosIndicators/PrivateFunctions/SpectralFormFactor.jl");
using .SFF

colors = ["dodgerblue","darkviolet","limegreen","indianred","magenta","darkblue","aqua","deeppink","dimgray","red","royalblue","slategray","black","lightseagreen","forestgreen","palevioletred"]
markers = ["o","x","v","*","H","D","s","d","P","2","|","<",">","_","+",","]
markers_line = ["-o","-x","-v","-H","-D","-s","-d","-P","-2","-|","-<","->","-_","-+","-,"]
line_style   = ["solid", "dotted", "dashed", "dashdot"]

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["text.usetex"] = true
rcParams["font.size"] = 12
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = "Computer Modern"

rcParams["axes.titlesize"] = 17
rcParams["axes.labelsize"] = 15
rcParams["xtick.labelsize"] = 11
rcParams["ytick.labelsize"] = 11
rcParams["legend.fontsize"] = 10

rcParams["axes.grid"] = true
rcParams["axes.grid.which"] = "both"
rcParams["savefig.bbox"] = "tight"

rcParams["axes.prop_cycle"] = PyPlot.matplotlib.cycler(color=colors)



struct ReturnObject
    Es:: Vector{Float64}
    cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}}

    function ReturnObject(Es:: Vector{Float64}, cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}})
        new(Es, cis);
    end
end


function GetDirectory():: String
    dir:: String = "./Plotting/ChaosIndicators/Data";
    return dir;
end

function GetFileName(L:: Int64, N:: Int64):: String
    fileName:: String = "ChaosIndicators_L$(L)_N$(N).jld2";
    return fileName;
end

function GetGroupPath(q:: Float64, η:: Float64):: String
    groupPath = "q=$q/η=$η";
    return groupPath
end






function r_of_q__L(L′s:: Vector{Int}, q′s:: Vector{Float64}, N′s:: Vector{Int}, ax, η:: Float64 = 0.5)

    for (i,L) in enumerate(L′s)
        N = N′s[i];
        dir:: String = GetDirectory();
        fileName:: String = GetFileName(L, N);

        # println(L)
        r′s = Vector{Float64}(undef, length(q′s));
        jldopen("$dir/$fileName", "r") do file
            for (j,q) in enumerate(q′s)
                groupPath:: String = GetGroupPath(q, η); 
                # println("  $q");
                # println("  $groupPath");
                r′s[j] = mean(file["$groupPath/rs"]);
            end
            ax.plot(q′s, r′s, label=L"$L=%$(L)$", zorder=3, markers_line[i]);
            # axins.plot(q′s, t_Th′s, zorder=3, markers_line[i]);
        end
        
    end


    ax.axhline(y = 0.3863, color="black", linestyle="dashed")
    ax.axhline(y = 0.5307, color="black", linestyle="dashed")
    
    ax.legend();
    ax.set_xscale("log");
    ax.set_xlabel(L"q");
    ax.set_ylabel(L"r");
    
end


function r_of_qOverE__L(L′s:: Vector{Int}, q′s:: Vector{Float64}, N′s:: Vector{Int}, ax, ν:: Float64, connected::Bool = false, η:: Float64 = 0.5)

    for (i,L) in enumerate(L′s)
        N = N′s[i];
        dir:: String = GetDirectory();
        fileName:: String = GetFileName(L, N);

        r′s = Vector{Float64}(undef, length(q′s));
        δE′s = Vector{Float64}(undef, length(q′s));
        jldopen("$dir/$fileName", "r") do file
            for (j,q) in enumerate(q′s)
                group:: String = GetGroupPath(q, η);
                r′s[j] = mean(file["$group/rs"]);
                δE′s[j] = connected ? 1/file["$group/Kc/t_H"] : 1/file["$group/K/t_H"];
            end
            ax.plot(q′s ./ δE′s .^ (1/ν), r′s, label=L"$L=%$(L)$", zorder=3, markers_line[i]);
            # axins.plot(q′s, t_Th′s, zorder=3, markers_line[i]);
        end
    end


    ax.axhline(y = 0.3863, color="black", linestyle="dashed")
    ax.axhline(y = 0.5307, color="black", linestyle="dashed")
    
    ax.legend();
    ax.set_xscale("log");
    ax.set_xlabel(L"q / (\delta E)^{1/%$(ν)}");
    ax.set_ylabel(L"r");
    
end


function q̄_of_L(L′s:: Vector{Int}, q′s:: Vector{Float64}, N′s:: Vector{Int}, r̄:: Float64, ax, connected::Bool = false, η:: Float64 = 0.5)
    q̄′s = Vector{Float64}(undef, length(L′s));

    for (i,L) in enumerate(L′s)
        N = N′s[i];
        dir:: String = GetDirectory();
        fileName:: String = GetFileName(L, N);

        r′s = Vector{Float64}(undef, length(q′s));
        δE′s = Vector{Float64}(undef, length(q′s));
        jldopen("$dir/$fileName", "r") do file
            for (j,q) in enumerate(q′s)
                group:: String = GetGroupPath(q, η);
                r′s[j] = mean(file["$group/rs"]);
                δE′s[j] = connected ? 1/file["$group/Kc/t_H"] : 1/file["$group/K/t_H"];
            end
        end

        i2 = findall(x -> x >= r̄, r′s)[1];
        i1 = i2-1;

        k = (r′s[i1]-r′s[i2])/(q′s[i1]-q′s[i2]);
        n = r′s[i1] - k * q′s[i1];

        q̄′s[i] = (r̄-n)/k
    end


    println(q̄′s);

    ax.plot(L′s, q̄′s, label=L"$\overline{r}=%$(r̄)$", zorder=3, markers_line[1]);
    
    ax.legend();
    ax.set_yscale("log");
    ax.set_ylabel(L"\overline{q}");
    ax.set_xlabel(L"L");

end


function q̄_of_δE(L′s:: Vector{Int}, q′s:: Vector{Float64}, N′s:: Vector{Int}, r̄:: Float64, ax, connected::Bool = false, η:: Float64 = 0.5)
    q̄′s = Vector{Float64}(undef, length(L′s));
    δE′s = Vector{Float64}(undef, length(L′s));

    for (i,L) in enumerate(L′s)
        N = N′s[i];
        dir:: String = GetDirectory();
        fileName:: String = GetFileName(L, N);

        r′s = Vector{Float64}(undef, length(q′s));
        δE′s_ = Vector{Float64}(undef, length(q′s));
        jldopen("$dir/$fileName", "r") do file
            for (j,q) in enumerate(q′s)
                group:: String = GetGroupPath(q, η);
                r′s[j] = mean(file["$group/rs"]);
                δE′s_[j] = connected ? 1/file["$group/Kc/t_H"] : 1/file["$group/K/t_H"];
            end
        end

        i2 = findall(x -> x >= r̄, r′s)[1];
        i1 = i2-1;

        k = (r′s[i1]-r′s[i2])/(q′s[i1]-q′s[i2]);
        n = r′s[i1] - k * q′s[i1];
        q̄′s[i] = (r̄-n)/k


        k = (δE′s_[i1]-δE′s_[i2])/(q′s[i1]-q′s[i2]);
        n = δE′s_[i1] - k * q′s[i1];
        δE′s[i] = k*q̄′s[i]+n;
    end

    ax.plot(δE′s, q̄′s, label=L"$\overline{r}=%$(r̄)$", zorder=3, markers_line[1]);
    
    ax.legend();
    # ax.set_yscale("log");
    ax.set_ylabel(L"\overline{q}");
    ax.set_xlabel(L"\delta E");

end

function q̄_of_δE2(L′s:: Vector{Int}, q′s:: Vector{Float64}, N′s:: Vector{Int}, r̄:: Float64, ax, connected::Bool = false, η:: Float64 = 0.5)
    q̄′s = Vector{Float64}(undef, length(L′s));
    δE′s = Vector{Float64}(undef, length(L′s));

    for (i,L) in enumerate(L′s)
        N = N′s[i];
        dir:: String = GetDirectory();
        fileName:: String = GetFileName(L, N);

        r′s = Vector{Float64}(undef, length(q′s));
        δE′s_ = Vector{Float64}(undef, length(q′s));
        jldopen("$dir/$fileName", "r") do file
            for (j,q) in enumerate(q′s)
                group:: String = GetGroupPath(q, η);
                r′s[j] = mean(file["$group/rs"]);
                δE′s_[j] = connected ? 1/file["$group/Kc/t_H"] : 1/file["$group/K/t_H"];
            end
        end

        i2 = findall(x -> x >= r̄, r′s)[1];
        i1 = i2-1;

        k = (r′s[i1]-r′s[i2])/(q′s[i1]-q′s[i2]);
        n = r′s[i1] - k * q′s[i1];
        q̄′s[i] = (r̄-n)/k


        k = (δE′s_[i1]-δE′s_[i2])/(q′s[i1]-q′s[i2]);
        n = δE′s_[i1] - k * q′s[i1];
        δE′s[i] = k*q̄′s[i]+n;
    end

    ax.plot(δE′s, q̄′s, label=L"$\overline{r}=%$(r̄)$", zorder=3, markers_line[1]);
    
    ax.legend();
    ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_ylabel(L"\overline{q}");
    ax.set_xlabel(L"\delta E");

end


function k_of_r̄(L′s:: Vector{Int}, q′s:: Vector{Float64}, N′s:: Vector{Int}, r̄′s:: Vector{Float64}, ax, connected::Bool = false, η:: Float64 = 0.5)

    k′s = Vector{Float64}(undef, length(r̄′s));

    for (l,r̄) in enumerate(r̄′s)
        q̄′s = Vector{Float64}(undef, length(L′s));

        for (i,L) in enumerate(L′s)
            N = N′s[i];
            dir:: String = GetDirectory();
            fileName:: String = GetFileName(L, N);

            r′s = Vector{Float64}(undef, length(q′s));
            δE′s = Vector{Float64}(undef, length(q′s));
            jldopen("$dir/$fileName", "r") do file
                for (j,q) in enumerate(q′s)
                    group:: String = GetGroupPath(q, η);
                    r′s[j] = mean(file["$group/rs"]);
                    δE′s[j] = connected ? 1/file["$group/Kc/t_H"] : 1/file["$group/K/t_H"];
                end
            end

            i2 = findall(x -> x >= r̄, r′s)[1];
            i1 = i2-1;

            k = (r′s[i1]-r′s[i2])/(q′s[i1]-q′s[i2]);
            n = r′s[i1] - k * q′s[i1];

            q̄′s[i] = (r̄-n)/k
        end

        fit = linear_fit(L′s, log10.(q̄′s));
        # y10 = fit[1] .+ fit[2] .* L′s;
        # y= 10 .^y10
        # println(y)
        # println(fit)

        ax.set_title(L"\overline{q} = k L + n", y=0.8);

        k′s[l] = fit[2];
    end

    ax.plot(r̄′s, k′s, zorder=3);
    
    # ax.legend();
    # ax.set_yscale("log");
    ax.set_ylabel(L"k");
    ax.set_xlabel(L"\overline{r}");

end


function η_of_r̄(L′s:: Vector{Int}, q′s:: Vector{Float64}, N′s:: Vector{Int}, r̄′s:: Vector{Float64}, ax, connected::Bool = false, η:: Float64 = 0.5)

    η′s = Vector{Float64}(undef, length(r̄′s));

    for (l,r̄) in enumerate(r̄′s)
        q̄′s = Vector{Float64}(undef, length(L′s));
        δE′s = Vector{Float64}(undef, length(L′s));

        for (i,L) in enumerate(L′s)
            N = N′s[i];
            dir:: String = GetDirectory();
            fileName:: String = GetFileName(L, N);

            r′s = Vector{Float64}(undef, length(q′s));
            δE′s_ = Vector{Float64}(undef, length(q′s));
            jldopen("$dir/$fileName", "r") do file
                for (j,q) in enumerate(q′s)
                    group:: String = GetGroupPath(q, η);
                    r′s[j] = mean(file["$group/rs"]);
                    δE′s_[j] = connected ? 1/file["$group/Kc/t_H"] : 1/file["$group/K/t_H"];
                end
            end
            i2 = findall(x -> x >= r̄, r′s)[1];
            i1 = i2-1;
    
            k = (r′s[i1]-r′s[i2])/(q′s[i1]-q′s[i2]);
            n = r′s[i1] - k * q′s[i1];
            q̄′s[i] = (r̄-n)/k
    
    
            k = (δE′s_[i1]-δE′s_[i2])/(q′s[i1]-q′s[i2]);
            n = δE′s_[i1] - k * q′s[i1];
            δE′s[i] = k*q̄′s[i]+n;
        end

        fit = power_fit(δE′s, q̄′s);
        # y10 = fit[1] .+ fit[2] .* L′s;
        # y= 10 .^y10
        # println(y)
        # println(fit)

        η′s[l] = fit[2];
    end

    ax.plot(r̄′s, η′s, zorder=3);
    
    # ax.legend();
    # ax.set_yscale("log");
    ax.set_ylabel(L"\eta");
    ax.set_xlabel(L"\overline{r}");
    ax.set_title(L"\overline{q} = a (\delta E) ^\eta", y=0.75);

end

function exp_of_r̄(L′s:: Vector{Int}, q′s:: Vector{Float64}, N′s:: Vector{Int}, r̄′s:: Vector{Float64}, ax, i_param:: Int64, connected::Bool = false, η:: Float64 = 0.5)

    η′s = Vector{Float64}(undef, length(r̄′s));

    for (l,r̄) in enumerate(r̄′s)
        q̄′s = Vector{Float64}(undef, length(L′s));
        δE′s = Vector{Float64}(undef, length(L′s));

        for (i,L) in enumerate(L′s)
            N = N′s[i];
            dir:: String = GetDirectory();
            fileName:: String = GetFileName(L, N);

            r′s = Vector{Float64}(undef, length(q′s));
            δE′s_ = Vector{Float64}(undef, length(q′s));
            jldopen("$dir/$fileName", "r") do file
                for (j,q) in enumerate(q′s)
                    group:: String = GetGroupPath(q, η);
                    r′s[j] = mean(file["$group/rs"]);
                    δE′s_[j] = connected ? 1/file["$group/Kc/t_H"] : 1/file["$group/K/t_H"];
                end
            end
            i2 = findall(x -> x >= r̄, r′s)[1];
            i1 = i2-1;
    
            k = (r′s[i1]-r′s[i2])/(q′s[i1]-q′s[i2]);
            n = r′s[i1] - k * q′s[i1];
            q̄′s[i] = (r̄-n)/k
    
    
            k = (δE′s_[i1]-δE′s_[i2])/(q′s[i1]-q′s[i2]);
            n = δE′s_[i1] - k * q′s[i1];
            δE′s[i] = k*q̄′s[i]+n;
        end

        fit = exp_fit(L′s, q̄′s);
        # y10 = fit[1] .+ fit[2] .* L′s;
        # y= 10 .^y10
        # println(y)
        # println(fit)

        η′s[l] = fit[i_param];
    end

    ax.plot(r̄′s, η′s, zorder=3);
    
    # ax.legend();
    # ax.set_yscale("log");
    param = ["a", "b"];
    ax.set_ylabel(L"%$(param[i_param])");
    ax.set_xlabel(L"\overline{r}");
    ax.set_title(L"\overline{q} = a e^{b L }", y=0.75);

end


function δr_of_q(L′s:: Vector{Int}, q′s:: Vector{Float64}, N′s:: Vector{Int}, ax, η:: Float64 = 0.5)
    r′s_L = Matrix{Float64}(undef, length(L′s), length(q′s));


    for (i,L) in enumerate(L′s)
        N = N′s[i];
        dir:: String = GetDirectory();
        fileName:: String = GetFileName(L, N);

        jldopen("$dir/$fileName", "r") do file
            for (j,q) in enumerate(q′s)
                group:: String = GetGroupPath(q, η);
                r′s_L[i,j] = mean(file["$group/rs"]);
            end
        end
    end


    r̄′s = Vector{Float64}(undef, length(L′s)-1);
    q̄′s = Vector{Float64}(undef, length(L′s)-1);


    for i in 1:length(L′s)-1
        δr = r′s_L[i+1,:] .- r′s_L[i,:];
        
        ax.plot(q′s, δr, label=L"%$(L′s[i+1]) - %$(L′s[i])")
    
        δr_max = maximum(δr);
        j_δr_max = findall(x -> x ≈ δr_max, δr)[end];

        println(j_δr_max);

        δr_cutted = δr[1:j_δr_max]

        j1 = findall(x -> x<0, δr_cutted)[end]
        j2 = j1 + 1;

        k = (δr[j1]-δr[j2])/(q′s[j1]-q′s[j2]);
        n = δr[j1] - k * q′s[j1];
        q̄ = (0. - n)/k
        
        ax.scatter([q̄], [0.])
    
    end 

    # ax.plot(r̄′s, η′s, zorder=3);
    
    ax.legend();
    ax.set_xscale("log");
    ax.set_ylabel(L"\delta r");
    ax.set_xlabel(L"q");
    # ax.set_title(L"\overline{q} = a (\delta E) ^\eta", y=0.75);
end


function r̄_of_q̄(L′s:: Vector{Int}, q′s:: Vector{Float64}, N′s:: Vector{Int}, ax, η:: Float64 = 0.5)
    r′s_L = Matrix{Float64}(undef, length(L′s), length(q′s));


    for (i,L) in enumerate(L′s)
        N = N′s[i];
        dir:: String = GetDirectory();
        fileName:: String = GetFileName(L, N);

        jldopen("$dir/$fileName", "r") do file
            for (j,q) in enumerate(q′s)
                group:: String = GetGroupPath(q, η);
                r′s_L[i,j] = mean(file["$group/rs"]);
            end
        end
    end


    r̄′s = Vector{Float64}(undef, length(L′s)-1);
    q̄′s = Vector{Float64}(undef, length(L′s)-1);


    for i in 1:length(L′s)-1
        δr = r′s_L[i+1,:] .- r′s_L[i,:];

        δr_max = maximum(δr);
        j_δr_max = findall(x -> x ≈ δr_max, δr)[end];

        println(j_δr_max);

        δr_cutted = δr[1:j_δr_max]

        j1 = findall(x -> x<0, δr_cutted)[end]
        j2 = j1 + 1;

        k = (δr[j1]-δr[j2])/(q′s[j1]-q′s[j2]);
        n = δr[j1] - k * q′s[j1];
        q̄ = (0. - n)/k
        
        ax.scatter([(L′s[i+1] + L′s[i])/2], [q̄], label=L"%$(L′s[i+1]) - %$(L′s[i])")
    
    end 

    # ax.plot(r̄′s, η′s, zorder=3);
    
    ax.legend();
    # ax.set_xscale("log");
    ax.set_ylabel(L"\overline{q}");
    ax.set_xlabel(L"L");
    # ax.set_title(L"\overline{q} = a (\delta E) ^\eta", y=0.75);
end







qmin1:: Int64 = -4; # set minimal value of q, in this case that would be 10^-4
qmax1:: Int64 = 0;  # set maximal value of q, in this case that would be 10^-4
ρq1:: Int64 = 6;

Nq1:: Int64 = Int((qmax1 - qmin1)*ρq1 +1);
x1 = LinRange(qmin1, qmax1, Nq1); 
q′s1 = round.(10 .^(x1), digits=5);


qmin2:: Int64 = -2; # set minimal value of q, in this case that would be 10^-4
qmax2:: Float64 = log10(0.2);  # set maximal value of q, in this case that would be 10^-4
x2 = LinRange(qmin2, qmax2, 8)[2:end-1]; 
q′s2 = round.(10 .^(x2), digits=5);
# println(q′s)

q′s = sort(vcat(q′s1, q′s2));


function Fig1()
    L′s = [8, 10, 12, 14, 16];
    N′s = [20_000, 5000, 1500, 900, 790];
    ν = 2.;

    fig, ax = plt.subplots(ncols=2)

    r_of_q__L(L′s, q′s, N′s, ax[1]);
    r_of_qOverE__L(L′s, q′s, N′s, ax[2], ν);

    plt.show();
end
# Fig1();


function Fig2()
    L′s = [8, 10, 12, 14, 16];
    N′s = [20_000, 5000, 1500, 900, 790];
    ν′s = [1., 2., 3., 4., 5., 10, 100000]; 

    fig, ax = plt.subplots(ncols=length(ν′s));
    
    for (i, ν) in enumerate(ν′s) 
        r_of_qOverE__L(L′s, q′s, N′s, ax[i], ν);
        if i != 1
            ax[i].set_ylabel("");
        end
    end
    
    plt.show();
end
# Fig2();



function Fig3()
    L′s = [8, 10, 12, 14, 16];
    N′s = [20_000, 5000, 1500, 900, 790];
    r̄′s = [0.48, 0.49, 0.5, 0.51, 0.52];
    r̄′s2 = collect(LinRange(0.46,0.52, 100));

    fig, ax = plt.subplots(ncols=3);

    r_of_q__L(L′s, q′s, N′s, ax[1]);

    for (i, r̄) in enumerate(r̄′s) 
        q̄_of_L(L′s, q′s, N′s, r̄, ax[2]);
    end
    ax[2].legend(loc="lower left");
    axin = ax[2].inset_axes([0.54,0.54,0.45,0.45]);
    k_of_r̄(L′s, q′s, N′s, r̄′s2, axin);


    for (i, r̄) in enumerate(r̄′s) 
        q̄_of_δE(L′s, q′s, N′s, r̄, ax[3]);
    end
    ax[3].legend(loc="lower right");
    axin = ax[3].inset_axes([0.15,0.59,0.45,0.395]);
    η_of_r̄(L′s, q′s, N′s, r̄′s2, axin);


    plt.show();
end
# Fig3();


function Fig4()
    # L′s = [8, 10, 12, 14, 16];
    # N′s = [20_000, 5000, 1500, 900, 790];
    L′s = [8, 10, 12, 14, 16];
    N′s = [20_000, 5000, 1500, 900, 790];
    q′s_ = q′s1;
    r̄′s = [0.48, 0.49, 0.5, 0.51, 0.52];
    r̄′s2 = collect(LinRange(0.46,0.52, 100));

    fig, ax = plt.subplots(ncols=3);

    r_of_q__L(L′s, q′s_, N′s, ax[1]);


    for (i, r̄) in enumerate(r̄′s) 
        q̄_of_δE2(L′s, q′s_, N′s, r̄, ax[2]);
    end

    ax[3].legend(loc="lower right");
    # axin = ax[3].inset_axes([0.15,0.59,0.45,0.395]);
    η_of_r̄(L′s, q′s_, N′s, r̄′s2, ax[3]);


    plt.show();
end
# Fig4();

function Fig5()
    # L′s = [8, 10, 12, 14, 16];
    # N′s = [20_000, 5000, 1500, 900, 790];
    L′s = [12, 14, 16];
    N′s = [1500, 900, 790];
    q′s_ = q′s1;
    r̄′s = [0.48, 0.49, 0.5, 0.51, 0.52];
    r̄′s2 = collect(LinRange(0.46,0.52, 100));

    fig, ax = plt.subplots(ncols=4);

    r_of_q__L(L′s, q′s_, N′s, ax[1]);


    for (i, r̄) in enumerate(r̄′s) 
        q̄_of_δE2(L′s, q′s_, N′s, r̄, ax[2]);
    end

    ax[3].legend(loc="lower right");
    # axin = ax[3].inset_axes([0.15,0.59,0.45,0.395]);
    exp_of_r̄(L′s, q′s_, N′s, r̄′s2, ax[3], 1);
    exp_of_r̄(L′s, q′s_, N′s, r̄′s2, ax[4], 2);


    plt.show();
end
Fig5();


function Fig6()
    L′s = [8, 10, 12, 14, 16];
    N′s = [20_000, 5000, 1500, 900, 790];
    r̄′s = [0.48, 0.49, 0.5, 0.51, 0.52];
    r̄′s2 = collect(LinRange(0.46,0.52, 100));

    fig, ax = plt.subplots(ncols=3);

    r_of_q__L(L′s, q′s, N′s, ax[1]);

    for (i, r̄) in enumerate(r̄′s) 
        q̄_of_δE(L′s, q′s, N′s, r̄, ax[2]);
    end

    k_of_r̄(L′s, q′s, N′s, r̄′s2, ax[3])

    plt.show();
end
# Fig6();


function Fig7()
    L′s = [8, 10, 12, 14, 16];
    N′s = [20_000, 5000, 1500, 900, 790];


    fig, ax = plt.subplots(ncols=2);

    
    δr_of_q(L′s, q′s, N′s, ax[1]);
    r̄_of_q̄(L′s, q′s, N′s, ax[2]);
    
    plt.show();
end
# Fig7();


