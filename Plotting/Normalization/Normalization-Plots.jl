

using Plots
pythonplot()
# using PyPlot
using JLD2 
using LinearAlgebra
using LaTeXStrings

# include("../../Helpers/FermionAlgebra.jl");
# using .FermionAlgebra;

include("../../Hamiltonians/H2.jl")
using .H2

include("../../Hamiltonians/H4.jl")
using .H4

include("../../Helpers/ChaosIndicators.jl");
using .ChaosIndicators;

include("../../Helpers/OperationsOnHamiltonian.jl");
using .OperationsOnH

colors = ["dodgerblue" "darkviolet" "limegreen" "indianred" "magenta" "darkblue" "aqua" "deeppink" "dimgray" "red" "royalblue" "slategray" "black" "lightseagreen" "forestgreen" "palevioletred"]
markers = ["o","x","v","*","H","D","s","d","P","2","|","<",">","_","+",","]
markers_line = ["-o","-x","-v","-H","-D","-s","-d","-P","-2","-|","-<","->","-_","-+","-,"]


#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

function NormalizationDepandanceOnNumberOfTries_H2_GetData()
    
    MaximalNumberOfIterations = 1000;


    L = [8,10,12,14];
    labels = ["8" "10" "12" "14"];
    S = 1/2;
    μ = 0.;
    mean = 0.;
    deviation = [0.5, 1, 2];


    norm = zeros(Float64, (length(deviation), length(L), MaximalNumberOfIterations));
    norm_avg = zeros(Float64, (length(deviation), length(L), MaximalNumberOfIterations));
    norm_avg_error = zeros(Float64, (length(deviation), length(L), MaximalNumberOfIterations));


    for t in eachindex(deviation), l in eachindex(L),  i=1:MaximalNumberOfIterations
        println(t, " ", l, " ", i);
        params = Hsyk2.GetParams_SKY2(L[l],S,μ,mean,deviation[t]);
        H₂ = Hsyk2.Ĥ₂(params, true);

        norm[t,l,i] = OperationsOnHamiltonian.GetNormOfMatrx(H₂)
        norm_avg[t,l,i] = sum(norm[t,l, 1:i])/i;
        norm_avg_error[t,l,i] = abs(norm_avg[t,l,i] - Hsyk2.AproximateAnaliticalNormOfHamiltonian(deviation[t], L[l], μ));
    end

    folder = jldopen("../Data/Norm_H2.jld2", "w");
    folder["norm"] = norm;
    folder["norm_avg"] = norm_avg;
    folder["norm_avg_error"] = norm_avg_error;
    folder["labels"] = labels;
    folder["MaximalNumberOfIterations"] = MaximalNumberOfIterations;
    close(folder)

end
   
function NormalizationDepandanceOnNumberOfTries_H2_PlotData()
    
    folder = jldopen("../Data/Norm_H2.jld2", "r")
    norm = folder["norm"]
    norm_avg = folder["norm_avg"]
    norm_avg_error = folder["norm_avg_error"]
    labels = folder["labels"]
    MaximalNumberOfIterations = folder["MaximalNumberOfIterations"]

    x = 1:MaximalNumberOfIterations;

    y1 = norm_avg[1,:,:];
    y2 = norm_avg[2,:,:];
    y3 = norm_avg[3,:,:];


    p1 = plot(x, [y1[1,:] y1[2,:] y1[3,:] y1[4,:]], label=labels, framestyle = :box, minorgrid=true, yscale=:log10, title = L"\vert t \vert=0.5, \mu=0");
    p2 = plot(x, [y2[1,:] y2[2,:] y2[3,:] y2[4,:]], label=labels, framestyle = :box, minorgrid=true, yscale=:log10, title = L"\vert t \vert=1, \mu=0");
    p3 = plot(x, [y3[1,:] y3[2,:] y3[3,:] y3[4,:]], label=labels, framestyle = :box, minorgrid=true, yscale=:log10, title = L"\vert t \vert=2, \mu=0");



    ratio = 2.5
    dim = 300
    p = plot(p1,p2,p3, layout=(1,3), legend=true, size=(Int(dim*ratio),dim))
    savefig(p, "../Images/Normalization_H2_L$(labels)_Iter$(MaximalNumberOfIterations).pdf")
    @show p


end



# NormalizationDepandanceOnNumberOfTries_H2_GetData()
# NormalizationDepandanceOnNumberOfTries_H2_PlotData()


#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

function NormalizationObtainedByAnaliticalCalculations_H2_GetData()

    L = [8,10,12];
    labels = ["8" "10" "12"];
    S = 1/2;
    μ = 0.;
    mean = 0.;
    deviation = LinRange(0,10,10);

    maxIterations = 100

    numericalNorm  = zeros(Float64, (length(L), length(deviation)));
    analiticalNorm = zeros(Float64, (length(L), length(deviation)));
    aproxAnalNorm  = zeros(Float64, (length(L), length(deviation)));


    for t in eachindex(deviation), l in eachindex(L), i in 1:maxIterations
        println(t, " ", l);
        params = Hsyk2.GetParams_SKY2(L[l],S,μ,mean,deviation[t]);
        H₂ = Hsyk2.Ĥ₂(params, true);

        numericalNorm[l,t] += OperationsOnHamiltonian.OperatorNorm(H₂)/maxIterations;
        analiticalNorm[l,t]+= Hsyk2.AnaliticalNormOfHamiltonian(params)/maxIterations;
        aproxAnalNorm[l,t] += Hsyk2.AproximateAnaliticalNormOfHamiltonian(deviation[t], params.L, params.μ)/maxIterations;
    end

    folder = jldopen("../Data/AnaliticalAndNumericalNormalization_Hsyk2.jld2", "w");
    folder["numericalNorm"] = numericalNorm;
    folder["analiticalNorm"] = analiticalNorm;
    folder["aproxAnalNorm"] = aproxAnalNorm;
    folder["labels"] = labels;
    folder["L"] = L;
    folder["deviation"] = deviation;
    folder["maxIterations"] = maxIterations;
    close(folder)

end
   
function NormalizationObtainedByAnaliticalCalculations_Hsyk2_PlotData()
    
    folder1 = jldopen("../Data/AnaliticalAndNumericalNormalization_H2.jld2", "r")
    numericalNorm = folder1["numericalNorm"]
    analiticalNorm = folder1["analiticalNorm"]
    aproxAnalNorm = folder1["aproxAnalNorm"]
    labels = folder1["labels"]
    L = folder1["L"]
    deviation = folder1["deviation"]
    maxIterations = folder1["maxIterations"]


    # println(size(deviation))
    # println(size(numericalNorm))
    # display(numericalNorm)


    # a = [numericalNorm[i,:] for i in eachindex(L)]
    # println(size(a))
    # display(a)



    p1 = plot(deviation, numericalNorm', label=labels, framestyle = :box, minorgrid=true, title = "Iterations = $(maxIterations), Numerični izračun", lc=:dodgerblue);
    p2 = plot(deviation, abs.(analiticalNorm' .- numericalNorm') .+ 10^(-16), label=labels, framestyle = :box, minorgrid=true, yscale=:log10 ,title = "Iterations = $(maxIterations), Napaka");

    xlabel!(p1, L"\vert t \vert");
    xlabel!(p2, L"\vert t \vert");
    ylabel!(p1, L"\vert\vert H_2 \vert\vert");
    ylabel!(p2, "Error");



    ratio = 2
    dim = 600
    p = plot(p1,p2,layout=(1,2), legend=true, size=(Int(dim*ratio),dim))
    savefig(p, "../Images/Normalization_H2_L$(labels).pdf")
    @show p


end



# NormalizationObtainedByAnaliticalCalculations_Hsyk2_GetData()
# NormalizationObtainedByAnaliticalCalculations_Hsyk2_PlotData()

#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

function NormalizationDepandanceOnNumberOfTries_syk4_GetData()
    
    MaximalNumberOfIterations = 100;


    L = [6,8];
    labels = ["6" "8"];
    S = 1/2;
    μ = 0.;
    mean = 0.;
    deviation = [0.5, 1, 2];


    norm = zeros(Float64, (length(deviation), length(L), MaximalNumberOfIterations));
    norm_avg = zeros(Float64, (length(deviation), length(L), MaximalNumberOfIterations));

    for u in eachindex(deviation), l in eachindex(L),  i=1:MaximalNumberOfIterations
        println(u, " ", l, " ", i);
        params = Hsyk4.GetParams_SKY4(L[l],S,μ,mean,deviation[u]);
        H₄ = Hsyk4.Ĥ₄(params, true);

        norm[u,l,i] = OperationsOnHamiltonian.GetNormOfMatrx(H₄)
        norm_avg[u,l,i] = sum(norm[u,l, 1:i])/i;
        # norm_avg_error[t,l,i] = abs(norm_avg[t,l,i] - Hsyk4.AnaliticalNormOfHamiltonian(deviation[t], L[l], μ));
    end

    folder = jldopen("Norm_syk4.jld2", "w");
    folder["norm"] = norm;
    folder["norm_avg"] = norm_avg;
    # folder["norm_avg_error"] = norm_avg_error;
    folder["labels"] = labels;
    folder["MaximalNumberOfIterations"] = MaximalNumberOfIterations;
    close(folder)

end
   
function NormalizationDepandanceOnNumberOfTries_syk4_PlotData()
    
    folder1 = jldopen("Norm_syk4.jld2", "r")
    norm = folder1["norm"]
    norm_avg = folder1["norm_avg"]
    # norm_avg_error = folder1["norm_avg_error"]
    labels = folder1["labels"]
    MaximalNumberOfIterations = folder1["MaximalNumberOfIterations"]

    x = 1:MaximalNumberOfIterations;

    y1 = norm_avg[1,:,:];
    y2 = norm_avg[2,:,:];
    y3 = norm_avg[3,:,:];


    p1 = plot(x, [y1[1,:] y1[2,:] ], label=labels, framestyle = :box, minorgrid=true, title = L"\vert U \vert =0.5, \mu=0");
    p2 = plot(x, [y2[1,:] y2[2,:] ], label=labels, framestyle = :box, minorgrid=true, title = L"\vert U \vert =1, \mu=0");
    p3 = plot(x, [y3[1,:] y3[2,:] ], label=labels, framestyle = :box, minorgrid=true, title = L"\vert U \vert =2, \mu=0");



    ratio = 2.5
    dim = 300
    p = plot(p1,p2,p3, layout=(1,3), legend=true, size=(Int(dim*ratio),dim))
    savefig(p, "Normalization_H4_L$(labels)_Iter$(MaximalNumberOfIterations).pdf")
    @show p


end



# NormalizationDepandanceOnNumberOfTries_syk4_GetData()
# NormalizationDepandanceOnNumberOfTries_syk4_PlotData()

#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

