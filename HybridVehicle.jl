using Pkg, JuMP, Gurobi, Random, Distributions,LinearAlgebra, DataFrames, CSV, StatsBase, Statistics, MathOptInterface

gurobi_env = Gurobi.Env()



##



d1 = [0.05, 0.30, 0.55, 0.80, 1.05, 1.30, 1.55, 1.80, 1.95, 1.70, 1.45, 1.20, 1.02, 1.12, 1.22, 1.32, 1.42, 1.52, 1.62, -1.72]
d2 = [1.73, 1.38, 1.03, 0.68, 0.33, -0.02,-0.37, -0.72, -0.94, -0.64 , -0.34,-0.04, 0.18, 0.08,-0.02,-0.12,-0.22,-0.32,-0.42,0.52]
d_original = vcat(d1,d2)

#solve the hybrid vehicle control problem, given the appropriate parameters.
function hybrid(d,gurobi_env)
    τ = 4
    α = 1
    β = 1
    γ = 1.5
    δ = 10
    Emax = 200
    Pmax = 1
    E0 = 40
    z0 = 0
    T = length(d)
    η = 0.1

    first = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env),"OutputFlag" => 0 ))
    @variable(first, E[1:T+1] >=0)
    @variable(first, Pb[1: T] >=0)
    @variable(first, Pe[1: T] >=0)
    @variable(first, z[1:(T)], Bin)
    @constraint(first, [t = 1:T], E[t+1]  == E[t] - τ*Pb[t])
    @constraint(first, [t = 1:T+1], E[t] <= Emax)
    @constraint(first, E[1] == E0)
    @constraint(first, [t = 1:T], Pe[t] <= Pmax*z[t])
    @constraint(first, [t = 1:T], Pb[t] + Pe[t] >= d[t])
    @objective(first, Min, η*(E[T+1] - Emax)^2 +sum(α*Pe[t]^2 + β*Pe[t] + γ*z[t]  + δ*(z[t] - z[t-1]) for t= 2:(T))
    + α*Pe[1]^2 + β*Pe[1] + γ*z[1]  + δ*(z[1] - z0))

    optimize!(first)

    E_opt = value.(E)
    Pb_opt = value.(Pb)
    Pe_opt = value.(Pe)
    z_opt = value.(z)

    o1 = objective_value(first)


    idx_1 = (abs.(E_opt) .<= 0.0001)
    idx_2 = (abs.(Pb_opt) .<= 0.0001)
    idx_3 = (abs.(Pe_opt) .<= 0.0001)
    idx_4 = (abs.(E_opt .- Emax) .<= 0.0001)
    idx_5 = (abs.(Pe_opt .- Pmax .* z_opt) .<= 0.0001)
    idx_6 = (abs.(Pb_opt .+ Pe_opt .- d) .<= 0.0001)
    idx_7 = Int.(z_opt .> 0.5)
    tight = vcat(idx_1, idx_2, idx_3, idx_4, idx_5, idx_6, idx_7)

    return o1, tight
end



#Solve the hybrid vehicle control problem, given a strategy st and a parameter d.
function hybrid_st(d,st, gurobi_env)
    #used penalty 1e7 in this code

    τ = 4
    α = 1
    β = 1
    γ = 1.5
    δ = 10
    Emax = 200
    Pmax = 1
    E0 = 40
    z0 = 0
    T = length(d)
    η = 0.1

    idx_1 = findall(!iszero, st[1:T+1])
    idx_2 = findall(!iszero, st[T+2:2*T+1])
    idx_3 = findall(!iszero, st[2*T+2:3*T+1])
    idx_4 = findall(!iszero, st[3*T+2:4*T+2])
    idx_5 = findall(!iszero, st[4*T+3:5*T+2])
    idx_6 = findall(!iszero, st[5*T+3:6*T+2])
    z = st[6*T+3:7*T+2]


    first = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env),"OutputFlag" => 0 ))
    @variable(first, E[1:T+1])
    @variable(first, Pb[1: T])
    @variable(first, Pe[1: T])
    for k ∈ idx_1
        @constraint(first, E[k] >=0)
    end
    for k ∈ idx_2
        @constraint(first, Pb[k] >=0)
    end
    for k ∈ idx_3
        @constraint(first, Pe[k] >=0)
    end
    @constraint(first, [t = 1:T], E[t+1]  == E[t] - τ*Pb[t])
    for k ∈ idx_4
        @constraint(first, [t = k], E[t] <= Emax)
    end
    @constraint(first, E[1] == E0)
    for k ∈ idx_5
        @constraint(first, [t = k], Pe[t] <= Pmax*z[t])
    end
    for k ∈ idx_6
        @constraint(first, [t = k], Pb[t] + Pe[t] >= d[t])
    end
    @objective(first, Min, η*(E[T+1] - Emax)^2 +sum(α*Pe[t]^2 + β*Pe[t] + γ*z[t]  + δ*(z[t] - z[t-1]) for t= 2:(T))
    + α*Pe[1]^2 + β*Pe[1] + γ*z[1]  + δ*(z[1] - z0))

    optimize!(first)



    base_val = 1e7

    if termination_status(first) == MathOptInterface.OPTIMAL
        base_val = JuMP.objective_value(first)
        E_opt = value.(E)
        Pb_opt = value.(Pb)
        Pe_opt = value.(Pe)



        idx1 = (E_opt) .< 0
        idx2 = (Pb_opt) .< 0
        idx3 = (Pe_opt) .< 0
        idx4 = E_opt  .- Emax .- 0.0001 .>= 0
        idx5 = Pe_opt .- Pmax .* z .- 0.0001 .>= 0
        idx6 = Pb_opt .+ Pe_opt .- d .+ 0.0001 .<= 0
        idx = sum(idx1) + sum(idx2) + sum(idx3) + sum(idx4) + sum(idx5) + sum(idx6)
        if idx != 0
            base_val = 1e7
        end
    end





    return base_val
end





#Generate s_num samples, from a ball with radius d_r
function sample_generating(d_r,d_bar,s_num,gurobi_env)
    feature = DataFrame()
    opt_x = DataFrame()
    optval = DataFrame()
    T = length(d_bar)

    for i = 1:s_num

        dn = randn((T+2, 1))
        norms_d=sum(dn.^2)^(0.5)
        dn = dn./norms_d
        d = (dn[1:T].*d_r) .+ d_bar #This is the sampled d

        opt_val, tight = hybrid(d,gurobi_env)

        columns = reshape(d, (1, T))
        tight = round.(Int, tight)
        tight = reshape(tight, (1,length(tight)))
        val = reshape([opt_val], (1,1))

        df1 = DataFrame(columns, :auto)
        df2 = DataFrame(tight, :auto)
        df3 = DataFrame(val, :auto)

        append!(feature, df1)
        append!(opt_x, df2)
        append!(optval, df3)

        if i % 100 == 0
            println(i)
        end
    end
    return feature, opt_x, optval
end

#N is the number of training set + test set

feature,opt_x,optval  = sample_generating(d_r,d_bar,N,gurobi_env)


num_train = Int(N * 0.7)
num_test = N - num_train

train_X = feature[1:num_train, :]
test_X = feature[num_train+1:end, :]
test_val = optval[num_train+1:N, :]






x_opt = []
for i = 1:num_train
    push!(x_opt, Array(opt_x[i,:]))
end

strategies = unique(x_opt)
numu = length(strategies)
dic = countmap(x_opt)
sor = sort(collect(dic), by = x -> x[2], rev = true)

list = []
for i = 1:numu
    push!(list, sor[i][1])
end




#Generate Reward Matrix
Reward = ones(Float64, (num_train, numu))

for l = 1:num_train

        d = Array(feature[l,:])


        for k = 1:numu
            strategy = list[k]
            real_val = hybrid_st(d,strategy, gurobi_env)
            Reward[l,k] = real_val
        end
        if l % 100 == 0
            println(l)
        end
end





train_y = []
for i = 1:num_train
    index = findall(x-> x == x_opt[i], list)[1]
    push!(train_y, string("x","$index"))
end

#list of ks that we would like to apply. In this code, assume that k = 1, Q = 0.
k_list = [1]
k = k_list[1]
Q = 0

#Defining the Reward Matrix for training, which depends on the Q we choose.
Reward_left = Reward[:, Q+1:end]
Reward_train = DataFrame(Reward_left, :auto)



grid = IAI.GridSearch(
    IAI.OptimalTreePolicyMinimizer(
        random_seed=1,
        minbucket=10,
    ),
    max_depth= [5,10],
)


IAI.fit!(grid, train_X, Reward_train, train_proportion=0.5);
rank = IAI.predict_treatment_rank(grid, test_X)


prune = []
for i = 1:Q
    push!(prune, sor[i][1])
end

left = []
for i = Q+1:numu
    push!(left, sor[i][1])
end

OPT_kQ = []
    for i = 1:num_test
    temp = []
    d = Array(test_X[i,:])
    ranks = rank[i,1:k-Q]
    ranks_st = []
    for j = 1:k-Q
        index = parse(Int64, strip(ranks[j], ['x']))
        push!(ranks_st, left[index])
    end
    options = vcat(prune, ranks_st)
    for j = 1:k
        strategy = options[j]
        real_val = hybrid_st(d,strategy, gurobi_env)
        push!(temp, real_val)
    end
    push!(OPT_kQ, minimum(temp))
end


infeasible_OPT = sum(OPT_kQ .== 1e7)
println("Infeasible for :",k,"is","  ", infeasible_OPT)
feasible_preds_OPT = OPT_kQ[OPT_kQ.!= 1e7]
real_values_OPT = test_val[!,1][OPT_kQ .!= 1e7]
subopt_OPT = (feasible_preds_OPT .- real_values_OPT) ./ real_values_OPT
num_OPT = sum(subopt_OPT .<= 0.001)
println("Max subopt is: ",k,"is","  ", maximum(subopt_OPT))
println("Mean subopt for: ",k,"is","  ", mean(subopt_OPT))
println("Number of subopt smaller than 0.0001 for: ",k,"is,","  ", num_OPT)



#Train OCT
grid2 = IAI.GridSearch(
    IAI.OptimalTreeClassifier(
        random_seed=1,
        minbucket=10
    ),
    max_depth =[5,10],
)

IAI.fit!(grid2, train_X, train_y)

rank_OCT = IAI.predict_proba(grid2, test_X)

OCT_all = zeros(Float64, (1,N-num_train))
for i = 1:N-num_train
    temp = []
    d = Array(test_X[i,:])
    ranks = sortperm(Array(rank_OCT[i,:]), rev = true)
    ranks_st = []
    for j = 1:k
        strat = names(rank_OCT)[ranks[j]]
        index = parse(Int64, strip(strat, ['x']))
        push!(ranks_st, list[index])
    end
    options = ranks_st
    for j = 1:length(options)
        strategy = options[j]
        real_val = hybrid_st(d,strategy, gurobi_env)
        push!(temp, real_val)
    end
    for l = 1:1
        temp_M = temp[1:1]
        OCT_all[l,i] = minimum(temp_M)
    end
end


OCT_all1 = OCT_all[1,:]


# OCT all1
infeasible_OCT1 = sum(OCT_all1 .== 1e7)
feasible_preds_OCT1 = OCT_all1[OCT_all1.!= 1e7]
real_values_OCT1 = test_val[!,1][OCT_all1 .!= 1e7]
subopt_OCT1 = (feasible_preds_OCT1 .- real_values_OCT1) ./ real_values_OCT1
num_OCT1 = sum(subopt_OCT1 .<= 0.001)

println("Infeasible for OCT_all1 is:",k,"  ", infeasible_OCT1)
println("Max subopt is(OCT_all1): ",k,"  ", maximum(subopt_OCT1))
println("Mean subopt is(OCT_all1): ",k,"  ", mean(subopt_OCT1))
println("Number of subopt smaller than 0.0001 is(OCT_all1): ",k,"  ", num_OCT1)













