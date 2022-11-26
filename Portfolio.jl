using Pkg, JuMP, Gurobi, Random, Distributions,LinearAlgebra, DataFrames, CSV, StatsBase, Statistics,MathOptInterface

gurobi_env = Gurobi.Env()

#solve the portfolio optimization, given the appropriate parameters.
function portfolio(F,D,μ,γ, gurobi_env)

    n = length(μ)
    Σ = F * transpose(F) .+ D

    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    @variable(model ,x[1:n] >=0)
    @constraint(model, sum(x[i] for i = 1:n) == 1)
    @objective(model, Max, sum(x[i] *  μ[i] for  i = 1:n)- sum(Σ[i,j]*x[i]*x[j] for i = 1:n, j = 1:n)   )
    optimize!(model)
    base_val = JuMP.objective_value(model)
    opt_x =  value.(x)

    tight_1 = abs.(opt_x) .<= 0.0001

    return base_val, tight_1
end

function portfolio_st(F,D,μ,γ, st,gurobi_env)
    # apply strategy st to portfolio optimization.
    # penalty M is set to 0 in this code.

    n = length(μ)
    Σ = F * transpose(F) .+ D
    indices = findall(!iszero, st)
    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    @variable(model ,x[1:n])
    for i in indices
        @constraint(model, x[i] >=0)
    end
    @constraint(model, sum(x[i] for i = 1:n) == 1)
    @objective(model, Max, sum(x[i] *  μ[i] for  i = 1:n)- sum(Σ[i,j]*x[i]*x[j] for i = 1:n, j = 1:n)   )
    optimize!(model)
    if termination_status(model) != MathOptInterface.OPTIMAL
        return 0
    end
    base_val = JuMP.objective_value(model)
    opt_x =  value.(x)

    if any(opt_x .+ 0.0001 .< 0)
        base_val = 0
    end


    return base_val
end


#Genereate parameter data.
function generate_data(n,p)
    γ = 1
    D = zeros(Float64, (n,n))
    for i = 1:n
        D[i,i] = rand() * sqrt(p)
    end
    F = randn(Float64, n*p)
    for i = 1:n*p
        if rand() < 0.5
            F[i] = 0
        end
    end
    F = reshape(F,  (n,p))
    return γ, D, F
end




#Generate s_num samples, from a ball with radius r
function sample_generating(n,p, s_num, gurobi_env, r)
    feature = DataFrame()
    opt_x = DataFrame()
    optval = DataFrame()
    γ, D, F = generate_data(n, p)
    μ_bar = rand(n) * 3

    for i = 1:s_num
        dn = randn((n +2, 1))
        norms_d=sum(dn.^2)^(0.5)
        dn = dn./norms_d
        μ = (dn[1:n].* r) .+ μ_bar  #This is the sampled \mu

        opt_val, tight = portfolio(F,D,μ,γ, gurobi_env)


        columns = reshape(μ, (1, n))
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
    return feature, opt_x, optval, γ, D, F
end

#N is the number of training set + test set
feature, opt_x, optval, γ, D, F = sample_generating(n,p,N,gurobi_env, r)




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
            real_val = portfolio_st(F,D,μ,γ, strategy,gurobi_env)
            Reward[l,k] = real_val
        end
        if l % 100 == 0
            println(l)
        end
end






#list of ks that we would like to apply. In this code, assume that k = 1.
k_list = [1]
k = k_list[1]
Q = 0


######## OPT

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
        real_val = portfolio_st(F,D,μ,γ, strategy,gurobi_env)
        push!(temp, real_val)
    end
    push!(OPT_kQ, minimum(temp))
end


infeasible_OPT = sum(OPT_kQ .== 0)
println("Infeasible for :",k,"is","  ", infeasible_OPT)
feasible_preds_OPT = OPT_kQ[OPT_kQ.!= 0]
real_values_OPT = test_val[!,1][OPT_kQ .!= 0]
subopt_OPT = (feasible_preds_OPT .- real_values_OPT) ./ real_values_OPT
num_OPT = sum(subopt_OPT .<= 0.001)
println("Max subopt is: ",k,"is","  ", maximum(subopt_OPT))
println("Mean subopt for: ",k,"is","  ", mean(subopt_OPT))
println("Number of subopt smaller than 0.0001 for: ",k,"is,","  ", num_OPT)



###### OCT

train_y = []
for i = 1:num_train
    index = findall(x-> x == x_opt[i], list)[1]
    push!(train_y, string("x","$index"))
end

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
        real_val = portfolio_st(F,D,μ,γ, strategy,gurobi_env)
        push!(temp, real_val)
    end
    for l = 1:1
        temp_M = temp[1:1]
        OCT_all[l,i] = minimum(temp_M)
    end
end


OCT_all1 = OCT_all[1,:]


# OCT all1
infeasible_OCT1 = sum(OCT_all1 .== 0)
feasible_preds_OCT1 = OCT_all1[OCT_all1.!= 0]
real_values_OCT1 = test_val[!,1][OCT_all1 .!= 0]
subopt_OCT1 = (feasible_preds_OCT1 .- real_values_OCT1) ./ real_values_OCT1
num_OCT1 = sum(subopt_OCT1 .<= 0.001)

println("Infeasible for OCT_all1 is:",k,"  ", infeasible_OCT1)
println("Max subopt is(OCT_all1): ",k,"  ", maximum(subopt_OCT1))
println("Mean subopt is(OCT_all1): ",k,"  ", mean(subopt_OCT1))
println("Number of subopt smaller than 0.0001 is(OCT_all1): ",k,"  ", num_OCT1)

















