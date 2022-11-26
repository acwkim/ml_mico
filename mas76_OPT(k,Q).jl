using Pkg, JuMP, Gurobi, Random, Distributions,LinearAlgebra, DataFrames, CSV, StatsBase, Statistics,MathOptInterface

gurobi_env = Gurobi.Env()




df1 = CSV.read("../mas76/feature.csv",  DataFrame) #feature matrix
Reward = CSV.read("../mas76/Reward.csv",  DataFrame)#reward matrix for training
xopt = CSV.read("../mas76/xopt.csv",  DataFrame) #strategies
val = CSV.read("../mas76/optval.csv",  DataFrame) #optimal objective values


###MIPLIB Files
coef = CSV.read("mas76/coef.csv", DataFrame,header = false)
A = CSV.read("mas76/A.csv",  DataFrame,header = false)
b = CSV.read("mas76/b.csv",  DataFrame,header = false)
lb = CSV.read("mas76/lb.csv",  DataFrame,header = false)
ub = CSV.read("mas76/ub.csv", DataFrame,header = false)
intcon = CSV.read("mas76/intcon.csv",  DataFrame,header = false)
###MIPLIB Files


intcon = convert(Array, intcon)
coef = convert(Array, coef)
A = convert(Array, A)
b = convert(Array, b)
ub = convert(Array, ub)
lb = convert(Array, lb)

coef = convert(Array{Float64,2}, coef)
b_center = convert(Array{Float64,2}, b)
c_center = deepcopy(coef)

c = coef

#Solve mas76 problem for a given parameter b and a strategy st
function mas_st(b,st,gurobi_env)
    n = size(A)[2]
    ni = length(intcon)
    nc = n - ni
    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    z = st[1:ni]
    idx_1 = findall(!iszero, st[ni+1:ni+nc])
    idx_2 = findall(!iszero, st[ni+nc+1:ni+nc+nc])
    idx_3 = findall(!iszero, st[ni+nc+nc+1:end])
    A_idx = A[idx_3, :]
    b_idx = b[idx_3, :]

    @variable(final ,x[1:nc])

    for q ∈ idx_1
        @constraint(final, x[q] >= 0)
    end

    for q ∈ idx_2
        @constraint(final, 1000000000000 >= x[q])
    end
    @constraint(final, A_idx[:, 1:ni] * z .+ A_idx[:, ni+1:n] * x .<= b_idx)
    @objective(final, Min, sum(c[i+ni] * x[i] for i = 1:nc) + sum(c[i] * z[i] for i = 1:ni));

    optimize!(final)
    base_val = 1e6
    if termination_status(final) == MathOptInterface.OPTIMAL
        opt_x = JuMP.value.(x)
        base_val = JuMP.objective_value(final)
        if any(opt_x .+ 0.0001 .< 0) || any(A[:, 1:ni] * z .+ A[:, ni+1:n] * opt_x .- 0.0005 .>= b) || any(opt_x .- 0.01 .> 1000000000000)
            base_val = 1e6
        end
    end


    return base_val
end


feature = df1[1:7000, :]
test_feature = df1[7001:10000, :]
test_X = test_feature
optval = val[7001:10000, :]

x_opt = []
for i = 1:7000
    push!(x_opt, convert(Array,xopt[i,:]))
end

numu = length(unique(x_opt))

dic = countmap(x_opt)
sor = sort(collect(dic), by = x -> x[2], rev = true)

list = []
for i = 1:numu
    push!(list, sor[i][1])
end

#how many strategies to apply. We set k=10 in this code.
k_list = [1,5,10,30,60]
k = k_list[3]

grid = IAI.GridSearch(
    IAI.OptimalTreePolicyMinimizer(
        random_seed=1,
        minbucket=10,
    ),
    max_depth= [5,10],
)


if k !=1
    Q = Int(k/2)
else
    Q = 0
end

#Defining the Reward Matrix for training, which depends on the Q we choose.
Reward = convert(Array, Reward)
Reward_left = Reward[:, Q+1:end]
Reward_left = convert(DataFrame, Reward_left)



Reward_train = Reward_left
train_X = feature
test_X = test_feature


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

combine_all = []
    for i = 1:3000
    temp = []
    d = convert(Array,test_X[i,:])
    ranks = rank[i,1:k-Q]
    ranks_st = []
    for j = 1:k-Q
        index = parse(Int64, strip(ranks[j], ['x']))
        push!(ranks_st, left[index])
    end
    options = vcat(prune, ranks_st)
    for k = 1:length(options)
        strategy = options[k]
        real_val = mas_st(d,strategy, gurobi_env)
        push!(temp, real_val)
    end
    push!(combine_all, minimum(temp))
end


infeasible_OPT = sum(combine_all .== 1e6)
println("Infeasible for :",k,"is","  ", infeasible_OPT)
feasible_preds_OPT = combine_all[combine_all.!= 1e6]
real_values_OPT = optval[!,1][combine_all .!= 1e6]
subopt_OPT = (feasible_preds_OPT .- real_values_OPT) ./ real_values_OPT
num_OPT = sum(subopt_OPT .<= 0.001)
println("Max subopt is: ",k,"is","  ", maximum(subopt_OPT))
println("Mean subopt for: ",k,"is","  ", mean(subopt_OPT))
println("Number of subopt smaller than 0.0001 for: ",k,"is,","  ", num_OPT)


