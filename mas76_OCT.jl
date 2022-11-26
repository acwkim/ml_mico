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
###


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

dic = countmap(x_opt)
sor = sort(collect(dic), by = x -> x[2], rev = true)

numu = length(unique(x_opt))

list = []
for i = 1:numu
    push!(list, sor[i][1])
end


treatments = []
for i = 1:7000
    index = findall(x-> x == x_opt[i], list)[1]
    push!(treatments, string("x","$index"))
end

k_list = [1,5,10,30]

numM = length(k_list)
larM = maximum(k_list)


# (train_X, train_y), (test_X, test_y) = IAI.split_data(:classification, X, treatments,
#                                                       seed=1)


grid2 = IAI.GridSearch(
    IAI.OptimalTreeClassifier(
        random_seed=1,
        minbucket=10
    ),
    max_depth =[5,10],
)





train_X = feature
test_X = test_feature


@time IAI.fit!(grid2, train_X, treatments)


rank_OCT = IAI.predict_proba(grid2, test_X)


OCT_all = zeros(Float64, (numM,3000))
for i = 1:3000
    temp = []
    d = convert(Array,test_X[i,:])
    ranks = sortperm(convert(Array,rank_OCT[i,:]), rev = true)
    ranks_st = []
    for j = 1:larM
        strat = names(rank_OCT)[ranks[j]]
        index = parse(Int64, strip(strat, ['x']))
        push!(ranks_st, list[index])
    end
    options = ranks_st
    for k = 1:length(options)
        strategy = options[k]
        real_val = mas_st(d,strategy, gurobi_env)
        push!(temp, real_val)
    end
    for l = 1:numM
        k = k_list[l]
        temp_M = temp[1:k]
        OCT_all[l,i] = minimum(temp_M)
    end
end

for p = 1:numM
    k = k_list[p]
    OCT_all1 = OCT_all[p,:]


# OCT all1
    infeasible_OCT1 = sum(OCT_all1 .== 1e6)
    feasible_preds_OCT1 = OCT_all1[OCT_all1.!= 1e6]
    real_values_OCT1 = optval[!,1][OCT_all1 .!= 1e6]
    subopt_OCT1 = (feasible_preds_OCT1 .- real_values_OCT1) ./ real_values_OCT1
    num_OCT1 = sum(subopt_OCT1 .<= 0.001)

    println("Infeasible for OCT is:",k,"  ", infeasible_OCT1)
    println("Max subopt is(OCT): ",k,"  ", maximum(subopt_OCT1))
    println("Mean subopt is(OC): ",k,"  ", mean(subopt_OCT1))
    println("Number of subopt smaller than 0.0001 is(OCT): ",k,"  ", num_OCT1)

end

