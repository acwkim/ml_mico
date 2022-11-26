##Assuming we have appropriate data set in a csv format

using Pkg, JuMP, Gurobi, Random, Distributions,LinearAlgebra, DataFrames, CSV, StatsBase, Statistics,MathOptInterface
gurobi_env = Gurobi.Env()

df1 = CSV.read("../binkar/feature.csv",  DataFrame) #feature matrix
Reward = CSV.read("../binkar/Reward.csv",  DataFrame)#reward matrix for training
xopt = CSV.read("../binkar/xopt.csv",  DataFrame) #strategies
val = CSV.read("../binkar/optval.csv",  DataFrame) #optimal objective values

##These files are MIPLIB files
A = CSV.read("../binkar/A.csv",DataFrame, header = false)
b = CSV.read("../binkar/b.csv", DataFrame,header = false)
c = CSV.read("../binkar/coef.csv",DataFrame, header = false)
Aeq = CSV.read("../binkar/Aeq.csv", DataFrame,header = false)
beq = CSV.read("../binkar/beq.csv", DataFrame,header = false)
lb = CSV.read("../binkar/lb.csv", DataFrame,header = false)
ub = CSV.read("../binkar/ub.csv",DataFrame, header = false)
intcon = CSV.read("../binkar/intcon.csv",DataFrame, header = false)
##

intcon = convert(Array, intcon)
A = convert(Array, A)
b = convert(Array, b)
c = convert(Array, c)
Aeq = convert(Array, Aeq)
beq = convert(Array, beq)
ub = convert(Array, ub)
lb = convert(Array, lb)


b_center = deepcopy(b)




#Solve binkar problem for a given parameter b and a strategy st
function binkar_st(b,st, gurobi_env)
    #solve binkar problem appying strategy st.
        # penalty is set to 1e6 in this case.
    m,n = size(A)[1], size(A)[2]


    integer = st[1:170]
    idx_1 = findall(!iszero, st[171:n])
    idx_2 = findall(!iszero, st[172+n:end])



    A_idx = A[idx_2, :]
    b_idx = b[idx_2, :]

    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    @variable(final ,x[1:n-170])
    for q ∈ idx_1
        @constraint(final, x[q] >= 0)
    end
    @variable(final, z[1:170])
    @constraint(final, z .== integer)
    @constraint(final, A_idx[:, 1:2128] * x .+ A_idx[:, 2129:n] * z .<= b_idx)
    @constraint(final, Aeq[:, 1:2128] * x .+ Aeq[:, 2129:n] * z .== beq)
    @objective(final, Min, sum(c[i] * x[i] for i = 1:2128) + sum(c[i+2128] * z[i] for i = 1:170))

    optimize!(final)
    base_val = 1e6
    if termination_status(final) == MathOptInterface.OPTIMAL
        opt_x = JuMP.value.(x)
        base_val = JuMP.objective_value(final)
        if any(opt_x .+ 0.0001 .< 0) || any(A[:, 1:2128] * opt_x .+ A[:, 2129:n] * integer .- 0.001 .>= b)
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

# Decide Q, given k
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

#train OPT(k,Q)
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
        real_val = binkar_st(d,strategy, gurobi_env)
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


