using Pkg, JuMP, Gurobi, Random, Distributions,LinearAlgebra, DataFrames, CSV, StatsBase, Statistics,MathOptInterface

gurobi_env = Gurobi.Env()

#solve the facility location problem, given the appropriate parameters.
function facility(s,d,c_matrix,f, gurobi_env)
    n,m = size(c_matrix)


    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    @variable(final ,x[1:n, 1:m] >=0)
    @variable(final, y[1:n], Bin)
    @constraint(final, [i = 1:n], sum(x[i,k] for k = 1:m) <= s[i] * y[i])
    @constraint(final, [k = 1:m], sum(x[i,k] for i = 1:n) >= d[k])
    @objective(final, Min, sum(c_matrix[i,k] * x[i,k] for i = 1:n, k= 1:m) + sum(f[i] * y[i] for i = 1:n))

    optimize!(final)
    base_val = JuMP.objective_value(final)

    yy = JuMP.value.(y)
    xx = JuMP.value.(x)

    syy = -yy .* s
    dsyy = vcat(d, syy)
    Matx1 = Matrix(I, m, m)
    for i = 1:n-1
        mat = Matrix(I, m, m)
        Matx1 = hcat(Matx1,mat)
    end

    Matx2 = zeros(Float64, (n, m*n))
    for i = 1:n
        for j = (i-1)*m + 1:i*m
            Matx2[i,j] = -1
        end
    end
    Matx = vcat(Matx1, Matx2)

    x_opty = reshape(transpose(xx), m*n)
    tight_x1 = (abs.(Matx * x_opty .- dsyy) .<= 0.0001)
    tight_x2 = (x_opty .<= 0.0001)
    tight_x = vcat(tight_x1, tight_x2)
    tight_1 = vcat(yy, tight_x)

    return base_val, tight_1
end

#Solve the facility location problem, given a strategy st.
function facility_st(s,d,c_matrix,f,st, gurobi_env)

    n,m = size(c_matrix)
    y = st[1:n]
    idx_1 = findall(!iszero, st[n+1:2*n+m])
    idx_2 = findall(!iszero, st[2*n+m+1:end])
    Matx1 = Matrix(I, m, m)
    for i = 1:n-1
        mat = Matrix(I, m, m)
        Matx1 = hcat(Matx1,mat)
    end

    Matx2 = zeros(Float64, (n, m*n))
    for i = 1:n
        for j = (i-1)*m + 1:i*m
            Matx2[i,j] = -1
        end
    end
    Matx = vcat(Matx1, Matx2)
    cc = reshape(transpose(c_matrix), m*n)

    syy = -y .* s
    dsyy = vcat(d, syy)
    Matx_idx = (Matx)[idx_1, :]
    dsyy_idx = dsyy[idx_1]

    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env),"OutputFlag" => 0 ))
    @variable(final ,x[1:n*m])
    for q ∈ idx_2
        @constraint(final, x[q] >= 0)
    end
    @constraint(final, Matx_idx*x .>= dsyy_idx)
    @objective(final, Min, dot(cc, x) + sum(f[i] * y[i] for i = 1:n))

    optimize!(final)
    base_val = 1e4
    if termination_status(final) == MathOptInterface.OPTIMAL
        opt_x = JuMP.value.(x)
        base_val = JuMP.objective_value(final)
        if any(opt_x .+ 0.0001 .< 0) || any(Matx * opt_x .+ 0.0001 .< dsyy)
            base_val = 1e4
        end
    end





    return base_val
end





#Genereate parameter data for the facility location problem.
function generate_data(n,m)
    c_matrix = rand(Float64, (n,m)) .* 10
    s = rand(Float64, n) .* 10 .+8
    f = rand(Float64, n) .* 10

    return s,c_matrix,f
end




#Generate s_num samples, from a ball with radius r
function sample_generating(n,m, s_num, gurobi_env, r)
    feature = DataFrame()
    opt_x = DataFrame()
    optval = DataFrame()
    s,c_matrix,f = generate_data(n, m)
    d_bar = rand(m) .* 5 .+ 1

    for i = 1:s_num
        dn = randn((m +2, 1))
        norms_d=sum(dn.^2)^(0.5)
        dn = dn./norms_d
        d = (dn[1:m].* r) .+ d_bar  #This is the sampled d

        opt_val, tight = facility(s,d,c_matrix,f, gurobi_env)


        columns = reshape(d, (1, m))
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
    return feature, opt_x, optval, s,c_matrix,f
end


#N is the number of training set + test set
feature, opt_x, optval, s,c_matrix,f  = sample_generating(n,m,N,gurobi_env, r)


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
            real_val = facility_st(s,d,c_matrix,f,strategy, gurobi_env)
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

#list of ks that we would like to apply. In this code, assume that k = 1 and Q = 0.
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
        real_val = facility_st(s,d,c_matrix,f,strategy, gurobi_env)
        push!(temp, real_val)
    end
    push!(OPT_kQ, minimum(temp))
end


infeasible_OPT = sum(OPT_kQ .== 1e4)
println("Infeasible for :",k,"is","  ", infeasible_OPT)
feasible_preds_OPT = OPT_kQ[OPT_kQ.!= 1e4]
real_values_OPT = test_val[!,1][OPT_kQ .!= 1e4]
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
        real_val = facility_st(s,d,c_matrix,f,strategy, gurobi_env)
        push!(temp, real_val)
    end
    for l = 1:1
        temp_M = temp[1:1]
        OCT_all[l,i] = minimum(temp_M)
    end
end


OCT_all1 = OCT_all[1,:]


# OCT all1
infeasible_OCT1 = sum(OCT_all1 .== 1e4)
feasible_preds_OCT1 = OCT_all1[OCT_all1.!= 1e4]
real_values_OCT1 = test_val[!,1][OCT_all1 .!= 1e4]
subopt_OCT1 = (feasible_preds_OCT1 .- real_values_OCT1) ./ real_values_OCT1
num_OCT1 = sum(subopt_OCT1 .<= 0.001)

println("Infeasible for OCT_all1 is:",k,"  ", infeasible_OCT1)
println("Max subopt is(OCT_all1): ",k,"  ", maximum(subopt_OCT1))
println("Mean subopt is(OCT_all1): ",k,"  ", mean(subopt_OCT1))
println("Number of subopt smaller than 0.0001 is(OCT_all1): ",k,"  ", num_OCT1)





