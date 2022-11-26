using Pkg, JuMP, Gurobi, Random, Distributions,LinearAlgebra, DataFrames, CSV, StatsBase, Statistics,MathOptInterface
gurobi_env = Gurobi.Env()


##MIPLIB files
A = CSV.read("../ns/A.csv", DataFrame, header = false)
b = CSV.read("../ns/b.csv", DataFrame,header = false)
coef = CSV.read("../ns/coef.csv", DataFrame,header = false)
Aeq = CSV.read("../ns/Aeq.csv",  DataFrame,header = false)
beq = CSV.read("../ns/beq.csv",  DataFrame,header = false)
lb = CSV.read("../ns/lb.csv",  DataFrame,header = false)
ub = CSV.read("../ns/ub.csv", DataFrame,header = false)
intcon = CSV.read("../ns/intcon.csv",  DataFrame,header = false)
##

intcon = convert(Array, intcon)
A = convert(Array, A)
b = convert(Array, b)
coef = convert(Array, coef)
Aeq = convert(Array, Aeq)
beq = convert(Array, beq)
ub = convert(Array, ub)
lb = convert(Array, lb)

coef = convert(Array{Float64,2}, coef)
b_center = convert(Array{Float64,2}, b)


b_center = reshape(b_center, 2367)

c = coef



#Solve ns problem for a given parameter b 
function ns(b,gurobi_env)
    n = size(A)[2]
    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    @variable(final , 1 >= x[1:171] >=0)
    @variable(final, z[1:1458], Bin)
    @constraint(final, A[:, 1:171] * x .+ A[:, 172:n] * z .<= b)
    @constraint(final, Aeq[:, 1:171] * x .+ Aeq[:, 172:n] * z .== beq)
    @objective(final, Min, sum(c[i] * x[i] for i = 1:171) + sum(c[i+171] * z[i] for i = 1:1458));;

    optimize!(final)
    base_val = JuMP.objective_value(final)

    xx = JuMP.value.(x)
    zz = round.(JuMP.value.(z))

    c1 = abs.(xx) .<= 0.01
    c2 = abs.(xx .- ones(length(xx))) .<= 0.01
    c3 = abs.(A[:, 1:171] * xx .+ A[:, 172:n] * zz .- b) .<= 0.01

    tight = vcat(zz,c1,c2,c3)

    return base_val, tight
end


#Solve binkar problem for a given parameter feat and a strategy st
function ns_st(feat,st, gurobi_env)

    m,n = size(A)[1], size(A)[2]

    b = deepcopy(b_center)
    b[577:576+50] = feat

    integer = st[1:1458]
    idx_1 = findall(!iszero, st[1459:1458 + 171])
    idx_2 = findall(!iszero, st[1458 + 172:1458 + 171 + 171])
    idx_3 = findall(!iszero, st[1458 + 171 + 171+1:end])


#     I_idx = Matrix(I, 1250, 1250)[idx_1,:]
    A_idx = A[idx_3, :]
    b_idx = b[idx_3, :]

    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    @variable(final ,x[1:n-1250])
    for q ∈ idx_1
        @constraint(final, x[q] >= 0)
    end
    for q ∈ idx_2
        @constraint(final, x[q] <= 1)
    end
    @variable(final, z[1:1250])
    @constraint(final, z .== integer)
    @constraint(final, A_idx[:, 1:171] * x .+ A_idx[:, 172:n] * z .<= b_idx)
    @constraint(final, Aeq[:, 1:171] * x .+ Aeq[:, 172:n] * z .== beq)
    @objective(final, Min, sum(c[i] * x[i] for i = 1:171) + sum(c[i+171] * z[i] for i = 1:1458));;

    optimize!(final)
    base_val = 1e6
    if termination_status(final) == MathOptInterface.OPTIMAL
        opt_x = JuMP.value.(x)
        base_val = JuMP.objective_value(final)
        if any(opt_x .+ 0.0001 .< 0) || any(A[:, 1:171] * opt_x .+ A[:, 172:n] * integer .- 0.001 .>= b) || any(opt_x .- 0.001 .>= 1)
            base_val = 1e6
        end
    end





    return base_val
end


#Generate s_num samples, from a ball with center b_center and radius r
function sample_generating(r,b_center, s_num, gurobi_env)
    feature = DataFrame()
    opt_x = DataFrame()
    optval = DataFrame()


    for i = 1:s_num
        #sample c from uncertainty set
        dn = randn((100 +2, 1))
        norms_d=sum(dn.^2)^(0.5)
        dn = dn./norms_d
        dn = dn[1:100] .* r
        b = deepcopy(b_center)[1:2367]
        b[577:576+100] = b[577:576+100] .+ dn
        feat = b[577:576+100]

#         b = deepcopy(b_center)[1:2367]
#         b[1:50] = b[1:50] .+ dn
#         feat = b[1:50]

#         c = deepcopy(c_center)
#         c[1:50] = c[1:50] .+ dn
#         feat = c[1:50]

        opt_val, tight = ns(b,gurobi_env)


        columns = reshape(feat, (1, 100))
        tight = round.(Int, tight)
        tight = reshape(tight, (1,length(tight)))
        val = reshape([opt_val], (1,1))

        df1 = convert(DataFrame, columns)
        df2 = convert(DataFrame, tight)
        df3 = convert(DataFrame, val)

        append!(feature, df1)
        append!(opt_x, df2)
        append!(optval, df3)
        if i % 100 == 0
            println(i)
        end
    end
    return feature, opt_x, optval
end



