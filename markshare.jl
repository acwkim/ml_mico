using Pkg, JuMP, Gurobi, Random, Distributions,LinearAlgebra, DataFrames, CSV, StatsBase, Statistics,MathOptInterface
gurobi_env = Gurobi.Env()



###MIPLIB Files
coef = CSV.read("../mark/coef.csv", DataFrame,header = false)
Aeq = CSV.read("../mark/Aeq.csv",  DataFrame,header = false)
beq = CSV.read("../mark/beq.csv",  DataFrame,header = false)
lb = CSV.read("../mark/lb.csv",  DataFrame,header = false)
ub = CSV.read("../mark/ub.csv", DataFrame,header = false)
intcon = CSV.read("../mark/intcon.csv",  DataFrame,header = false)
###MIPLIB Files


intcon = convert(Array, intcon)
coef = convert(Array, coef)
Aeq = convert(Array, Aeq)
beq = convert(Array, beq)
ub = convert(Array, ub)
lb = convert(Array, lb)

coef = convert(Array{Float64,2}, coef)
c_center = deepcopy(coef)


#solve markshare problem, given a parameter c.
function mark(c,gurobi_env)
    n = size(Aeq)[2]
    ni = length(intcon)
    nc = n - ni
    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    set_optimizer_attribute(final, "MIPGap", 0.005)
@variable(final , x[1:nc] >= 0)
@variable(final, z[1:ni], Bin)
# @constraint(final, A[:, 1:nc] * x .+ A[:, nc+1:n] * z .<= b)
@constraint(final, Aeq[:, 1:nc] * x .+ Aeq[:, nc+1:n] * z .== beq)
@objective(final, Min, sum(c[i] * x[i] for i = 1:nc) + sum(c[i+nc] * z[i] for i = 1:ni));

    optimize!(final)
    base_val = JuMP.objective_value(final)

    xx = JuMP.value.(x)
    zz = round.(JuMP.value.(z))

    c1 = abs.(xx) .<= 0.01
#     c3 = abs.(A[:, 1:nc] * xx .+ A[:, nc+1:n] * zz .- b) .<= 0.01

    tight = vcat(zz,c1)

    return base_val, Int.(tight)
end

#solve markshare problem, given a parameter feat and a strategy st.
function mark_st(feat,st, gurobi_env)

    m,n = size(Aeq)[1], size(Aeq)[2]
    ni = length(intcon)
    nc = n - ni
    
    c = deepcopy(c_center)
    c[1:4] = feat

    integer = st[1:ni]
    idx_1 = findall(!iszero, st[ni+1:ni+nc])


    final = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))
    @variable(final ,x[1:nc])
    for q ∈ idx_1
        @constraint(final, x[q] >= 0)
    end
    @constraint(final, Aeq[:, 1:nc] * x .+ Aeq[:, nc+1:n] * integer .== beq)
    @objective(final, Min, sum(c[i] * x[i] for i = 1:nc) + sum(c[i+nc] * integer[i] for i = 1:ni));

    optimize!(final)
    base_val = 1e6
    if termination_status(final) == MathOptInterface.OPTIMAL
        opt_x = JuMP.value.(x)
        base_val = JuMP.objective_value(final)
        if any(opt_x .+ 0.0001 .< 0) 
            base_val = 1e6
        end
    end





    return base_val
end







#Generate s_num samples, from a ball with center c_center and radius r
function sample_generating(r,c_center, s_num, gurobi_env)
    feature = DataFrame()
    opt_x = DataFrame()
    optval = DataFrame()


    for i = 1:s_num
        #sample c from uncertainty set
        dn = randn((4 +2, 1))
        norms_d=sum(dn.^2)^(0.5)
        dn = dn./norms_d
        dn = dn[1:4] .* r
        c = deepcopy(c_center)
        c[1:4] = c[1:4] .+ dn
        feat = c[1:4]


        opt_val, tight = mark(c,gurobi_env)


        columns = reshape(feat, (1, 4))
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




