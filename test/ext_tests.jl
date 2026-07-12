using Test
using EarthSciAST
using EarthSciAST: ASTExpr, VarExpr, OpExpr, Equation, Model, ModelVariable,
                   StateVariable, ParameterVariable, build_evaluator, flatten, load
using EarthSciASTSplitter
using SciMLBase
using OrdinaryDiffEqOperatorSplitting
using OrdinaryDiffEqLowOrderRK: Euler

const ESMDIR = joinpath(@__DIR__, "esm")

# a tiny scalar model for the nparts-guard test
_V(n) = VarExpr(n)
_O(o, a...) = OpExpr(o, ASTExpr[a...])
_Dt(v) = OpExpr("D", ASTExpr[_V(v)], wrt="t")
function tiny_model()
    vars = Dict{String,ModelVariable}(
        "x" => ModelVariable(StateVariable; default=1.0),
        "k" => ModelVariable(ParameterVariable; default=0.3))
    eqs = Equation[Equation(_Dt("x"), _O("+", _O("-", _O("*", _V("k"), _V("x"))), _V("k")))]
    Model(vars, eqs)
end

@testset "extensions" begin
    flat = flatten(load(joinpath(ESMDIR, "reaction_advection_1d.esm")))
    se = build_split_evaluator(flat, stencil_vs_pointwise)   # (transport, reaction)

    @testset "SciMLBase: split_ode_problem" begin
        prob = split_ode_problem(se)
        @test prob.f isa SciMLBase.SplitFunction
        @test prob.u0 == se.u0
        @test prob.tspan == se.tspan
        # SplitODEProblem's f1 + f2 reproduce the full un-split RHS
        f0!, u0, p0, _, _ = build_evaluator(flat)
        du_full = similar(u0); f0!(du_full, u0, p0, 0.0)
        d1 = similar(u0); d2 = similar(u0)
        prob.f.f1(d1, u0, se.p, 0.0); prob.f.f2(d2, u0, se.p, 0.0)
        @test isapprox(d1 .+ d2, du_full; rtol=1e-12, atol=1e-12)
        # errors on a non-2-part split
        se3 = build_split_evaluator(tiny_model(), t -> 1; nparts=3)
        @test_throws ArgumentError split_ode_problem(se3)
    end

    @testset "OrdinaryDiffEqOperatorSplitting: operator_splitting_problem" begin
        prob = operator_splitting_problem(se)
        @test prob isa OperatorSplittingProblem
        @test prob.u0 == se.u0

        dt = 0.005; N = 50
        alg = LieTrotterGodunov((Euler(), Euler()))

        # (a) the integrator's step! loop must equal a manual Lie–Trotter loop
        #     (op1 then op2, forward Euler) — a deterministic wiring/correctness check
        #     that exercises BOTH the transport and reaction operators.
        umanual = let u = copy(se.u0)
            for _ in 1:N
                d1 = similar(u); se.funcs[1](d1, u, se.p, 0.0); u = u .+ dt .* d1
                d2 = similar(u); se.funcs[2](d2, u, se.p, 0.0); u = u .+ dt .* d2
            end
            u
        end
        integ = init(operator_splitting_problem(se), alg; dt=dt)
        for _ in 1:N; step!(integ); end
        @test isapprox(integ.u, umanual; rtol=1e-9, atol=1e-12)
        @test all(isfinite, integ.u)
        @test integ.t ≈ N*dt

        # (b) constant IC ⇒ periodic advection is inert, so the split solve reduces
        #     to pure linear-decay reaction: every cell → (1 - dt*k)^N exactly (k=0.5).
        f0!, u0, p0, _, var_map = build_evaluator(flat)
        ic = Dict(name => 1.0 for name in keys(var_map))
        se_c = build_split_evaluator(flat, stencil_vs_pointwise; initial_conditions=ic)
        integ_c = init(operator_splitting_problem(se_c), alg; dt=dt)
        for _ in 1:N; step!(integ_c); end
        expected = (1 - dt*0.5)^N
        @test all(x -> isapprox(x, expected; rtol=1e-9), integ_c.u)

        # dofs count must match the number of parts
        @test_throws ArgumentError operator_splitting_problem(se; dofs=(collect(1:length(se.u0)),))
    end
end
