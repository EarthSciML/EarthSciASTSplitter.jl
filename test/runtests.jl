using Test
using EarthSciAST
using EarthSciAST: ASTExpr, VarExpr, IntExpr, NumExpr, OpExpr, Equation, Model,
                   ModelVariable, StateVariable, ParameterVariable, flatten,
                   build_evaluator
using EarthSciASTSplitter

# ---- tiny AST builders for tests ----
V(n) = VarExpr(n)
O(op, args...) = OpExpr(op, ASTExpr[args...])
Dt(v) = OpExpr("D", ASTExpr[V(v)], wrt="t")

@testset "EarthSciASTSplitter" begin

    @testset "additive_terms / sum_terms" begin
        # n-ary and nested `+`
        e = O("+", O("+", V("a"), V("b")), V("c"))
        @test length(additive_terms(e)) == 3

        # binary minus splits into a and -b
        e2 = O("-", V("a"), V("b"))
        ts = additive_terms(e2)
        @test length(ts) == 2
        @test ts[1] == V("a")
        @test ts[2] isa OpExpr && ts[2].op == "-" && length(ts[2].args) == 1

        # unary minus distributes over the sum
        e3 = O("-", O("+", V("a"), V("b")))
        @test length(additive_terms(e3)) == 2

        # a product is a single indivisible term
        e4 = O("*", V("a"), V("b"))
        @test length(additive_terms(e4)) == 1

        # empty sum is the additive identity 0
        z = sum_terms(ASTExpr[])
        @test z isa IntExpr && z.value == 0
        @test sum_terms(ASTExpr[V("a")]) == V("a")
    end

    idx(v, a...) = OpExpr("index", ASTExpr[V(v), a...])
    Dof(inner) = OpExpr("D", ASTExpr[inner], wrt="t")

    @testset "generic rule helpers" begin
        e = O("*", V("k"), idx("u", V("i")))
        @test references(e, "k")
        @test references(e, ["q", "u"])
        @test !references(e, "z")
        @test contains_op(e, "index")
        @test contains_op(e, ("index", "aggregate"))
        @test !contains_op(e, "makearray")
    end

    @testset "spatial locality (real transport-vs-pointwise rule)" begin
        # output cell = u[3]
        ctx3 = TermContext(Dof(idx("u", IntExpr(3))), Set(["u"]))
        # neighbor reads ⇒ transport
        @test spatially_coupled(idx("u", IntExpr(4)), ctx3)
        @test spatially_coupled(O("-", idx("u", IntExpr(4)), idx("u", IntExpr(2))), ctx3)
        # same-cell read ⇒ pointwise
        @test !spatially_coupled(O("*", V("k"), idx("u", IntExpr(3))), ctx3)
        # multi-species, SAME cell ⇒ pointwise (a naive per-variable rule fails here)
        ctxc = TermContext(Dof(idx("c1", IntExpr(3))), Set(["c1", "c2"]))
        @test !spatially_coupled(O("*", idx("c1", IntExpr(3)), idx("c2", IntExpr(3))), ctxc)
        # 0-D / bare field reference ⇒ pointwise
        ctxf = TermContext(Dof(V("u")), Set(["u"]))
        @test !spatially_coupled(O("*", V("k"), V("u")), ctxf)
        # a fixed-cell read in a field equation ⇒ transport (boundary coupling)
        @test spatially_coupled(idx("u", IntExpr(1)), ctxf)
        # an aggregate that reduces a state over a range ⇒ transport (nonlocal)
        agg = OpExpr("aggregate", ASTExpr[]; output_idx=Any["i"],
                     expr_body=idx("u", V("j")))   # reads j ≠ output i
        @test spatially_coupled(agg, ctxf)
        # transport_vs_pointwise wraps it: 1 = transport, 2 = pointwise
        @test transport_vs_pointwise(idx("u", IntExpr(4)), ctx3) == 1
        @test transport_vs_pointwise(O("*", V("k"), idx("u", IntExpr(3))), ctx3) == 2
    end

    # Shared model: D(x,t) ~ -k*x + a*y ;  D(y,t) ~ x*y - k
    vars() = Dict{String,ModelVariable}(
        "x" => ModelVariable(StateVariable; default=1.5),
        "y" => ModelVariable(StateVariable; default=2.0),
        "k" => ModelVariable(ParameterVariable; default=0.5),
        "a" => ModelVariable(ParameterVariable; default=3.0),
    )
    eqs() = Equation[
        Equation(Dt("x"), O("+", O("-", O("*", V("k"), V("x"))), O("*", V("a"), V("y")))),
        Equation(Dt("y"), O("-", O("*", V("x"), V("y")), V("k"))),
    ]

    # Reference is built from the SAME source as `se` so the state naming /
    # var_map match; reconstruction is checked on index-aligned random states.
    test_us = [[2.3, 1.1], [-0.7, 4.2], [3.0, 0.2], [1.0, 1.0]]

    function assert_reconstructs(se, ref_source)
        f0!, u0, p0, _, vm0 = build_evaluator(ref_source)
        @test se.var_map == vm0
        @test se.u0 == u0
        for u in test_us
            @assert length(u) == length(u0)
            du_full = similar(u); f0!(du_full, u, p0, 0.3)
            du_sum = zeros(length(u)); tmp = similar(u)
            for f! in se.funcs
                f!(tmp, u, se.p, 0.3); du_sum .+= tmp
            end
            @test isapprox(du_sum, du_full; rtol=1e-12, atol=1e-12)
        end
    end

    @testset "split_equations structure" begin
        rule = t -> references(t, "k") ? 1 : 2
        src = eqs()
        parts = split_equations(src, rule; nparts=2)
        @test length(parts) == 2
        @test all(length(p) == 2 for p in parts)  # one D-eq per state per part
        # both parts reuse the ORIGINAL lhs objects, in order (OpExpr `==` is
        # identity, so compare by `===` to the source equations' lhs)
        @test all(p[1].lhs === src[1].lhs && p[2].lhs === src[2].lhs for p in parts)
    end

    @testset "non-derivative equations copied to every part" begin
        obs = Equation(V("w"), O("*", V("x"), V("y")))   # an observed-style eq
        eqlist = vcat(eqs(), Equation[obs])
        parts = split_equations(eqlist, t -> 1; nparts=3)
        @test length(parts) == 3
        # the observed eq appears (unchanged) in each part
        @test all(any(e -> e === obs, p) for p in parts)
    end

    @testset "end-to-end numeric reconstruction (Model path)" begin
        rule = t -> references(t, "k") ? 1 : 2
        se = build_split_evaluator(Model(vars(), eqs()), rule)
        @test se isa SplitEvaluator
        assert_reconstructs(se, Model(vars(), eqs()))
    end

    @testset "end-to-end numeric reconstruction (FlattenedSystem path)" begin
        flat = flatten(Model(vars(), eqs()))
        rule = t -> references(t, "k") ? 1 : 2
        se = build_split_evaluator(flat, rule)
        assert_reconstructs(se, flat)   # reference from the same flattened source
        # split_system returns FlattenedSystems sharing the variable set
        systems = split_system(flat, rule)
        @test length(systems) == 2
        @test all(s.state_variables === flat.state_variables for s in systems)
    end

    @testset "all-in-one-part leaves the other part identically zero" begin
        se = build_split_evaluator(Model(vars(), eqs()), t -> 2)  # everything to part 2
        u = [1.3, 2.7]
        du1 = similar(u); se.funcs[1](du1, u, se.p, 0.0)
        @test all(iszero, du1)
        assert_reconstructs(se, Model(vars(), eqs()))
    end

    @testset "3-way split reconstructs" begin
        # part by a cyclic assignment over the term index is awkward; use content:
        rule = t -> references(t, "k") ? 1 : (references(t, "a") ? 2 : 3)
        se = build_split_evaluator(Model(vars(), eqs()), rule; nparts=3)
        @test length(se.funcs) == 3
        assert_reconstructs(se, Model(vars(), eqs()))
    end

    @testset "sum_terms(additive_terms(e)) is semantically equal to e" begin
        e = O("+", O("*", V("k"), V("x")), O("-", O("*", V("a"), V("y")), V("k")))
        m1 = Model(vars(), Equation[Equation(Dt("x"), e)])
        m2 = Model(vars(), Equation[Equation(Dt("x"), sum_terms(additive_terms(e)))])
        f1!, u0, p, _, vm = build_evaluator(m1)
        f2!, _, _, _, _ = build_evaluator(m2)
        base = [1.7, 2.9, 0.4, 1.1]
        for u in (base[1:length(vm)], reverse(base[1:length(vm)]))
            d1 = similar(u); d2 = similar(u)
            f1!(d1, u, p, 0.0); f2!(d2, u, p, 0.0)
            @test isapprox(d1, d2; rtol=1e-12, atol=1e-12)
        end
    end

    @testset "invalid rule output errors" begin
        @test_throws ArgumentError split_equations(eqs(), t -> 5; nparts=2)
        @test_throws ArgumentError split_equations(eqs(), t -> 0; nparts=2)
    end

    # ---- Self-contained discretized fixtures (test/esm, no external repo) ----
    esmdir = joinpath(@__DIR__, "esm")
    nonuniform(n) = [sin(2pi*i/n) + 1.5 for i in 1:n]  # a non-uniform test state

    @testset "discretized fixtures load + split + reconstruct" begin
        for name in ("advection_1d.esm", "reaction_advection_1d.esm")
            flat = flatten(load(joinpath(esmdir, name)))
            @test flat.independent_variables == [:t]        # discretized ⇒ pure ODE
            f0!, u0, p0, _, _ = build_evaluator(flat)
            @test length(u0) == 8                            # N = 8 grid
            se = build_split_evaluator(flat, transport_vs_pointwise)
            @test length(se.funcs) == 2
            u = nonuniform(length(u0))
            du_full = similar(u); f0!(du_full, u, p0, 0.0)
            acc = zeros(length(u)); tmp = similar(u)
            for f! in se.funcs
                f!(tmp, u, se.p, 0.0); acc .+= tmp
            end
            @test isapprox(acc, du_full; rtol=1e-10, atol=1e-10)
        end
    end

    @testset "reaction+transport split isolates the operators" begin
        flat = flatten(load(joinpath(esmdir, "reaction_advection_1d.esm")))
        se = build_split_evaluator(flat, transport_vs_pointwise)
        u = nonuniform(length(se.u0))
        d_transport = similar(u); se.funcs[1](d_transport, u, se.p, 0.0)  # stencil part
        d_reaction  = similar(u); se.funcs[2](d_reaction,  u, se.p, 0.0)  # pointwise part
        @test any(!iszero, d_transport)          # transport (makearray stencil) is active
        @test any(!iszero, d_reaction)           # reaction is active
        # the pointwise part must be exactly the reaction term -k*u  (k = 0.5 in the fixture)
        @test isapprox(d_reaction, -0.5 .* u; rtol=1e-10, atol=1e-10)
        # the transport part touches neighboring cells (has off-diagonal structure)
        @test !isapprox(d_transport, d_transport[1] .* ones(length(u)); atol=1e-8)
    end
end

# Extension tests (SciMLBase SplitODEProblem, OrdinaryDiffEqOperatorSplitting)
include("ext_tests.jl")
