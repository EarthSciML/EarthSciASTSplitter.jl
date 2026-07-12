"""
    EarthSciASTSplitter

Split an [EarthSciAST](https://github.com/EarthSciML/EarthSciAST) system of
equations into several sub-systems by **user-defined rules**, for use with
operator-splitting / IMEX time integration
([`SplitODEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/solvers/split_ode_solve/),
[OrdinaryDiffEqOperatorSplitting.jl](https://github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl)).

The package operates on the **post-discretization** system — i.e. after any
spatial operators have been lowered to grid/stencil array expressions (in
EarthSciAST v0.8+, by `expression_template_imports`), so every equation is an
ODE `D(x, t) ~ rhs` over scalar/gridded state and the whole thing is consumed by
`EarthSciAST.build_evaluator`.

## How it works

For each time-derivative equation `D(x, t) ~ rhs`, the right-hand side is
decomposed into its **additive terms** (summands). A user-supplied `rule`
assigns each term to one of `nparts` sub-systems. Every sub-system keeps the
*identical* variable set (states, parameters, observeds) — only the split
right-hand sides differ, with an absent contribution becoming `0`. Because
`build_evaluator` derives `u0`, `p`, and the state-index `var_map` from the
variables (not the equations), the resulting per-part RHS closures all act on
one shared state vector `u` and satisfy `f_full = Σ f_part`.

Non-derivative equations (observed-variable definitions, `ic` initial
conditions) are copied unchanged into every part, since each part must be able
to evaluate the shared observeds.

## Quick start

```julia
using EarthSciAST, EarthSciASTSplitter

flat = flatten(load("model.esm"))          # or flatten(model)

# A rule maps one additive term (an `ASTExpr`) to a part index in 1:nparts.
# Example: put stencil/transport terms in part 1 (implicit), the rest in part 2.
rule = term -> contains_op(term, STENCIL_OPS) ? 1 : 2

se = build_split_evaluator(flat, rule)      # nparts = 2 by default
# se.funcs :: NTuple of in-place f!(du,u,p,t); se.u0, se.p, se.tspan, se.var_map

# IMEX (DiffEqDocs): du/dt = f1(u,p,t) + f2(u,p,t), f1 implicit.
using OrdinaryDiffEqBDF          # or the SciMLBase extension helper below
prob = SplitODEProblem(se.funcs[1], se.funcs[2], se.u0, se.tspan, se.p)

# With SciMLBase loaded, `split_ode_problem(se)` builds the same problem:
prob = split_ode_problem(se)
```

See [`split_system`](@ref), [`split_equations`](@ref),
[`build_split_evaluator`](@ref), and the rule helpers [`references`](@ref),
[`contains_op`](@ref), [`is_spatial_derivative`](@ref).
"""
module EarthSciASTSplitter

using EarthSciAST
using EarthSciAST: ASTExpr, IntExpr, VarExpr, OpExpr, Equation, Model,
                   FlattenedSystem, EsmFile, flatten, build_evaluator

export additive_terms, sum_terms,
       split_equations, split_system, build_split_evaluator, SplitEvaluator,
       split_ode_problem, operator_splitting_problem,
       references, contains_op, is_spatial_derivative, has_stencil_op,
       STENCIL_OPS, SPATIAL_DERIVATIVE_OPS,
       stencil_vs_pointwise, spatial_vs_pointwise

"""
    split_ode_problem(se::SplitEvaluator; kwargs...)

Assemble a `SciMLBase.SplitODEProblem` from a 2-part [`SplitEvaluator`](@ref):
`du/dt = se.funcs[1](u,p,t) + se.funcs[2](u,p,t)`, with `funcs[1]` the implicit
part. Available only when `SciMLBase` (or a solver package that re-exports it)
is loaded; provided by the `EarthSciASTSplitterSciMLBaseExt` package extension.
`kwargs` are forwarded to the `SplitODEProblem` constructor.
"""
function split_ode_problem end

"""
    operator_splitting_problem(se::SplitEvaluator; dofs=nothing)

Assemble an `OrdinaryDiffEqOperatorSplitting.OperatorSplittingProblem` from a
[`SplitEvaluator`](@ref): one sub-operator per part, wrapped in a
`GenericSplitFunction`. By default every operator acts on all state components
(`dofs = ntuple(_->1:length(se.u0), nparts)`); pass `dofs` (a tuple of index
vectors, one per part) to restrict them. Solve with a splitting algorithm, e.g.
`init(prob, LieTrotterGodunov((Euler(), Euler())); dt=…)` and iterate with
`TimeChoiceIterator`. Available only when `OrdinaryDiffEqOperatorSplitting` (and
`SciMLBase`) are loaded; provided by the
`EarthSciASTSplitterOperatorSplittingExt` package extension.
"""
function operator_splitting_problem end

# ---------------------------------------------------------------------------
# Operator vocabularies (heuristics; users are free to define their own rules)
# ---------------------------------------------------------------------------

"""
    STENCIL_OPS

Array/stencil-producing operator names. A term containing one of these is, in a
template-discretized system, almost always a transport/finite-difference term
(the lowered form of a spatial derivative) rather than a pointwise reaction
term. Intended as a starting heuristic for [`stencil_vs_pointwise`](@ref); the
authoritative classification is whatever `rule` you pass.
"""
const STENCIL_OPS = ("makearray", "aggregate", "arrayop")

"""
    SPATIAL_DERIVATIVE_OPS

Continuous spatial-differential operator names (pre-discretization). Used by
[`is_spatial_derivative`](@ref) / [`spatial_vs_pointwise`](@ref) to split a
system *before* its spatial operators have been lowered to stencils.
"""
const SPATIAL_DERIVATIVE_OPS = ("grad", "div", "laplacian")

# ---------------------------------------------------------------------------
# Additive decomposition
# ---------------------------------------------------------------------------

_neg(t::ASTExpr) = OpExpr("-", ASTExpr[t])

"""
    additive_terms(expr::ASTExpr) -> Vector{ASTExpr}

Decompose `expr` into the list of additive summands it is a sum of. Descends
through n-ary `+`, binary `-` (`a - b` → `[…a…, -(…b…)]`), and unary `-`
(negation distributed over the operand's summands). Any other node is returned
as a single, indivisible term.

`sum_terms(additive_terms(e))` is semantically equal to `e` (it may differ
structurally, e.g. `a - b` becomes `a + (-b)`).
"""
function additive_terms(expr::ASTExpr)::Vector{ASTExpr}
    if expr isa OpExpr
        if expr.op == "+"
            terms = ASTExpr[]
            for a in expr.args
                append!(terms, additive_terms(a))
            end
            return terms
        elseif expr.op == "-" && length(expr.args) == 2
            return vcat(additive_terms(expr.args[1]),
                        ASTExpr[_neg(t) for t in additive_terms(expr.args[2])])
        elseif expr.op == "-" && length(expr.args) == 1
            return ASTExpr[_neg(t) for t in additive_terms(expr.args[1])]
        end
    end
    return ASTExpr[expr]
end

"""
    sum_terms(terms) -> ASTExpr

Combine `terms` back into a single expression with a left-folded binary `+`.
An empty collection yields the additive identity `IntExpr(0)`.
"""
function sum_terms(terms::AbstractVector{<:ASTExpr})::ASTExpr
    isempty(terms) && return IntExpr(0)
    length(terms) == 1 && return terms[1]
    return foldl((a, b) -> OpExpr("+", ASTExpr[a, b]), terms)
end

# ---------------------------------------------------------------------------
# Rule helpers
# ---------------------------------------------------------------------------

"""
    references(expr::ASTExpr, name::AbstractString) -> Bool
    references(expr::ASTExpr, names) -> Bool

`true` if `expr` contains a `VarExpr` whose name equals `name` (or is in the
iterable `names`). Handy for rules that partition by which variables a term
touches.
"""
references(expr::ASTExpr, name::AbstractString)::Bool =
    references(expr, (String(name),))

function references(expr::ASTExpr, names)::Bool
    nameset = names isa AbstractSet ? names : Set(String(n) for n in names)
    _references(expr, nameset)
end

function _references(expr::ASTExpr, names)::Bool
    expr isa VarExpr && return expr.name in names
    if expr isa OpExpr
        for a in expr.args
            _references(a, names) && return true
        end
    end
    return false
end

"""
    contains_op(expr::ASTExpr, op::AbstractString) -> Bool
    contains_op(expr::ASTExpr, ops) -> Bool

`true` if any `OpExpr` in `expr` has an `op` equal to `op` (or contained in the
iterable `ops`, e.g. [`STENCIL_OPS`](@ref)).
"""
contains_op(expr::ASTExpr, op::AbstractString)::Bool = contains_op(expr, (String(op),))

function contains_op(expr::ASTExpr, ops)::Bool
    opset = ops isa AbstractSet ? ops : Set(String(o) for o in ops)
    _contains_op(expr, opset)
end

function _contains_op(expr::ASTExpr, ops)::Bool
    if expr isa OpExpr
        expr.op in ops && return true
        for a in expr.args
            _contains_op(a, ops) && return true
        end
    end
    return false
end

"""
    has_stencil_op(expr::ASTExpr) -> Bool

`true` if `expr` contains any array/stencil op ([`STENCIL_OPS`](@ref)) — a
heuristic for "this is a (discretized) transport term".
"""
has_stencil_op(expr::ASTExpr)::Bool = contains_op(expr, STENCIL_OPS)

"""
    is_spatial_derivative(expr::ASTExpr) -> Bool

`true` if `expr` contains a continuous spatial differential operator: any of
[`SPATIAL_DERIVATIVE_OPS`](@ref) (`grad`/`div`/`laplacian`), or a `D` operator
differentiating with respect to a spatial variable (`wrt != "t"`). Use this to
split a system *before* discretization.
"""
function is_spatial_derivative(expr::ASTExpr)::Bool
    if expr isa OpExpr
        expr.op in SPATIAL_DERIVATIVE_OPS && return true
        if expr.op == "D" && expr.wrt !== nothing && expr.wrt != "t"
            return true
        end
        for a in expr.args
            is_spatial_derivative(a) && return true
        end
    end
    return false
end

"""
    stencil_vs_pointwise(term::ASTExpr) -> Int

Example binary rule: `1` if `term` contains a stencil op ([`has_stencil_op`](@ref)),
else `2`. Places (discretized) transport terms in part 1 and pointwise terms in
part 2 — the usual convention for treating transport implicitly under IMEX.
"""
stencil_vs_pointwise(term::ASTExpr)::Int = has_stencil_op(term) ? 1 : 2

"""
    spatial_vs_pointwise(term::ASTExpr) -> Int

Example binary rule for a **pre-discretization** system: `1` if `term` contains
a spatial derivative ([`is_spatial_derivative`](@ref)), else `2`.
"""
spatial_vs_pointwise(term::ASTExpr)::Int = is_spatial_derivative(term) ? 1 : 2

# ---------------------------------------------------------------------------
# Equation splitting
# ---------------------------------------------------------------------------

_is_time_derivative_eq(eq::Equation)::Bool =
    eq.lhs isa OpExpr && eq.lhs.op == "D" &&
    (eq.lhs.wrt === nothing || eq.lhs.wrt == "t")

"""
    split_equations(equations, rule; nparts=2) -> Vector{Vector{Equation}}

Partition a list of `Equation`s into `nparts` equation lists.

For each **time-derivative** equation `D(x, t) ~ rhs`, `rhs` is decomposed with
[`additive_terms`](@ref) and each term is assigned to a part by
`rule(term)::Int` (a value in `1:nparts`). Every part then receives one
equation `D(x, t) ~ sum_of_its_terms` (an empty part gets `~ 0`), so each part
still covers `x`.

Every **non**-time-derivative equation (observed definitions, `ic` equations) is
copied unchanged into *all* parts, because each part must evaluate the shared
observeds.
"""
function split_equations(equations::AbstractVector{Equation}, rule;
                         nparts::Integer = 2)::Vector{Vector{Equation}}
    nparts >= 1 || throw(ArgumentError("nparts must be >= 1, got $nparts"))
    parts = [Equation[] for _ in 1:nparts]
    for eq in equations
        if _is_time_derivative_eq(eq)
            buckets = [ASTExpr[] for _ in 1:nparts]
            for term in additive_terms(eq.rhs)
                p = rule(term)
                (p isa Integer && 1 <= p <= nparts) ||
                    throw(ArgumentError(
                        "rule returned $(repr(p)); expected an Integer in 1:$nparts"))
                push!(buckets[p], term)
            end
            for p in 1:nparts
                push!(parts[p], Equation(eq.lhs, sum_terms(buckets[p]);
                                         _comment = eq._comment))
            end
        else
            for p in 1:nparts
                push!(parts[p], eq)
            end
        end
    end
    return parts
end

# ---------------------------------------------------------------------------
# System splitting
# ---------------------------------------------------------------------------

"""
    split_system(flat::FlattenedSystem, rule; nparts=2) -> Vector{FlattenedSystem}
    split_system(model::Model, rule; nparts=2) -> Vector{Model}
    split_system(file::EsmFile, rule; nparts=2) -> Vector{FlattenedSystem}

Split a system into `nparts` sub-systems that share the *identical* variable set
(states, parameters, observeds, events, domain, …) and differ only in their
(partitioned) equations. See [`split_equations`](@ref) for how the right-hand
sides are partitioned by `rule`.

The return type mirrors the input so downstream `EarthSciAST.build_evaluator`
produces the same state naming as it would for the un-split system: a `Model`
splits into `Model`s, a `FlattenedSystem` into `FlattenedSystem`s. An `EsmFile`
is `flatten`ed first. See [`build_split_evaluator`](@ref) for the one-call path.
"""
function split_system(flat::FlattenedSystem, rule;
                      nparts::Integer = 2)::Vector{FlattenedSystem}
    eqparts = split_equations(flat.equations, rule; nparts = nparts)
    # Keyword copy-constructor keeps every other field (and the *same*
    # variable OrderedDicts) so each part's `build_evaluator` yields an
    # identical `u0` / `p` / `var_map`.
    return [FlattenedSystem(flat; equations = eqs) for eqs in eqparts]
end

function split_system(model::Model, rule; nparts::Integer = 2)::Vector{Model}
    eqparts = split_equations(model.equations, rule; nparts = nparts)
    # Every part carries the same variables/events/subsystems; only equations
    # differ. Same variable set ⇒ identical `build_evaluator` state ordering.
    return [Model(model.variables, eqs;
                  discrete_events = model.discrete_events,
                  continuous_events = model.continuous_events,
                  subsystems = model.subsystems,
                  tolerance = model.tolerance,
                  tests = model.tests,
                  initialization_equations = model.initialization_equations,
                  guesses = model.guesses,
                  system_kind = model.system_kind) for eqs in eqparts]
end

split_system(file::EsmFile, rule; nparts::Integer = 2) =
    split_system(flatten(file), rule; nparts = nparts)

# ---------------------------------------------------------------------------
# Split evaluator
# ---------------------------------------------------------------------------

"""
    SplitEvaluator

Result of [`build_split_evaluator`](@ref). Fields:

- `funcs::NTuple{N,Function}` — one in-place RHS closure `f!(du, u, p, t)` per
  part. They sum to the full system's RHS: `Σ_i funcs[i](du,u,p,t) == f_full`.
- `u0::Vector{Float64}` — shared initial state.
- `p` — shared parameter object (a `NamedTuple`, as returned by
  `build_evaluator`).
- `tspan::Tuple{Float64,Float64}` — default time span.
- `var_map::Dict{String,Int}` — state-name → index into `u0`/`du`.

For a 2-part split under an IMEX solver, `funcs[1]` is the implicit part and
`funcs[2]` the explicit part:
`SplitODEProblem(se.funcs[1], se.funcs[2], se.u0, se.tspan, se.p)` (or
[`split_ode_problem`](@ref)`(se)` with SciMLBase loaded).
"""
struct SplitEvaluator{N,P}
    funcs::NTuple{N,Function}
    u0::Vector{Float64}
    p::P
    tspan::Tuple{Float64,Float64}
    var_map::Dict{String,Int}
end

nparts(::SplitEvaluator{N}) where {N} = N

function Base.show(io::IO, se::SplitEvaluator{N}) where {N}
    print(io, "SplitEvaluator($N parts, $(length(se.u0)) states, tspan=$(se.tspan))")
end

"""
    build_split_evaluator(source, rule; nparts=2, kwargs...) -> SplitEvaluator

Split `source` (a `FlattenedSystem`, `Model`, or `EsmFile`) by `rule` and build
one `EarthSciAST.build_evaluator` RHS closure per part, all sharing a single
`u0`/`p`/`var_map`.

`kwargs` are forwarded to `build_evaluator` (e.g. `initial_conditions`,
`parameter_overrides`, `tspan`, `registered_functions`). The shared `u0`, `p`,
and `tspan` are taken from part 1; the per-part `var_map`s are asserted equal.
"""
function build_split_evaluator(source, rule; nparts::Integer = 2, kwargs...)
    parts = split_system(source, rule; nparts = nparts)
    fs = Function[]
    u0 = nothing
    p = nothing
    tspan = nothing
    var_map = nothing
    for (i, part) in enumerate(parts)
        f!, u0_i, p_i, tspan_i, vm_i = build_evaluator(part; kwargs...)
        push!(fs, f!)
        if i == 1
            u0, p, tspan, var_map = u0_i, p_i, tspan_i, vm_i
        else
            vm_i == var_map || error(
                "split part $i has a different state var_map than part 1; " *
                "this should not happen when parts share a variable set")
        end
    end
    return SplitEvaluator{Int(nparts),typeof(p)}(
        Tuple(fs), u0, p, tspan, var_map)
end

end # module EarthSciASTSplitter
