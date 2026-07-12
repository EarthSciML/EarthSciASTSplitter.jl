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

# The default rule `transport_vs_pointwise` puts spatially-coupled (transport)
# terms in part 1 (implicit) and pointwise (reaction) terms in part 2.
se = build_split_evaluator(flat, transport_vs_pointwise)   # nparts = 2 by default
# se.funcs :: NTuple of in-place f!(du,u,p,t); se.u0, se.p, se.tspan, se.var_map

# IMEX (DiffEqDocs): du/dt = f1(u,p,t) + f2(u,p,t), f1 implicit.
using OrdinaryDiffEqBDF          # or the SciMLBase extension helper below
prob = SplitODEProblem(se.funcs[1], se.funcs[2], se.u0, se.tspan, se.p)

# With SciMLBase loaded, `split_ode_problem(se)` builds the same problem:
prob = split_ode_problem(se)
```

See [`split_system`](@ref), [`split_equations`](@ref),
[`build_split_evaluator`](@ref), the default rule [`transport_vs_pointwise`](@ref)
(and its predicate [`spatially_coupled`](@ref)), and the generic rule helpers
[`references`](@ref) and [`contains_op`](@ref).
"""
module EarthSciASTSplitter

using EarthSciAST
using EarthSciAST: ASTExpr, IntExpr, VarExpr, OpExpr, Equation, Model,
                   FlattenedSystem, EsmFile, flatten, build_evaluator

export additive_terms, sum_terms,
       split_equations, split_system, build_split_evaluator, SplitEvaluator,
       split_ode_problem, operator_splitting_problem,
       TermContext, references, contains_op,
       spatially_coupled, transport_vs_pointwise

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
iterable `ops`). A generic building block for custom rules — note it says
nothing about the *semantics* of an op (op names are not fixed by EarthSciAST).
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

# ---------------------------------------------------------------------------
# Spatial-locality analysis — the real "transport vs pointwise" criterion
#
# EarthSciAST v0.8+ has NO distinguished spatial operators: `grad`/`div`/
# `laplacian` are meaningless tokens, and discretization is done by expression
# templates that lower a spatial derivative into whatever stencil AST they like
# (`makearray`/`aggregate`/`arrayop`/`index`-gathers, …). So a transport term
# cannot be recognized by any fixed op name.
#
# What *is* invariant is the DATA DEPENDENCY. In the (discretized) equation for
# a state cell, a **pointwise** term reads state only at that same cell; a
# **transport** (spatially coupled) term reads state at some OTHER cell — a
# neighbor, a boundary cell, or a range it reduces over. That is exactly what
# makes a term amenable to operator splitting, and it is representation-
# independent: it holds whether the stencil is a `makearray`, a scalarized
# `index(u, i-1)`, or an `aggregate` reduction.
#
# `spatially_coupled(term, ctx)` decides this by comparing, for every state
# read in `term`, the cell it accesses against the equation's output cell.
# ---------------------------------------------------------------------------

"""
    TermContext(lhs::ASTExpr, states::Set{String})

The per-equation context a context-aware rule receives as its second argument:
the equation's left-hand side `lhs` (e.g. `D(u)` or `D(index(u, i))`) and the
set of state-variable base names `states`. Built automatically by
[`split_equations`](@ref); most user rules ignore it (they take just the term).
"""
struct TermContext
    lhs::ASTExpr
    states::Set{String}
end

# base variable name, stripping any `[...]` cell suffix (defensive; the AST
# normally accesses cells via `index(u, …)`, not bracketed names).
_base_name(name::AbstractString) = String(first(split(name, '['; limit=2)))

# The state variable and output-cell index of a `D(...)` left-hand side.
# Returns (base_name, cell) where `cell` is `nothing` for a whole-field / 0-D
# state (`D(u)`) or a `Vector{ASTExpr}` index tuple for a specific cell
# (`D(index(u, i))`). Returns (nothing, nothing) if `lhs` is not `D(state…)`.
function _lhs_state_cell(lhs::ASTExpr)
    (lhs isa OpExpr && lhs.op == "D" && !isempty(lhs.args)) || return (nothing, nothing)
    inner = lhs.args[1]
    inner isa VarExpr && return (_base_name(inner.name), nothing)
    if inner isa OpExpr && inner.op == "index" && !isempty(inner.args) &&
       inner.args[1] isa VarExpr
        return (_base_name(inner.args[1].name), ASTExpr[inner.args[2:end]...])
    end
    return (nothing, nothing)
end

# The base state names declared by a set of equations (one per `D(state,t)`).
function _state_base_names(equations)::Set{String}
    s = Set{String}()
    for eq in equations
        _is_time_derivative_eq(eq) || continue
        name, _ = _lhs_state_cell(eq.lhs)
        name === nothing || push!(s, name)
    end
    return s
end

# Structural equality of two index expressions (`OpExpr`'s own `==` is identity).
function _ast_equal(a::ASTExpr, b::ASTExpr)::Bool
    a isa VarExpr && b isa VarExpr && return a.name == b.name
    a isa IntExpr && b isa IntExpr && return a.value == b.value
    a isa NumExpr && b isa NumExpr && return a.value == b.value
    a isa IntExpr && b isa NumExpr && return a.value == b.value
    a isa NumExpr && b isa IntExpr && return a.value == b.value
    if a isa OpExpr && b isa OpExpr
        a.op == b.op || return false
        length(a.args) == length(b.args) || return false
        return all(((x, y),) -> _ast_equal(x, y), zip(a.args, b.args))
    end
    return false
end

_cells_equal(a::Vector{ASTExpr}, b::Vector{ASTExpr}) =
    length(a) == length(b) && all(((x, y),) -> _ast_equal(x, y), zip(a, b))

# An `output_idx` field (`Vector{Any}` of index-symbol `String`s / literal
# `Int`s) as a cell tuple of `ASTExpr`.
_output_idx_cell(oi) = ASTExpr[e isa Integer ? IntExpr(Int64(e)) : VarExpr(String(e)) for e in oi]

"""
    spatially_coupled(term::ASTExpr, ctx::TermContext) -> Bool

`true` if `term` reads a state variable at a cell **other than** the equation's
own output cell — i.e. it couples distinct grid cells (a transport /
finite-difference / nonlocal term). `false` for a purely **pointwise** term
that reads state only at the output cell (a reaction / source / local term).
This is the representation-independent transport-vs-pointwise criterion; see
[`transport_vs_pointwise`](@ref) for the rule built on it.

State access is assumed to be by bare reference (whole-field / 0-D, always
local) or `index(state, …)`; the output cell is the index on the `D(…)`
left-hand side, or — inside an `aggregate`/`arrayop` — that node's `output_idx`.
"""
spatially_coupled(term::ASTExpr, ctx::TermContext)::Bool =
    _reads_nonlocal(term, _lhs_cell(ctx.lhs), ctx.states)

_lhs_cell(lhs::ASTExpr) = _lhs_state_cell(lhs)[2]  # nothing (field/0-D) or Vector

# Does `node` read any state at a cell ≠ the current output cell `oc`
# (`nothing` = the field/elementwise output; `Vector` = a concrete/symbolic
# index tuple, e.g. from the LHS or an enclosing aggregate)?
function _reads_nonlocal(node::ASTExpr, oc, states::Set{String})::Bool
    if node isa VarExpr
        # a bare state reference is elementwise ⇒ always the output cell (local)
        return false
    elseif node isa IntExpr || node isa NumExpr
        return false
    elseif node isa OpExpr
        if node.op == "index" && !isempty(node.args) && node.args[1] isa VarExpr &&
           _base_name(node.args[1].name) in states
            readcell = ASTExpr[node.args[2:end]...]
            # local iff it reads the current output cell; a fixed-cell read in a
            # field equation (oc === nothing) is nonlocal by definition.
            if oc === nothing || !_cells_equal(readcell, oc)
                return true
            end
            # still descend into the index expressions (they may themselves
            # gather state, e.g. an indirect/gather index).
            return any(a -> _reads_nonlocal(a, oc, states), node.args[2:end])
        end
        # entering an aggregate/arrayop rebinds the output cell to its output_idx
        child_oc = oc
        if (node.op == "aggregate" || node.op == "arrayop") && node.output_idx !== nothing
            child_oc = _output_idx_cell(node.output_idx)
        end
        for c in _child_exprs(node)
            _reads_nonlocal(c, child_oc, states) && return true
        end
        return false
    end
    return false
end

# Every `ASTExpr`-valued child of an `OpExpr` (args plus the structural fields
# that carry sub-expressions: stencil bodies, bounds, makearray values, …).
function _child_exprs(node::OpExpr)::Vector{ASTExpr}
    cs = ASTExpr[]
    append!(cs, node.args)
    node.expr_body === nothing || push!(cs, node.expr_body)
    node.lower === nothing || push!(cs, node.lower)
    node.upper === nothing || push!(cs, node.upper)
    node.key === nothing || push!(cs, node.key)
    node.filter === nothing || push!(cs, node.filter)
    node.values === nothing || append!(cs, node.values)
    if node.table_axes !== nothing
        append!(cs, values(node.table_axes))
    end
    return cs
end

"""
    transport_vs_pointwise(term::ASTExpr, ctx::TermContext) -> Int

The default two-way splitting rule: `1` (transport) if `term` is
[`spatially_coupled`](@ref) — it reads state at another cell — else `2`
(pointwise / reaction). Part 1 is the transport operator, part 2 the pointwise
operator, matching the usual IMEX convention of treating transport implicitly.

This is a *context-aware* rule: pass it straight to [`build_split_evaluator`](@ref)
/ [`split_system`](@ref) / [`split_equations`](@ref), which supply the
`TermContext`.
"""
transport_vs_pointwise(term::ASTExpr, ctx::TermContext)::Int =
    spatially_coupled(term, ctx) ? 1 : 2

# ---------------------------------------------------------------------------
# Equation splitting
# ---------------------------------------------------------------------------

_is_time_derivative_eq(eq::Equation)::Bool =
    eq.lhs isa OpExpr && eq.lhs.op == "D" &&
    (eq.lhs.wrt === nothing || eq.lhs.wrt == "t")

# Apply a rule that is either `rule(term)` or context-aware `rule(term, ctx)`.
function _apply_rule(rule, term::ASTExpr, ctx::TermContext)
    if applicable(rule, term, ctx)
        return rule(term, ctx)
    elseif applicable(rule, term)
        return rule(term)
    end
    throw(ArgumentError(
        "rule is not callable as rule(term) or rule(term, ctx::TermContext)"))
end

"""
    split_equations(equations, rule; nparts=2) -> Vector{Vector{Equation}}

Partition a list of `Equation`s into `nparts` equation lists.

For each **time-derivative** equation `D(x, t) ~ rhs`, `rhs` is decomposed with
[`additive_terms`](@ref) and each term is assigned to a part in `1:nparts` by
`rule`. A rule is either `rule(term)::Int` or the context-aware form
`rule(term, ctx::TermContext)::Int` (the context carries the equation's LHS and
the state-variable names — needed by locality-based rules like
[`transport_vs_pointwise`](@ref)); `split_equations` dispatches on whichever the
rule accepts. Every part then receives one equation
`D(x, t) ~ sum_of_its_terms` (an empty part gets `~ 0`), so each part still
covers `x`.

Every **non**-time-derivative equation (observed definitions, `ic` equations) is
copied unchanged into *all* parts, because each part must evaluate the shared
observeds.
"""
function split_equations(equations::AbstractVector{Equation}, rule;
                         nparts::Integer = 2)::Vector{Vector{Equation}}
    nparts >= 1 || throw(ArgumentError("nparts must be >= 1, got $nparts"))
    states = _state_base_names(equations)
    parts = [Equation[] for _ in 1:nparts]
    for eq in equations
        if _is_time_derivative_eq(eq)
            ctx = TermContext(eq.lhs, states)
            buckets = [ASTExpr[] for _ in 1:nparts]
            for term in additive_terms(eq.rhs)
                p = _apply_rule(rule, term, ctx)
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
