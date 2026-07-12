# EarthSciASTSplitter.jl

Split an [EarthSciAST](https://github.com/EarthSciML/EarthSciAST) system of
equations into several sub-systems by **user-defined rules**, for operator-
splitting / IMEX time integration with
[`SplitODEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/solvers/split_ode_solve/)
or
[OrdinaryDiffEqOperatorSplitting.jl](https://github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl).

The motivating example: put the **transport (PDE) terms** in one system and the
**pointwise (reaction) terms** in another, so a solver can treat the stiff
transport implicitly and the reaction explicitly, or the reverse (or drive each with its own
sub-integrator).

## What it does

It works on the **post-discretization** system — after spatial operators have
been lowered to grid/stencil array expressions (in EarthSciAST v0.8+ this is
done by `expression_template_imports`; see
[EarthSciDiscretizations](https://github.com/EarthSciML/EarthSciDiscretizations)).
At that point every equation is an ODE `D(x, t) ~ rhs` over scalar/gridded
state, and the whole system is consumed by `EarthSciAST.build_evaluator`.

For each derivative equation, the right-hand side is decomposed into its
**additive terms**, and a rule assigns each term to a part:

```
D(c, t) ~ (-v)*makearray(…stencil…)  +  k1*c*c2  -  k2*c
          └────── transport ───────┘   └──── reaction ────┘
                     part 1                    part 2
```

Every part keeps the **identical** variable set — only the split right-hand
sides differ (an absent contribution becomes `0`). Because `build_evaluator`
derives `u0`, `p`, and the state-index `var_map` from the *variables* (not the
equations), all parts act on one shared state vector `u` and satisfy
`f_full = Σ f_part` exactly.

Non-derivative equations (observed definitions, `ic` equations) are copied
unchanged into every part.

## Design note: this is entirely on EarthSciAST's public API

- The AST (`ASTExpr`/`NumExpr`/`IntExpr`/`VarExpr`/`OpExpr`), `Equation`,
  `Model`, `FlattenedSystem`, `flatten`, and `build_evaluator` are all public,
  exported EarthSciAST symbols. A *rule* is just a predicate over the public AST.
- The split is additive over each equation's RHS, so it is representation-
  agnostic: it works whether the discretized RHS is a field-level `makearray`
  expression or a per-cell scalarized `arrayop`.

## Usage

```julia
using EarthSciAST, EarthSciASTSplitter

flat = flatten(load("model.esm"))        # a post-discretization FlattenedSystem

# A rule maps one additive term to a part index in 1:nparts. The default rule
# `transport_vs_pointwise` puts spatially-coupled (transport) terms in part 1
# and pointwise (reaction) terms in part 2.
se = build_split_evaluator(flat, transport_vs_pointwise)

# se.funcs :: NTuple of in-place f!(du,u,p,t); se.u0, se.p, se.tspan, se.var_map

# --- IMEX (DiffEqDocs): du/dt = f1(u,p,t) + f2(u,p,t), f1 implicit ---
using SciMLBase, OrdinaryDiffEqSDIRK
prob = SplitODEProblem(se.funcs[1], se.funcs[2], se.u0, se.tspan, se.p)
# or, with SciMLBase loaded, the extension helper:
prob = split_ode_problem(se)
sol  = solve(prob, KenCarp4())

# --- OrdinaryDiffEqOperatorSplitting.jl (Lie/Strang) ---
using OrdinaryDiffEqOperatorSplitting, OrdinaryDiffEqLowOrderRK
osprob = operator_splitting_problem(se)              # extension helper
integ  = init(osprob, LieTrotterGodunov((Euler(), Euler())); dt=0.005)
for _ in 1:200; step!(integ); end                    # or TimeChoiceIterator(integ, tstops)
u_final = integ.u
```

### The default rule: spatial locality (not a stencil heuristic)

`grad`/`div`/`laplacian` are **not** meaningful operators in EarthSciAST v0.8+
— they are ordinary tokens that expression templates rewrite into arbitrary
stencil ASTs — so a transport term cannot be recognized by any fixed op name.
What is invariant is the **data dependency**:

> a term is **pointwise** if, in the (discretized) equation for a state cell, it
> reads state only at that same cell; it is **transport** (spatially coupled) if
> it reads state at another cell — a neighbor, a boundary cell, or a range it
> reduces over.

`transport_vs_pointwise(term, ctx)` decides this by comparing, for every state
read in the term, the cell it accesses against the equation's output cell. It is
representation-independent: it works whether the stencil is a `makearray`, a
scalarized `index(u, i-1)`, or an `aggregate` reduction, and it correctly keeps
a multi-species same-cell reaction (`k·c1[i]·c2[i]`) on the pointwise side.

### Writing your own rule

A rule is any `term::ASTExpr -> Int`, or a context-aware
`(term, ctx::TermContext) -> Int` where `ctx` carries the equation's LHS and the
state-variable names (`split_equations` supplies it and dispatches on whichever
form the rule accepts). Helpers:

| Helper | Meaning |
|---|---|
| `spatially_coupled(term, ctx)` | the real transport test — term reads state at a cell ≠ the output cell |
| `transport_vs_pointwise(term, ctx)` | default rule: `1` if `spatially_coupled`, else `2` |
| `references(term, names)` | term mentions any variable in `names` |
| `contains_op(term, ops)` | term contains any `OpExpr` whose `op ∈ ops` (generic; op names carry no fixed meaning) |

```julia
# a custom rule: implicit part = anything touching species c1 or c2
rule = term -> references(term, ("c1", "c2")) ? 1 : 2
se = build_split_evaluator(flat, rule; nparts = 2)
```

## API

- `build_split_evaluator(source, rule; nparts=2, kwargs...) -> SplitEvaluator`
  — one call: split + build one RHS closure per part sharing `u0`/`p`/`var_map`.
  `kwargs` are forwarded to `build_evaluator`.
- `split_system(flat_or_model_or_file, rule; nparts=2) -> Vector{FlattenedSystem}`
  — the split systems, each sharing the variable set.
- `split_equations(equations, rule; nparts=2) -> Vector{Vector{Equation}}`
  — the equation-level primitive.
- `additive_terms(expr)` / `sum_terms(terms)` — the additive decomposition.
- `split_ode_problem(se; kwargs...)` — build a `SplitODEProblem` (needs SciMLBase).

## Extensions

Loaded automatically when the trigger packages are present:

- **`SciMLBase`** → `split_ode_problem(se)` builds a `SplitODEProblem` (IMEX).
- **`OrdinaryDiffEqOperatorSplitting`** → `operator_splitting_problem(se)` builds
  an `OperatorSplittingProblem` (Lie–Trotter / Strang).

## Status

Early scaffold, but verified end-to-end:

- Splitting core reconstructs `EarthSciAST.build_evaluator` exactly
  (`Σ f_part == f_full`), on hand-built models and on self-contained discretized
  fixtures in `test/esm/` (a 1-D periodic advection and an advection **+
  linear-decay reaction** system, both with a real `makearray` stencil).
- The reaction+transport split isolates the pointwise reaction term (`-k·u`)
  from the stencil transport term.
- Both extensions are tested: the IMEX `SplitODEProblem`'s `f₁+f₂` reproduces the
  full RHS, and the `OperatorSplittingProblem` integrates (a `step!` loop matches
  a manual Lie–Trotter loop exactly; a constant initial field decays to the exact
  analytic reaction rate).

The `test/esm/` fixtures are self-contained (no `expression_template_imports`,
no external repo) — they are the canonical *post-discretization* AST, so tests
never depend on [EarthSciDiscretizations](https://github.com/EarthSciML/EarthSciDiscretizations)
being checked out.
