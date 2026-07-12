module EarthSciASTSplitterOperatorSplittingExt

using EarthSciASTSplitter
using EarthSciASTSplitter: SplitEvaluator, nparts
import OrdinaryDiffEqOperatorSplitting as OS
import SciMLBase

"""
    operator_splitting_problem(se::SplitEvaluator; dofs=nothing) -> OS.OperatorSplittingProblem

Build an `OperatorSplittingProblem` with one sub-operator per split part. Each
`se.funcs[i]` is wrapped in an `ODEFunction`; the shared parameter object
`se.p` is captured in the wrapper, so the operator-splitting integrator's own
parameter slot is ignored. By default each operator acts on all DOFs.
"""
function EarthSciASTSplitter.operator_splitting_problem(se::SplitEvaluator; dofs = nothing)
    N = nparts(se)
    ndof = length(se.u0)
    d = dofs === nothing ? ntuple(_ -> collect(1:ndof), N) : Tuple(dofs)
    length(d) == N || throw(ArgumentError(
        "operator_splitting_problem needs $N dof groups (one per part), got $(length(d))"))
    p = se.p
    fns = ntuple(i -> SciMLBase.ODEFunction((du, u, _p, t) -> se.funcs[i](du, u, p, t)), N)
    gsf = OS.GenericSplitFunction(fns, d)
    return OS.OperatorSplittingProblem(gsf, copy(se.u0), se.tspan)
end

end # module
