module EarthSciASTSplitterSciMLBaseExt

using EarthSciASTSplitter
using EarthSciASTSplitter: SplitEvaluator, nparts
import SciMLBase

"""
    split_ode_problem(se::SplitEvaluator; kwargs...) -> SciMLBase.SplitODEProblem

Build an IMEX `SplitODEProblem` from a 2-part `SplitEvaluator`. The first part
(`se.funcs[1]`) is the implicit operator, the second (`se.funcs[2]`) the
explicit one. `kwargs` are forwarded to the `SplitODEProblem` constructor.
"""
function EarthSciASTSplitter.split_ode_problem(se::SplitEvaluator; kwargs...)
    nparts(se) == 2 || throw(ArgumentError(
        "split_ode_problem requires a 2-part SplitEvaluator (got $(nparts(se))); " *
        "SplitODEProblem is for du/dt = f1 + f2."))
    return SciMLBase.SplitODEProblem(se.funcs[1], se.funcs[2], se.u0, se.tspan,
                                     se.p; kwargs...)
end

end # module
