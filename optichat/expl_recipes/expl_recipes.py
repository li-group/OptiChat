EXPL_RECIPES = """
**FACTUAL EXPLANATION: TRADE-OFFS**
- Users may be curious about the rationale behind certain decisions made by the model.
This is caused by the trade-offs that the model balances competing objectives while satisfying all constraints.
- To locate the root cause, focus on what decision variables are penalized in the objective function, and by how much, 
and how these decision variables interact with other decision variables through constraints.

**COUNTERFACTUAL EXPLANATION: NEW SCENARIO**
- Users may be suspicious of certain decisions or interested in exploring alternatives.
This represents a new scenario that requires the original model to be modified and re-solved for comparison.
- Since resolving new models is costly, this explanation strategy should only be used 
when the user explicitly requests it in the query and indicates a clear modification object and extent.
    e.g. "What will happen if the penalty of overage is doubled?"
    e.g. "What if we need the generator A and B to produce at least X MW in total every hour?"
    e.g. "Why can't we force the station to have Y routes instead of Z routes every day?"
"""