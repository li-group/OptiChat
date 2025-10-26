from pyomo.opt import SolverStatus, TerminationCondition

TerminationConditionMap = {TerminationCondition.infeasible, 
                           TerminationCondition.infeasibleOrUnbounded}