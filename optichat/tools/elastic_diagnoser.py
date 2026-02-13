import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ElasticInfeasibilityDiagnoser:
    def __init__(self, model):
        self.model = model
        self.original_constraints = {}
        self.elastic_slacks = {} # Map (constraint_name, index) -> slack_var
        self.elastic_constraints = {} # Map (constraint_name, index, type) -> constraint_data
        self.solver = SolverFactory('gurobi')

    def elasticize_model(self):
        """
        Robustly relaxes MIP -> LP and adds elastic slacks to inequalities AND equalities.
        """
        logger.info("--- 1. RELAXING DISCRETE VARIABLES (Integers/Binaries -> Continuous) ---")
        for var_container in self.model.component_objects(pyo.Var, active=True):
            is_discrete = False
            for v_data in var_container.values():
                if v_data.domain in [pyo.Binary, pyo.Integers]:
                    is_discrete = True
                    break
            
            if is_discrete:
                logger.info(f"Relaxing Variable: {var_container.name}")
                for v_data in var_container.values():
                    current_lb = v_data.lb
                    current_ub = v_data.ub
                    
                    v_data.domain = pyo.NonNegativeReals 
                    if current_lb is None and v_data.domain is pyo.Binary:
                        v_data.setlb(0)
                    if current_ub is None and v_data.domain is pyo.Binary:
                        v_data.setub(1)

        logger.info("\n--- 2. ELASTICIZING CONSTRAINTS (Handling ==, <=, >=) ---")
        
        if not hasattr(self.model, 'elastic_slacks'):
            self.model.elastic_slacks = pyo.Var(pyo.Any, within=pyo.NonNegativeReals)
        
        for constr in self.model.component_objects(pyo.Constraint, active=True):
            constr_name = constr.name
            
            if constr_name.startswith("elastic_"):
                continue
            
            constr.deactivate()
            
            new_constr_name = f"elastic_{constr_name}"
            if not hasattr(self.model, new_constr_name):
                self.model.add_component(new_constr_name, pyo.ConstraintList())
            new_constr_list = getattr(self.model, new_constr_name)
            
            for index in constr:
                c_obj = constr[index]
                
                if c_obj.equality:
                    s_pos_idx = (constr_name, index, '+')
                    s_neg_idx = (constr_name, index, '-')
                    
                    self.model.elastic_slacks[s_pos_idx]
                    self.model.elastic_slacks[s_neg_idx]
                    
                    s_pos = self.model.elastic_slacks[s_pos_idx]
                    s_neg = self.model.elastic_slacks[s_neg_idx]
                    
                    self.elastic_slacks[s_pos_idx] = s_pos
                    self.elastic_slacks[s_neg_idx] = s_neg
                    
                    new_con = new_constr_list.add(c_obj.body + s_pos - s_neg == c_obj.lower)
                    self.elastic_constraints[s_pos_idx] = new_con
                    self.elastic_constraints[s_neg_idx] = new_con

                else:
                    if c_obj.lower is not None:
                        s_idx = (constr_name, index, 'lb')
                        self.model.elastic_slacks[s_idx]
                        s_lb = self.model.elastic_slacks[s_idx]
                        self.elastic_slacks[s_idx] = s_lb
                        
                        # "Body >= Lower" becomes "Body + Slack >= Lower"
                        new_con = new_constr_list.add(c_obj.body + s_lb >= c_obj.lower)
                        self.elastic_constraints[s_idx] = new_con

                    # Handle Upper Bound (LHS <= UB  -->  LHS - s <= UB)
                    if c_obj.upper is not None:
                        s_idx = (constr_name, index, 'ub')
                        self.model.elastic_slacks[s_idx]
                        s_ub = self.model.elastic_slacks[s_idx]
                        self.elastic_slacks[s_idx] = s_ub
                        
                        # "Body <= Upper" becomes "Body - Slack <= Upper"
                        new_con = new_constr_list.add(c_obj.body - s_ub <= c_obj.upper)
                        self.elastic_constraints[s_idx] = new_con

    def solve_phase_1(self):
        """
        Solves the elastic model to minimize the sum of infeasibilities.
        """
        # Deactivate ALL existing objectives to prevent "multiple active objectives" error
        for obj in self.model.component_objects(pyo.Objective, active=True):
            obj.deactivate()
        
        if not self.elastic_slacks:
            logger.warning("No elastic slacks found! Phase 1 will be trivial (SINF=0).")
            
        slacks = [self.model.elastic_slacks[idx] for idx in self.elastic_slacks]
        
        if hasattr(self.model, 'elastic_obj'):
             self.model.del_component(self.model.elastic_obj)
             
        self.model.elastic_obj = pyo.Objective(expr=sum(slacks), sense=pyo.minimize)
        
        logger.info("Solving Phase 1 (Min SINF)...")
        self.solver.solve(self.model)
        
        safety_set = [idx for idx, s_var in self.elastic_slacks.items() if pyo.value(s_var) > 1e-5]
        return safety_set

    def run_heuristic_2(self, k=7):
        """
        Heuristic 2 with Top-K Sensitivity and NINF (Look-ahead) checking.
        """
        logger.info("\n--- STARTING HEURISTIC 2 (Iterative Reduction) ---")
        
        # Ensure we can access duals for sensitivity analysis
        if not hasattr(self.model, 'dual'):
            self.model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        if not hasattr(self.model, 'rc'):
            self.model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        # Initialize Elastic Weights
        self.model.slack_weights = pyo.Param(pyo.Any, initialize=1.0, mutable=True)
        for idx in self.elastic_slacks:
            self.model.slack_weights[idx] = 1.0
        
        # Set Objective: Minimize sum of weighted slacks
        if hasattr(self.model, 'elastic_obj'):
            self.model.del_component(self.model.elastic_obj)
        self.model.elastic_obj = pyo.Objective(
            expr=sum(self.model.slack_weights[idx] * self.elastic_slacks[idx] for idx in self.elastic_slacks), 
            sense=pyo.minimize
        )
        
        # Initial Solve to get Baseline
        self.solver.solve(self.model)
        
        safety_set = [idx for idx, s_var in self.elastic_slacks.items() if pyo.value(s_var) > 1e-5]
        safety_size = len(safety_set)
        
        cover_set = []
        current_sinf = pyo.value(self.model.elastic_obj)
        
        logger.info(f"Baseline SafetySize: {safety_size}")
        logger.info(f"Baseline SINF: {current_sinf}")

        # --- MAIN LOOP ---
        while current_sinf > 1e-5:
            logger.info(f"\n--- Iteration Start (Current Cover Size: {len(cover_set)}) ---")
            
            # 1. Identify Valid Candidates (violated and not already in cover)
            raw_candidates = [idx for idx, s_var in self.elastic_slacks.items() 
                            if pyo.value(s_var) > 1e-5 and idx not in cover_set]

            # 2. Top-K Sensitivity Selection
            # We rank candidates by 'Sensitivity' (abs(Dual) * Slack_Value)
            candidate_scores = []
            for idx in raw_candidates:
                slack_val = pyo.value(self.elastic_slacks[idx])
                
                dual_val = 1.0 
                if hasattr(self, 'elastic_constraints') and idx in self.elastic_constraints:
                    # Get dual of the constraint (or the bound on the slack)
                    constr = self.elastic_constraints[idx]
                    dual_val = abs(self.model.dual.get(constr, 1.0))

                score = slack_val * dual_val
                candidate_scores.append((idx, score))
            
            # Sort by score descending and take top K
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            top_k_candidates = [x[0] for x in candidate_scores[:k]]
            
            logger.info(f"Selected Top-{len(top_k_candidates)} candidates based on sensitivity.")

            winner = None
            winner_sinf = float('inf')
            next_winner = None # The 'NINF=1' look-ahead constraint

            # 3. Trial Deletion Loop
            for cand in top_k_candidates:
                # 'Remove' constraint by setting weight to 0 (free to violate)
                self.model.slack_weights[cand] = 0.0
                
                self.solver.solve(self.model)
                trial_sinf = pyo.value(self.model.elastic_obj)
                
                # Check for Magic Bullet (SINF=0)
                if trial_sinf <= 1e-5:
                    logger.info(f"  -> Magic Bullet found: {cand} makes SINF=0!")
                    winner = cand
                    winner_sinf = 0.0
                    next_winner = None
                    self.model.slack_weights[cand] = 1.0 # Reset before break
                    break
                
                # Check if this is a better winner
                if trial_sinf < current_sinf:
                    if trial_sinf < winner_sinf:
                        winner = cand
                        winner_sinf = trial_sinf
                                                    
                        # 1. Identify all currently violated constraints (slacks > 0)
                        remaining_violations = [idx for idx, s_var in self.elastic_slacks.items() 
                                                if pyo.value(s_var) > 1e-5]
                        
                        # 2. Filter: Only count violations that are NOT currently removed (weight == 1.0)
                        active_violations = [
                            idx for idx in remaining_violations 
                            if pyo.value(self.model.slack_weights[idx]) > 1e-5
                        ]
                        
                        ninf = len(active_violations)
                        
                        if ninf == 1:
                            # We found a 'Next Winner'!
                            next_winner = active_violations[0]
                            # logger.info(f"    -> Look-ahead: Removal leaves only {next_winner} (NINF=1)")
                        else:
                            next_winner = None

                # Reset weight for next trial
                self.model.slack_weights[cand] = 1.0 

            # 4. Process Results (Commitment Phase)
            if winner:
                logger.info(f"Winner: {winner} (New SINF: {winner_sinf})")
                
                # Add Winner
                cover_set.append(winner)
                self.model.slack_weights[winner] = 0.0 # Permanently remove
                current_sinf = winner_sinf
                
                # --- CHECK NEXT WINNER (Double Play) ---
                if next_winner:
                    logger.info(f"NextWinner identified: {next_winner}. Adding both and EXITING.")
                    cover_set.append(next_winner)
                    # We can stop immediately because we know these two fix the problem
                    return cover_set

                # Check Bailout
                if len(cover_set) >= (safety_size - 1):
                    logger.warning("Bailout Triggered: Heuristic not efficient.")
                    return safety_set
                
            else:
                logger.error("No winner found to reduce SINF. Solver stuck.")
                break
                
        logger.info("\n--- FINAL RESULTS ---")
        return cover_set

    def verify_feasibility(self, constraints_to_remove):
        """
        Verifies if removing the constraints in 'constraints_to_remove'
        makes the original model feasible.
        """
        logger.info(f"\n--- VERIFYING FEASIBILITY ---")
        logger.info(f"Removing {len(constraints_to_remove)} constraints and enforcing the rest...")

        if hasattr(self.model, 'elastic_obj'):
            self.model.elastic_obj.deactivate()
            
        original_obj_activated = False
        for obj in self.model.component_objects(pyo.Objective, active=False):
            if obj.name != 'elastic_obj':
                obj.activate()
                original_obj_activated = True
                logger.info(f"Original Objective '{obj.name}' Activated.")
        
        if not original_obj_activated:
            logger.warning("No original objective found/activated! Model might solve with 0 cost.")
            # Create dummy 0-cost objective if strictly needed by solver/writer, though Gurobi usually handles 0 cost.
            # But Pyomo's LP writer crashes if NO objective exists.
            if not any(self.model.component_objects(pyo.Objective, active=True)):
                 self.model.dummy_zero_obj = pyo.Objective(expr=0.0)

        for idx, s_var in self.elastic_slacks.items():
            
            if idx in constraints_to_remove:
                s_var.unfix()
                s_var.setlb(0) 
                
            else:
                s_var.fix(0.0)
        logger.info("Solving with Original Objective...")
        result = self.solver.solve(self.model, tee=True)
        
        status = result.solver.termination_condition
        logger.info(f"Solver Status: {status}")
        
        if status == pyo.TerminationCondition.optimal:
            # Find the active objective to get the value
            active_objs = [o for o in self.model.component_objects(pyo.Objective, active=True)]
            if active_objs:
                obj_val = pyo.value(active_objs[0])
            else:
                obj_val = 0.0

            # Extract slack values for the relaxed constraints
            slack_values = {}
            for idx in constraints_to_remove:
                if idx in self.elastic_slacks:
                    slack_val = pyo.value(self.elastic_slacks[idx])
                    slack_values[idx] = slack_val
                    if slack_val > 1e-6:  # Only log significant violations
                        logger.info(f"  Slack for {idx}: {slack_val:.4f}")

            logger.info(f" -> RESULT: FEASIBLE!")
            logger.info(f" -> Optimal Original Cost (with relaxations): {obj_val}")
            return True, obj_val, slack_values
        else:
            logger.info(" -> RESULT: STILL INFEASIBLE (or Unbounded).")
            return False, None, {}
