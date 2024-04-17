#Look for #IMPLEMENT tags in this file. These tags indicate what has
#to be implemented to complete problem solution.

'''This file will contain different constraint propagators to be used within 
   bt_search.

   propagator == a function with the following template
      propagator(csp, newly_instantiated_variable=None)
           ==> returns (True/False, [(Variable, Value), (Variable, Value) ...]

      csp is a CSP object---the propagator can use this to get access
      to the variables and constraints of the problem. The assigned variables
      can be accessed via methods, the values assigned can also be accessed.

      newly_instaniated_variable is an optional argument.
      if newly_instantiated_variable is not None:
          then newly_instantiated_variable is the most
           recently assigned variable of the search.
      else:
          progator is called before any assignments are made
          in which case it must decide what processing to do
           prior to any variables being assigned. SEE BELOW

       The propagator returns True/False and a list of (Variable, Value) pairs.
       Return is False if a deadend has been detected by the propagator.
       in this case bt_search will backtrack
       return is true if we can continue.

      The list of variable values pairs are all of the values
      the propagator pruned (using the variable's prune_value method). 
      bt_search NEEDS to know this in order to correctly restore these 
      values when it undoes a variable assignment.

      NOTE propagator SHOULD NOT prune a value that has already been 
      pruned! Nor should it prune a value twice

      PROPAGATOR called with newly_instantiated_variable = None
      PROCESSING REQUIRED:
        for plain backtracking (where we only check fully instantiated 
        constraints) 
        we do nothing...return true, []

        for forward checking (where we only check constraints with one
        remaining variable)
        we look for unary constraints of the csp (constraints whose scope 
        contains only one variable) and we forward_check these constraints.


      PROPAGATOR called with newly_instantiated_variable = a variable V
      PROCESSING REQUIRED:
         for plain backtracking we check all constraints with V (see csp method
         get_cons_with_var) that are fully assigned.

         for forward checking we forward check all constraints with V
         that have one unassigned variable left

   '''

from cspbase import *
def prop_BT(
    csp: CSP,
    newVar: Variable = None,
):
    '''Do plain backtracking propagation. That is, do no 
    propagation at all. Just check fully instantiated constraints'''

    if not newVar:
        return True, []
    for c in csp.get_cons_with_var(newVar):
        if c.get_n_unasgn() == 0:
            vals = []
            vars = c.get_scope()
            for var in vars:
                vals.append(var.get_assigned_value())
            if not c.check(vals):
                return False, []
    return True, []

def prop_FC(
    csp: CSP,
    newVar: Variable = None,
):
    '''Do forward checking. That is check constraints with 
       only one uninstantiated variable. Remember to keep 
       track of all pruned variable,value pairs and return '''
    """ CSC384 BEGIN """
    prune_l: list[tuple[Variable, int]] = []
    cons_l: list[Constraint] = csp.get_all_cons() if newVar is None else csp.get_cons_with_var(newVar)

    for cons in cons_l:
        if cons.get_n_unasgn() != 1:
            continue

        cons_var_list = [var.get_assigned_value() for var in cons.get_scope()] # `cons_var_list` has exactly 1 `None`

        unasgn_var: Variable = cons.get_unasgn_vars()[0]
        unasgn_var_domain = unasgn_var.cur_domain()
        unasgn_var_index = cons_var_list.index(None)

        for val in unasgn_var_domain:
            cons_var_list[unasgn_var_index] = val

            if cons.check(cons_var_list) == False:
                unasgn_var.prune_value(val)
                prune_l.append((unasgn_var, val))

            cons_var_list[unasgn_var_index] = None

        if unasgn_var.cur_domain_size() == 0: # DWO
            return False, prune_l

    return True, prune_l
    """ CSC384 END """
    
def prop_FI(
    csp: CSP,
    newVar: Variable = None,
):
    '''Do full inference. If newVar is None we initialize the queue
       with all variables.'''
    """ CSC384 BEGIN """
    prune_l: list[tuple[Variable, int]] = []
    cons_l: list[Constraint] = csp.get_all_cons() if newVar is None else csp.get_cons_with_var(newVar)

    full_inf_q: list[Constraint] = cons_l
    
    def add_cons_including_violated_var_to_full_inf_q_if_not_included(
        violated_var: Variable,
    ):
        cons_including_violated_var_l: list[Constraint] = csp.get_cons_with_var(violated_var)
        for cons in cons_including_violated_var_l:
            if cons not in full_inf_q:
                full_inf_q.append(cons)

    while len(full_inf_q) > 0:
        cons = full_inf_q.pop(0)
        unasgn_var_l: list[Variable] = cons.get_unasgn_vars()

        for unasgn_var in unasgn_var_l:
            for unasgn_var_val_to_asgn in unasgn_var.cur_domain():
                if cons.has_support(unasgn_var, unasgn_var_val_to_asgn) == False:
                    unasgn_var.prune_value(unasgn_var_val_to_asgn)
                    prune_l.append((unasgn_var, unasgn_var_val_to_asgn))

                    if unasgn_var.cur_domain_size() == 0: # DWO
                        return False, prune_l

                    add_cons_including_violated_var_to_full_inf_q_if_not_included(unasgn_var)

    return True, prune_l

    """ CSC384 END """
