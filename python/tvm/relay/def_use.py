from . import expr
from .expr_functor import ExprVisitor
from .analysis import free_vars

import attr
from typing import List, Dict

@attr.s(auto_attribs=True)
class DefUse:
    defn: expr.Var
    uses: List[expr.Expr]

class DefUseAnalysis(ExprVisitor):
    def __init__(self):
        super().__init__()
        self.results: Dict[expr.Var, DefUse] = {}

    def visit_function(self, func):
        for param in func.params:
            self.results[param] = DefUse(param, [])
        super().visit_function(func)

    def visit_let(self, let):
        while isinstance(let, expr.Let):
            du = DefUse(let.var, [])
            self.results[let.var] = du
            # Find all variables used in RHS.
            self.visit(let.value)
            used_vars = free_vars(let.value)
            for uvar in used_vars:
                self.results[uvar].uses.append(let.value)
            let = let.body

        # Find all variables used in body.
        used_vars = free_vars(let)
        for uvar in used_vars:
            self.results[uvar].uses.append(let)

def def_use(expr):
    analyzer = DefUseAnalysis()
    analyzer.visit(expr)
    return analyzer.results
