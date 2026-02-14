from typing import Dict, List, Optional, Protocol, Tuple, Any
import numpy.random as rnd
from core.data_structures import RvrpState

class _OperatorType(Protocol):
    __name__: str
    def __call__(self, state: RvrpState, rng: rnd.Generator, **kwargs) -> RvrpState: ...

class ALNS4PPO:
    def __init__(self, rng: rnd.Generator = rnd.default_rng()):
        self._rng = rng
        self._d_ops: Dict[str, _OperatorType] = {}
        self._r_ops: Dict[str, _OperatorType] = {}

    @property
    def destroy_operators(self) -> List[Tuple[str, _OperatorType]]:
        return list(self._d_ops.items())

    @property
    def repair_operators(self) -> List[Tuple[str, _OperatorType]]:
        return list(self._r_ops.items())

    def add_destroy_operator(self, op: _OperatorType, name: Optional[str] = None):
        op_name = name if name else getattr(op, "__name__", "unknown_destroy_op")
        self._d_ops[op_name] = op

    def add_repair_operator(self, op: _OperatorType, name: Optional[str] = None):
        op_name = name if name else getattr(op, "__name__", "unknown_repair_op")
        self._r_ops[op_name] = op

    def iterate(
        self, 
        current_solution: RvrpState, 
        pre_solution: RvrpState,
        d_idx: int, 
        r_idx: int, 
        accept_mode: int,
        **kwargs
    ) -> Tuple[RvrpState, RvrpState]:        
        d_ops_list = self.destroy_operators
        r_ops_list = self.repair_operators
        
        if d_idx >= len(d_ops_list) or r_idx >= len(r_ops_list):
            return current_solution, current_solution

        _, d_operator = d_ops_list[d_idx]
        _, r_operator = r_ops_list[r_idx]

        destroyed_state = d_operator(current_solution, self._rng, **kwargs)
        candidate_solution = r_operator(destroyed_state, self._rng, **kwargs)

        curr_obj = current_solution.objective()
        cand_obj = candidate_solution.objective()
        
        if cand_obj < curr_obj:
            return current_solution, candidate_solution
            
        else:
            if accept_mode == 1: 
                return current_solution, candidate_solution
            else:
                return current_solution, current_solution

    def reset_opt(self):
        self._d_ops.clear()
        self._r_ops.clear()
        self._rng = rnd.default_rng()