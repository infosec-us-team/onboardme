"""Local Solidity project analysis module"""
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import hashlib

from slither import Slither
from analysis.flow_walk import build_entry_point_flows
from analysis.state_vars import _state_var_record

ProgressCallback = Callable[[str, Dict[str, Any] | None], None]


class LocalProjectAnalyzer:
    """Local Solidity project analyzer"""
    
    def __init__(self, source_dir: Path, progress_cb: Optional[ProgressCallback] = None):
        self.source_dir = source_dir.resolve()
        self.progress_cb = progress_cb
        self.slither = None
        
    def _report_progress(self, message: str, **meta):
        """Report progress"""
        if self.progress_cb:
            try:
                self.progress_cb(message, meta or None)
            except Exception:
                pass
    
    def _validate_source_dir(self) -> None:
        """Validate source directory"""
        if not self.source_dir.exists():
            raise ValueError(f"Directory does not exist: {self.source_dir}")
        
        if not self.source_dir.is_dir():
            raise ValueError(f"Path is not a directory: {self.source_dir}")
    
    def _detect_solidity_files(self) -> List[Path]:
        """Detect Solidity files in the directory"""
        sol_files = list(self.source_dir.rglob("*.sol"))
        if not sol_files:
            raise ValueError(f"No Solidity files (.sol) found in {self.source_dir}")
        return sol_files
    
    def _get_project_hash(self) -> str:
        """Generate a unique hash for the project directory"""
        # Create a string representation of the directory structure
        dir_info = []
        for sol_file in self.source_dir.rglob("*.sol"):
            if sol_file.is_file():
                rel_path = str(sol_file.relative_to(self.source_dir))
                content = sol_file.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                dir_info.append(f"{rel_path}:{digest}")
        
        if not dir_info:
            raise ValueError("No Solidity files found for hashing")
            
        dir_info_str = "|".join(sorted(dir_info))
        return hashlib.md5(dir_info_str.encode()).hexdigest()[:12]
    
    def _detect_main_contract(self) -> Optional[str]:
        """
        Automatically detect the main contract
        
        Heuristic rules:
        1. Avoid interfaces and libraries
        2. Avoid test/mock contracts
        3. Prioritize contracts with external/public functions
        4. Prioritize contracts with constructors
        """
        if not self.slither:
            return None
            
        contracts = self.slither.contracts
        if not contracts:
            return None
        
        self._report_progress(f"Found {len(contracts)} contracts")
        
        # Filter out interfaces and libraries
        candidate_contracts = []
        for contract in contracts:
            contract_name = contract.name
            
            # Skip interfaces and libraries
            if getattr(contract, 'is_interface', False) or getattr(contract, 'is_library', False):
                continue
                
            # Skip test/mock contracts
            if (contract_name.lower().endswith('test') or 
                'mock' in contract_name.lower() or
                'test' in contract_name.lower()):
                continue
            
            candidate_contracts.append(contract)
        
        if not candidate_contracts:
            # If no candidates, return first non-interface/library contract
            for contract in contracts:
                if not getattr(contract, 'is_interface', False) and not getattr(contract, 'is_library', False):
                    return contract.name
            return None
        
        # Score candidates
        contract_scores = {}
        for contract in candidate_contracts:
            score = 0
            contract_name = contract.name
            
            # Rule 1: Contract size (number of functions)
            func_count = len(contract.functions)
            score += min(func_count / 5, 10)
            
            # Rule 2: Has constructor
            if contract.constructor:
                score += 5
                
            # Rule 3: Number of external/public functions
            external_funcs = [f for f in contract.functions 
                            if getattr(f, 'visibility', '') in ['external', 'public']]
            score += len(external_funcs) * 2
            
            # Rule 4: Is inherited from by other contracts
            inheritors = sum(1 for c in contracts if contract in getattr(c, 'inheritance', []))
            score += inheritors * 3
            
            contract_scores[contract_name] = score
        
        # Return highest scoring contract
        main_contract = max(contract_scores.items(), key=lambda x: x[1])[0]
        return main_contract
    
    def _get_slither_instance(self) -> Slither:
        """Get Slither instance for the project"""
        try:
            # Try direct analysis first
            self._report_progress("Analyzing with Slither...")
            return Slither(str(self.source_dir))
        except Exception as e:
            # If direct analysis fails, try with crytic-compile
            self._report_progress("Compiling with crytic-compile...")
            try:
                from crytic_compile import cryticparser
                from crytic_compile.crytic_compile import CryticCompile
                
                crytic_compile = CryticCompile(
                    str(self.source_dir), 
                    **cryticparser.parsing_args()
                )
                return Slither(crytic_compile)
            except ImportError:
                raise ValueError(
                    "crytic-compile is required for complex projects. "
                    "Install with: pip install crytic-compile"
                )
            except Exception as e2:
                raise ValueError(f"Analysis failed: {str(e2)}")
    
    def analyze(self, contract_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze local Solidity project
        
        Args:
            contract_name: Specified main contract name, if None then auto-detect
            
        Returns:
            Dictionary containing analysis results
        """
        # 1. Validate directory
        self._validate_source_dir()
        
        # 2. Check for Solidity files
        sol_files = self._detect_solidity_files()
        self._report_progress(f"Found {len(sol_files)} Solidity files")
        
        # 3. Analyze with Slither
        self.slither = self._get_slither_instance()
        
        # 4. Get available contracts
        available_contracts = [c.name for c in self.slither.contracts]
        if not available_contracts:
            raise ValueError("No contracts found in the project")
        
        # 5. Determine main contract
        if contract_name:
            if contract_name not in available_contracts:
                raise ValueError(
                    f"Contract '{contract_name}' not found. "
                    f"Available contracts: {', '.join(available_contracts)}"
                )
            main_contract_name = contract_name
        else:
            main_contract_name = self._detect_main_contract()
            if not main_contract_name:
                raise ValueError(
                    f"Unable to automatically determine main contract. "
                    f"Please specify one of: {', '.join(available_contracts)}"
                )
        
        self._report_progress(f"Analyzing contract: {main_contract_name}")
        
        # 6. Get target contract
        target_contracts = [c for c in self.slither.contracts if c.name == main_contract_name]
        if not target_contracts:
            raise ValueError(f"Contract '{main_contract_name}' not found in Slither analysis")
        
        # 7. Build execution flows
        self._report_progress("Building execution flow graph...")
        try:
            data = build_entry_point_flows(
                self.slither, 
                target_contracts, 
                progress_cb=self.progress_cb
            )
        except Exception as e:
            raise ValueError(f"Failed to build execution flows: {str(e)}")
        
        # 8. Collect state variables (filter out constant variables)
        extra_storage_vars = []
        seen_vars = set()
        
        for contract in self.slither.contracts:
            for var in getattr(contract, "state_variables", []) or []:
                # Skip constant variables
                if getattr(var, "is_constant", False):
                    continue
                    
                try:
                    record = _state_var_record(var)
                    key = record.get("qualified_name") or record.get("name", "")
                    if not key or key in seen_vars:
                        continue
                        
                    seen_vars.add(key)
                    extra_storage_vars.append(record)
                except Exception as e:
                    # Skip variables that can't be processed
                    continue
        
        # 9. Generate project hash
        project_hash = self._get_project_hash()
        
        return {
            "project_hash": project_hash,
            "main_contract": main_contract_name,
            "contracts": available_contracts,
            "entry_points": data,
            "state_variables": extra_storage_vars,
            "source_dir": str(self.source_dir),
            "slither": self.slither
        }