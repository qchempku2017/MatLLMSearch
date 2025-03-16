import numpy as np
from typing import List, Tuple, Dict, Optional
import torch
from pymatgen.core.structure import Structure
from dataclasses import dataclass
from chgnet.model.dynamics import EquationOfState

from .timeout import timeout


@dataclass
class StabilityResult:
    energy: float = None
    energy_relaxed: float = None
    delta_e: float = np.inf
    e_hull_distance: float = np.inf
    bulk_modulus: float = -np.inf
    bulk_modulus_relaxed: float = -np.inf
    structure_relaxed: Optional[Structure] = None

class StabilityCalculator:
    def __init__(self, chgnet_model, relaxer_model, e_hull_model):
        self.chgnet = chgnet_model
        self.relaxer = relaxer_model
        self.e_hull = e_hull_model
        
    def compute_stability(self, structures: List[Structure], wo_ehull=False, wo_bulk=True) -> Tuple[List[float], List[float]]:
        """Compute stability metrics for a list of structures.

        Note: 20250316: turned off default bulk modulus calculation.
        """
        results = []
        for structure in structures:
            result = self.process_single_structure(structure, wo_ehull=wo_ehull, wo_bulk=wo_bulk)
            results.append(result)
        return results

    def process_single_structure(self, structure: Structure, wo_ehull=False, wo_bulk=True) -> Optional[StabilityResult]:
        """Process single structure stability with error handling."""
        if structure.composition.num_atoms == 0:
            return None
            
        try:
            # Initial energy computation
            energy = self.compute_energy_per_atom(structure)
            if energy is None:
                return None

            # Structure relaxation
            relaxation = self.relax_structure(structure)
            if not relaxation or not relaxation['final_structure']:
                return None

            # Final energy computation
            # energy_relaxed = relaxation['trajectory'].energies[-1] # not per atom
            structure_relaxed = relaxation['final_structure']
            energy_relaxed = self.compute_energy_per_atom(structure_relaxed)

            delta_e = energy_relaxed - energy if energy_relaxed is not None else None
            # delta_e = delta_e / structure_relaxed.num_sites if ((structure_relaxed.num_sites is not None) and (delta_e is not None)) else None
            
            # E-hull distance calculation
            e_hull_distance = None if wo_ehull else self.compute_ehull_dist(structure_relaxed, energy_relaxed) 
            
            # Bulk modulus calculation
            bulk_modulus = None if wo_bulk else self.compute_bulk_modulus(structure)
            bulk_modulus_relaxed = None if wo_bulk else self.compute_bulk_modulus(structure_relaxed)
            
            return StabilityResult(
                energy=energy,
                e_hull_distance=e_hull_distance,
                delta_e=delta_e,
                bulk_modulus=bulk_modulus,
                energy_relaxed=energy_relaxed,
                bulk_modulus_relaxed=bulk_modulus_relaxed,
                structure_relaxed=structure_relaxed
            )
            
        except Exception as e:
            print(f"Error processing structure: {e}")
            return None

    @timeout(60, error_message="Energy computation timed out after 60 seconds")
    def compute_energy(self, structure: Structure) -> Optional[float]:
        """Compute structure energy."""
        try:
            prediction = self.chgnet.predict_structure(structure)
            return float(prediction['e'] * structure.num_sites)
        except Exception as e:
            print(f"Energy computation error: {e}")
            return None

    @timeout(60, error_message="Energy per atom computation timed out after 60 seconds")
    def compute_energy_per_atom(self, structure: Structure) -> Optional[float]:
        """Compute structure energy (per atom)."""
        try:
            prediction = self.chgnet.predict_structure(structure)
            return float(prediction['e'])
        except Exception as e:
            print(f"Energy per atom computation error: {e}")
            return None

    @timeout(120, error_message="Relaxation timed out after 120 seconds")
    def relax_structure(self, structure: Structure) -> Optional[Dict]:
        """Relax structure with timeout."""
        try:
            return self.relaxer.relax(structure)
        except Exception as e:
            print(f"Relaxation error: {e}")
            return None

    @timeout(60, error_message="E-hull distance computation timed out after 60 seconds")
    def compute_ehull_dist(self, structure: Structure, energy_per_atom: float) -> Optional[float]:
        """Compute energy hull distance."""
        try:
            hull_data = [{
                'structure': structure,
                'energy': energy_per_atom * structure.num_sites
            }]
            return self.e_hull.get_e_hull(hull_data)[0]['e_hull']
        except Exception as e:
            print(f"E-hull computation error: {e}")
            return np.inf

    @timeout(60, error_message="bulk modulus computation timed out after 60 seconds")
    def compute_bulk_modulus(self, structure: Structure) -> Optional[float]:
        """Compute bulk modulus."""
        try:
            eos = EquationOfState(model=self.chgnet)
            eos.fit(atoms=structure, steps=500, fmax=0.1, verbose=False)
            return eos.get_bulk_modulus(unit="eV/A^3")
        except Exception as e:
            print(f"Bulk modulus computation error: {e}")
            return 0.0
    

    def check_stability_rate(self, e_hull_distances: List[float], threshold: float = 0.03) -> Dict:
        """Compute stability statistics for given threshold."""
        if not e_hull_distances:
            return {
                f"stable_rate_{threshold}": 0.0,
                f"stable_num_{threshold}": 0,
                "min_ehull_dist": np.inf,
                "avg_ehull_dist": np.inf
            }
        valid_distances = [d for d in e_hull_distances if not (np.isnan(d) or np.isinf(d) or d is None)]
        stabilities = [d < threshold for d in e_hull_distances]
        return {
            f"stable_rate_{threshold}": np.mean(stabilities),
            f"stable_num_{threshold}": sum(stabilities),
            "min_ehull_dist": min(valid_distances) if valid_distances else np.inf,
            "avg_ehull_dist": np.mean(valid_distances) if valid_distances else np.inf
        }

    def check_local_stability(self, delta_e: List[float]) -> Dict:
        """Compute local stability statistics."""
        valid_delta_e = [d for d in delta_e 
                        if not (np.isnan(d) or np.isinf(d) or d is None)]
        return {
            "avg_delta_e": np.mean(valid_delta_e) if valid_delta_e else np.inf
        }


