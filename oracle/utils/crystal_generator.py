import torch
from pathlib import Path
import pytorch_lightning as pl
from torch_geometric.data import DataLoader, Batch, Data
from typing import Optional, List
from pymatgen.core import Structure, Element, Lattice
from flowmm.data import NUM_ATOMIC_BITS, NUM_ATOMIC_TYPES
from flowmm.rfm.manifold_getter import ManifoldGetter
from flowmm.model.eval_utils import (
    register_omega_conf_resolvers,
    load_model,
    load_cfg,
)
from diffcsp.common.data_utils import lattices_to_params_shape

class CrystalGenerator:
    """Process crystal structures through a trained model."""
    
    def __init__(
        self, 
        checkpoint_path: str,
        batch_size: int = 32,
        num_steps: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the crystal structure processor."""
        self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        self.device = device
        
        register_omega_conf_resolvers()
        self.cfg, self.model = load_model(self.checkpoint_path)
        
        if num_steps is not None:
            self.cfg.integrate.num_steps = num_steps
            
        self.model.to(device)
        self.model.eval()
        
        self.manifold_getter = ManifoldGetter(
            atom_type_manifold=self.cfg.model.manifold_getter.atom_type_manifold,
            coord_manifold=self.cfg.model.manifold_getter.coord_manifold,
            lattice_manifold=self.cfg.model.manifold_getter.lattice_manifold,
            dataset=self.cfg.data.dataset_name,
            analog_bits_scale=self.cfg.model.manifold_getter.get("analog_bits_scale", None),
            length_inner_coef=self.cfg.model.manifold_getter.get("length_inner_coef", None),
        )

    def _structure_to_data(self, structure: Structure, batch_idx: int) -> Data:
        """Convert a pymatgen Structure to a PyG Data object with batch index."""
        # Get number of atoms
        num_atoms = len(structure)
        
        # Create batch index tensor
        batch_index = torch.full((num_atoms,), batch_idx, dtype=torch.long)
        
        # Convert atomic numbers to tensor
        atom_types = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
        
        # Convert fractional coordinates to tensor
        frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float)
        
        # Convert lattice to tensor and compute lengths and angles
        lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float)
        lengths, angles = lattices_to_params_shape(lattice.unsqueeze(0))
        
        # Create Data object with all required fields
        data = Data(
            batch=batch_index,
            atom_types=atom_types,
            frac_coords=frac_coords,
            lattices=lattice.unsqueeze(0),  # Shape: [1, 3, 3]
            lengths=lengths,  # Shape: [1, 3]
            angles=angles,   # Shape: [1, 3]
            num_atoms=torch.tensor([num_atoms], dtype=torch.long)
        )
        
        return data

    def _batch_structures(self, structures: List[Structure]) -> Batch:
        """Convert a list of structures to a properly formatted PyG Batch."""
        data_list = []
        
        for i, structure in enumerate(structures):
            data = self._structure_to_data(structure, i)
            data_list.append(data)
        
        # Create batch
        batch = Batch.from_data_list(data_list)
        
        # Ensure proper shapes for batch
        num_total_atoms = sum(len(s) for s in structures)
        num_structures = len(structures)
        print(
            batch.batch.shape,
            batch.atom_types.shape,
            batch.frac_coords.shape,
            batch.lengths.shape,
            batch.angles.shape
        )
        
        assert batch.batch.shape == (num_total_atoms,)
        assert batch.atom_types.shape == (num_total_atoms,)
        assert batch.frac_coords.shape == (num_total_atoms, 3)
        assert batch.lengths.shape == (num_structures, 3)
        assert batch.angles.shape == (num_structures, 3)
        
        return batch

    def _process_batch(self, batch: Batch) -> List[Structure]:
        """Process a batch through the model and convert results back to structures."""
        batch = batch.to(self.device)
        
        with torch.no_grad():
            results = self.model.compute_reconstruction(
                batch, 
                num_steps=self.cfg.integrate.num_steps
            )
        
        processed_structures = []
        start_idx = 0
        
        for i in range(batch.num_graphs):
            # Get number of atoms for this structure
            num_atoms = int((batch.batch == i).sum())
            
            # Extract atom types
            if self.manifold_getter.atom_type_manifold == "analog_bits":
                atom_types = self.manifold_getter._inverse_atomic_bits(
                    results["atom_types"][start_idx:start_idx + num_atoms]
                )
            else:  # simplex
                atom_types = results["atom_types"][start_idx:start_idx + num_atoms]
            
            # Convert to elements
            species = [Element.from_Z(int(z)) for z in atom_types.cpu()]
            
            # Create structure
            structure = Structure(
                lattice=Lattice(results["lattices"][i].cpu().numpy()),
                species=species,
                coords=results["frac_coords"][start_idx:start_idx + num_atoms].cpu().numpy(),
                coords_are_cartesian=False
            )
            processed_structures.append(structure)
            
            start_idx += num_atoms
            
        return processed_structures

    def generate(self, structures: List[Structure]) -> List[Structure]:
        """Process a list of structures through the model."""
        all_processed = []
        
        # Process in batches
        for i in range(0, len(structures), self.batch_size):
            batch_structures = structures[i:i + self.batch_size]
            batch = self._batch_structures(batch_structures)
            processed_batch = self._process_batch(batch)
            all_processed.extend(processed_batch)
        
        return all_processed