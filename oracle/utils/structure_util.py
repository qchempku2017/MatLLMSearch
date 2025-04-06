import torch
import numpy as np
from pymatgen.core import Structure, Lattice, Composition
from collections import Counter

from .validity_util import structure_validity, smact_validity
from .chemical_symbols import chemical_symbols
from .featurize_util import timeout_featurize, CompFP


def lattices_to_params_shape(lattices):

    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[...,i] = torch.clamp(torch.sum(lattices[...,j,:] * lattices[...,k,:], dim = -1) /
                            (lengths[...,j] * lengths[...,k]), -1., 1.)
    angles = torch.arccos(angles) * 180.0 / np.pi

    return lengths, angles


class Crystal(object):

    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict

        self.get_structure()
        self.get_composition()
        self.get_validity()

        if self.valid:
            self.get_fingerprints()
        else:
            self.comp_fp = None
            self.struct_fp = None

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        try:
            self.comp_valid = smact_validity(self.elems, self.comps)
        except Exception as e:
            print(f"Error evaluating validity: {e}")
            self.comp_valid = False
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [timeout_featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception as e:
            print('Error constructing fingerprint for crystal. ', e)
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


def cif_str_to_crystal(cif_str):
    try:
        structure = Structure.from_str(cif_str, fmt="cif")
        crystal = Crystal({
            "frac_coords": structure.frac_coords,
            "atom_types": [chemical_symbols.index(str(x)) for x in structure.species],
            "lengths": np.array(structure.lattice.parameters[:3]),
            "angles": np.array(structure.lattice.parameters[3:])
        })
    except Exception as e:
        print(e)
        # print(cif_str)
        return None

    return crystal


def structure_to_crystal(structure):
    try:
        crystal = Crystal({
            "frac_coords": structure.frac_coords,
            "atom_types": [chemical_symbols.index(str(x)) for x in structure.species],
            "lengths": np.array(structure.lattice.parameters[:3]),
            "angles": np.array(structure.lattice.parameters[3:])
        })
    except Exception as e:
        print(e)
        # print(cif_str)
        return None

    return crystal