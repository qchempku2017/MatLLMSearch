import argparse
from typing import Tuple
from pathlib import Path
import random

import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure

from chgnet.model.model import CHGNet
from chgnet.model import StructOptimizer

# from .utils.config import LANTHANIDES, ACTINIDES
from .utils.llm_manager import LLMManager
from .utils.generator import StructureGenerator, GenerationResult
from .utils.evaluator import StructureEvaluator
from .utils.stability import StabilityCalculator
from .utils.e_hull_calculator import EHullCalculator
from .utils.file_util import load_gzip, load_pkl


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Task and Model arguments
    parser.add_argument('--model_path', default='./llama3')
    parser.add_argument('--backend', default='vllm')
    parser.add_argument('--fmt', choices=['poscar', 'cif'], default='poscar')
    parser.add_argument('--tensor_parallel_size', type=int, default=4)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.95)
    parser.add_argument('--api_base', type=str, default='http://localhost:11434/v1')
    parser.add_argument('--model_name', default='llama3_instruct')
    parser.add_argument('--chat_template_style', default='llama3')
    
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--reproduction_size', type=int, default=5)
    parser.add_argument('--context_size', type=int, default=2)
    parser.add_argument('--max_iter', type=int, default=10)
    
    parser.add_argument('--save_label', type=str, default='eval')
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--ppd_path', type=str, default='./2023-02-07-ppd-mp.pkl.gz')

    parser.add_argument("--seed_path", type=str, default="./seed_structures_processed_3500_matbench.csv",
                        help="Specify the path to the seed structures file. Optional.")
    parser.add_argument("--pool_size", type=int, default=3500)

    # Deprecated bulk modulus, only energy above hull is used.
    parser.add_argument(
        '--opt_goal',
                     choices=['e_hull_distance'],
                     default='e_hull_distance'
    )
    parser.add_argument('--task', choices=['csg', 'csg-space', 'csp'], default='csg')
    parser.add_argument("--chem_space", type=str, default="Li, P, S",
                        help="Chemical space for csg-space task.")
    
    return parser.parse_args()


def initialize_models(args: argparse.Namespace) -> Tuple:
    """Initialize all required models."""
    base_path = Path(f'{args.save_path}/{args.save_label}')
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize LLM
    llm_manager = LLMManager(
        model_path=args.model_path,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_base=args.api_base,
        model_name=args.model_name,
        chat_template_style=args.chat_template_style,
    )
    # Initialize Oracles
    generator = StructureGenerator(llm_manager, base_path, args)
    
    evaluator = StructureEvaluator(base_path=base_path)
    chgnet = CHGNet.load()
    relaxer = StructOptimizer()
    if not "pkl" in args.ppd_path:
        raise ValueError("Invalid PhaseDiagram file format. Must be a pickled file.")
    elif args.ppd_path.endswith(".gz"):
        ppd = load_gzip(args.ppd_path)
    else:
        ppd = load_pkl(args.ppd_path)
    e_hull_calculator = EHullCalculator(ppd)
    stability_calculator = StabilityCalculator(chgnet, relaxer, e_hull_calculator)
    
    return llm_manager, generator, evaluator, stability_calculator

    
def initialize_task_data(args: argparse.Namespace):
    """Initialize and prepare task data."""

    seed_structures_df = pd.read_csv(args.seed_path)

    seed_structures_df['structure'] = seed_structures_df['structure'].apply(lambda x: Structure.from_str(x, fmt='json') if pd.notna(x) else None)
    seed_structures_df['composition'] = [s.composition for s in seed_structures_df['structure']]
    seed_structures_df['composition_str'] = [s.composition.formula for s in seed_structures_df['structure']]
        
    required_columns = ['structure', 'composition', 'composition_str', 'composition_len', 'e_hull_distance', 'delta_e', 'bulk_modulus', 'bulk_modulus_relaxed']
    seed_structures_df = (seed_structures_df
        [(seed_structures_df['is_balanced'] == 1) & 
         (seed_structures_df['is_bond_valid'] == True) &
         (seed_structures_df['composition_len'].between(3, 6))]
    )
    # Conditional Filtering
    if 'gappbe_to_3' in seed_structures_df.columns:
        seed_structures_df = seed_structures_df.sort_values('gappbe_to_3', ascending=True)
    seed_structures_df = seed_structures_df[np.isfinite(seed_structures_df['e_hull_distance'])]
    seed_structures_df = seed_structures_df[:args.pool_size]
    # seed_structures_df = seed_structures_df[~seed_structures_df['composition_str'].str.contains('|'.join(LANTHANIDES + ACTINIDES), na=False)]
    seed_structures_df = seed_structures_df[required_columns].sample(frac=1, random_state=args.random_seed)
    seed_structures_df['source'] = args.seed_path.split("_")[-1].split(".")[0]
    print(f"Using extra pool of {len(seed_structures_df)} structures...")
    
    return seed_structures_df

def run_generation_iteration(
    iteration: int,
    curr_generation: GenerationResult,
    generator: StructureGenerator,
    evaluator: StructureEvaluator,
    stability_calculator: StabilityCalculator,
    args: argparse.Namespace
) -> GenerationResult:
    """Run a single generation iteration and return combined results."""
    print(f'Iteration {iteration}...')

    llm_structures, llm_parents = generator.generate_structures(curr_generation.structure)
    num_a = len(llm_structures)
    print(f'Generated {num_a} structures')
    
    llm_crystals, llm_structures = evaluator.to_crys(llm_structures)
    num_b = len(llm_crystals)
    validity_b = evaluator.check_validity(llm_crystals)
    print(f'Converted {num_b} structures to crystals')
    
    
    # Filter both structures and parents
    llm_structures, llm_parents = evaluator.filter_balanced_structures(llm_structures, llm_parents)
    llm_crystals, llm_structures = evaluator.to_crys(llm_structures) # call to_crystals again
    num_c = len(llm_structures)
    print(f'Filtered to {num_c} balanced structures')
    
    # Compute stability
    stability_results = stability_calculator.compute_stability(llm_structures)
    if not stability_results:
        stability_results = [None] * len(llm_structures)
    
    generation_result = process_stability_results(stability_results, llm_structures)
    list_attrs = ['parents', 'composition', 'objective', 'e_hull_distance', 
                     'delta_e', 'crystal', 'bulk_modulus', 'structure_relaxed', 'bulk_modulus_relaxed']
    for attr in list_attrs:
        value = getattr(generation_result, attr)
        if value is not None:
            if len(value) != len(generation_result.structure):
                print(f"Warning: {attr} length mismatch. Expected {len(generation_result.structure)}, got {len(value)}")
    return GenerationResult(
        structure=generation_result.structure,
        parents=llm_parents,  
        composition=generation_result.composition,
        objective=generation_result.objective,
        e_hull_distance=generation_result.e_hull_distance,
        delta_e=generation_result.delta_e,
        source=['llm'] * len(generation_result.structure),
        crystal=llm_crystals,
        bulk_modulus=generation_result.bulk_modulus,
        structure_relaxed=generation_result.structure_relaxed,
        bulk_modulus_relaxed=generation_result.bulk_modulus_relaxed,
        validity_b=validity_b,
        num_a=num_a,
        num_b=num_b,
        num_c=num_c
    )

def process_stability_results(stability_results, structures):
    result_dicts = []
    for r in stability_results:
        if r is not None:
            result_dict = {
                'e_hull_distance': r.e_hull_distance if r.e_hull_distance is not None else float('inf'),
                'delta_e': r.delta_e if r.delta_e is not None else float('inf'),
                'bulk_modulus': r.bulk_modulus if r.bulk_modulus is not None else float('-inf'),
                'bulk_modulus_relaxed': r.bulk_modulus_relaxed if r.bulk_modulus_relaxed is not None else float('-inf'),
                'structure_relaxed': getattr(r, 'structure_relaxed', None)
            }
        else:
            result_dict = {
                'e_hull_distance': float('inf'),
                'delta_e': float('inf'),
                'bulk_modulus': float('-inf'),
                'bulk_modulus_relaxed': float('-inf'),
                'structure_relaxed': None
            }
        result_dicts.append(result_dict)
    if len(result_dicts) == 0:
        return GenerationResult(**{field: [None] for field in GenerationResult.__dataclass_fields__})
    df = pd.DataFrame(result_dicts)
    return GenerationResult(
        structure=structures,
        composition=[s.composition for s in structures],
        objective=[10 * e - b if (b != float('-inf') and not pd.isna(b)) else None 
            for e, b in zip(df['e_hull_distance'], df['bulk_modulus'])],
        e_hull_distance=df['e_hull_distance'].tolist(),
        delta_e=df['delta_e'].tolist(),
        bulk_modulus=df['bulk_modulus'].tolist(),
        structure_relaxed=df['structure_relaxed'].tolist(),
        bulk_modulus_relaxed=df['bulk_modulus_relaxed'].tolist()
    )

def get_parent_generation(input_generation: GenerationResult, parent_generation: GenerationResult,
                     full_df: pd.DataFrame, sort_target: str, args: argparse.Namespace, iter: int) -> GenerationResult:
    """Get sorted generation combining input structures and parent generation results."""
    interested_columns = ['structure', 'composition', 'composition_str', 'e_hull_distance', 'delta_e', 'source']
    if input_generation and parent_generation: 
        generation_df = pd.DataFrame({
            'structure': input_generation.structure + parent_generation.structure,
            'composition': [s.composition for s in (input_generation.structure + parent_generation.structure)],
            'e_hull_distance': input_generation.e_hull_distance + parent_generation.e_hull_distance,
            'delta_e': input_generation.delta_e + parent_generation.delta_e,
            'source': input_generation.source + parent_generation.source
        })
        ascending = (sort_target not in ['bulk_modulus', 'bulk_modulus_relaxed'])        
        generation_df = pd.concat([generation_df, full_df[interested_columns]], ignore_index=True)

        generation_df['objective'] = 10 * generation_df['e_hull_distance'] - generation_df['bulk_modulus_relaxed']
        generation_df = generation_df.sort_values(sort_target, ascending=ascending)
        generation_df['composition_str'] = generation_df['composition'].apply(lambda x: x.formula)
    else:
        generation_df = full_df[interested_columns].copy()
        generation_df['objective'] = 10 * generation_df['e_hull_distance'] - generation_df['bulk_modulus_relaxed']
        generation_df = generation_df.sample(frac=1, random_state=args.random_seed)

    # Why do you have to drop structures with the same formula here?
    generation_df = generation_df.drop_duplicates(subset='composition_str')
    # generation_df = generation_df[~generation_df['composition_str'].str.contains('|'.join(LANTHANIDES + ACTINIDES), na=False)]
    generation_df = generation_df.drop(columns=['composition_str'])

    print(f'Preparing {args.topk * args.context_size} parent structures for next generation from {len(generation_df)} structures...')
    generation_df = generation_df.iloc[:args.topk * args.context_size]
    rng = np.random.RandomState(args.random_seed)
    indices = rng.choice(len(generation_df), size=args.topk * args.context_size, replace=(len(generation_df) < args.topk * args.context_size))
    sampled_df = generation_df.iloc[indices]

    parents_df = sampled_df.copy()
    parents_file = Path(f'{args.save_path}/{args.save_label}/parents.csv')
    parents_df['structure'] = parents_df['structure'].apply(lambda x: Structure.to(x, fmt='json'))
    parents_df['iteration'] = iter
    parents_df.to_csv(parents_file, mode='a', header=not parents_file.exists(), index=False)
    return GenerationResult(
        structure=sampled_df['structure'].tolist(),
        bulk_modulus=sampled_df['bulk_modulus'].tolist(),
        bulk_modulus_relaxed=sampled_df['bulk_modulus_relaxed'].tolist(),
        e_hull_distance=sampled_df['e_hull_distance'].tolist(),
        delta_e=sampled_df['delta_e'].tolist(),
        source=sampled_df['source'].tolist(),
        objective=sampled_df['objective'].tolist()
    )
    
def main():
    """Main execution function."""
    # Parse arguments and setup environment
    args = parse_arguments()
    random.seed(args.random_seed)
    # Initialize models and components
    llm_manager, generator, evaluator, stability_calculator = initialize_models(args)
    
    # Initialize task data
    seed_structures_df = initialize_task_data(args)
    # Select initial generation
    curr_generation = get_parent_generation(
        input_generation=None,
        parent_generation=None,
        full_df=seed_structures_df,
        sort_target=None,
        args = args,
        iter = 0
    )
    
    for iteration in range(1, args.max_iter + 1):
        generation_result = run_generation_iteration(
            iteration,
            curr_generation,
            generator,
            evaluator,
            stability_calculator,
            args
        )
        # Calculate metrics for evaluation
        metrics = evaluator.evaluate_generation(generation_result, iteration=iteration, args=args)
        
        # Save results
        evaluator.save_results(generation_result, metrics, iteration=iteration, args=args)
        
        # Update current generation
        opt_goal = args.opt_goal
        if opt_goal == "multi-obj":
            # opt_goal = "e_hull_distance" if iteration % 2 == 1 else "bulk_modulus_relaxed"
            opt_goal = "objective"
        curr_generation = get_parent_generation(generation_result, curr_generation, seed_structures_df, opt_goal, args, iteration)
        # e_hull_distance or bulk_modulus_relaxed
        print(f'Completed iteration {iteration}')
        print(f'Current metrics: {metrics}')

if __name__ == "__main__":
    main()






    