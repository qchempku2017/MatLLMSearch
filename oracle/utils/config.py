import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['VLLM_ALLOW_RUNTIME_LORA_UPDATING'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['ENFORCE_EAGER'] = 'true'

# Crystal Structure Generation
PROMPT_PATTERN_CSG = """You are an expert material scientist. Your task is to propose hypotheses for {rep_size} new materials with valid stable structures and compositions. No isolated or overlapped atoms are allowed.

The proposed new materials can be a modification or combination of the base materials given below.

Format requirements:
1. Each proposed structure must be formatted in JSON with the following structure:
{{
    "i": {{
        "formula": "composition_formula",
        "{fmt}": "{fmt}_format_string"
    }}
}}
2. Use proper JSON escaping for newlines (\\n) and other special characters

Base material structure for reference:
{input}

Your task:
1. Generate {rep_size} new structure hypotheses
2. Each structure should be stable and physically reasonable
3. Format each structure exactly as shown in the input

Output your hypotheses below:
"""

# Crystal Structure Generation in a specific chemical space.
PROMPT_PATTERN_CSGS = """You are an expert material scientist. Your task is to propose hypotheses for {rep_size} new materials with valid stable structures and compositions.

The generated materials must contain only elements {chem_space}. At least one of these elements must be present, and no other elements are allowed.
 
No isolated or overlapped atoms are allowed.

The proposed new materials can be a modification or combination of the base materials given below.

Format requirements:
1. Each proposed structure must be formatted in JSON with the following structure:
{{
    "i": {{
        "formula": "composition_formula",
        "{fmt}": "{fmt}_format_string"
    }}
}}
2. Use proper JSON escaping for newlines (\\n) and other special characters

Base material structure for reference:
{input}

Your task:
1. Generate {rep_size} new structure hypotheses
2. Each structure should be stable and physically reasonable
3. Format each structure exactly as shown in the input

Output your hypotheses below:
"""

# Crystal Structure Prediction (deprecated)
# PROMPT_PATTERN_CSP = """You are an expert material scientist. Your task is to design {rep_size} novel, thermodynamically stable variants of sodium aluminum chloride with the general formula (Na3AlCl6)*n, where n represents different multiples of the base composition. No isolated or overlapped atoms are allowed. You may refer to the reference structures provided below as inspiration, but ensure you propose novel atomic arrangements beyond simple atomic substitution.
#
# Crystal structures for reference:
# {input}
#
# Format requirements:
# 1. Each proposed structure must be formatted in JSON with the following structure:
# {{
#     "i": {{
#         "formula": "Na3AlCl6",
#         "{fmt}": "{fmt}_format_string"
#     }}
# }}
# 2. Use proper JSON escaping for newlines (\\n) and other special characters.
#
# Output your hypotheses below:
# """

# Model Settings (deprecated)
# MODEL_SETTINGS = {
#     "70b": {
#         "max_token_length": 7904,
#     },
#     "mistral": {
#         "max_token_length": 32000,
#     }
# }

# Stability Thresholds
STABILITY_THRESHOLDS = [0.03, 0.06]

# API Keys, deprecated as local machine has internet restriction.
# OPENAI_API_KEY = ""
# HF_TOKEN = ""
# HF_TOKEN_W = ""


LANTHANIDES = ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
ACTINIDES = ['Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
METAL_OXIDATION_STATES= {
    "Li": [1],
    "Na": [1],
    "K": [1],
    "Rb": [1],
    "Cs": [1],
    "Be": [2],
    "Mg": [2],
    "Ca": [2],
    "Sr": [2],
    "Ba": [2],
    "Sc": [3],
    "Ti": [4, 3],
    "V": [5, 4, 3, 2],
    "Cr": [2, 3, 4, 6],
    "Mn": [2, 3, 4, 7],
    "Fe": [2, 3, 4],
    "Co": [2, 3, 4],
    "Ni": [2, 3, 4],
    "Cu": [2, 1],
    "Zn": [2],
    "Y": [3],
    "Zr": [4],
    "Nb": [5],
    "Mo": [6, 4],
    "Tc": [7],
    "Ru": [4, 3],
    "Rh": [3],
    "Pd": [2, 4],
    "Ag": [1],
    "Cd": [2],
    "La": [3],
    "Hf": [4],
    "Ta": [5],
    "W": [6],
    "Re": [7],
    "Os": [4, 8],
    "Ir": [3, 4],
    "Pt": [2, 4],
    "Au": [3, 1],
    "Hg": [2, 1],
    "Al": [3],
    "Ga": [3],
    "In": [3],
    "Tl": [1, 3],
    "Sn": [4, 2],
    "Pb": [2, 4],
    "Bi": [3],
}
