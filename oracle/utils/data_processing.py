import tiktoken
import pandas as pd
import numpy as np
import re
import json
from typing import Any, Dict, List, Union, Tuple
from pymatgen.core.structure import Structure

def truncate_text(text: str, max_tokens: int, encoding_name: str = "gpt-3.5-turbo") -> str:
    """Truncate text to fit within max_tokens limit."""
    max_tokens = max_tokens - 1
    text = str(text)
    try:
        encoding = tiktoken.encoding_for_model(encoding_name)
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            print(f'Text truncated from {len(tokens)} to {max_tokens}')
            text = encoding.decode(tokens[-max_tokens:])
    except:
        tokens = text.split()
        max_words = max_tokens
        if len(tokens) > max_words:
            text = ' '.join(tokens[-max_words:])
            print("Error in tiktoken encoding, using split instead.")
    return text


def sigmoid(x: float) -> float:
    """Compute sigmoid function with adjusted parameters."""
    return 0.5 + 0.5 / (1 + np.exp(-10 * (x - 0.5)))


def correct_json_string(input_string: str) -> Dict[str, Any]:
    """Correct and parse JSON string."""
    input_string = input_string.replace("'", '"').replace('(', '').replace(')', '')
    input_string = input_string.replace("True", "true").replace("False", "false")
    
    # Extract and parse sites
    sites_index = input_string.find('"sites":')
    sites = []
    if sites_index != -1:
        sites_string = input_string[sites_index:].strip('"sites":').strip()
        sites_string = correct_brackets(sites_string)
        input_string = input_string[:sites_index] + '"sites": []}'
        input_string = correct_brackets(input_string)
        sites = parse_sites(sites_string)
        
    try:
        parsed_json = json.loads(input_string)
        if isinstance(parsed_json, dict):
            parsed_json["sites"] = sites
            return round_floats(parsed_json)
        else:
            print(f"Invalid JSON structure: {input_string}")
            return None
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None


def round_floats(obj: Any, decimals: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, decimals) for item in obj]
    return obj

def parse_matrix(matrix: Union[str, List[List[float]]]) -> List[List[float]]:
    if isinstance(matrix, str):
        rows = re.findall(r'\[([^\]]+)\]', matrix)
        return [ensure_list(row, 3) for row in rows]
    elif isinstance(matrix, list) and all(isinstance(row, list) for row in matrix):
        return [ensure_list(row, 3) for row in matrix]
    else:
        raise ValueError("Invalid matrix format")


def parse_float(s: str) -> float:
    clean = re.sub(r'[^\d.+-e]+', '', s.lower())
    main, _, exp = clean.partition('e')
    main = re.sub(r'\..*', lambda m: '.' + m.group().replace('.', ''), main)
    sign = '-' if main.startswith('-') else ''
    main = main.lstrip('+-0')
    dot_index = main.find('.')
    if dot_index == -1:
        main = main or '0'
    else:
        main = (main[:dot_index] or '0') + main[dot_index:]
    exp = re.sub(r'[^\d+-]', '', exp)[:1] + re.sub(r'[^\d]', '', exp[1:])
    try:
        return float(f"{sign}{main}e{exp or '0'}")
    except ValueError:
        return 0.0  
        
def ensure_list(value: str, expected_length: int = None) -> List[float]:
    values = re.findall(r'-?\d+\.?\d*(?:e[-+]?\d+)?', value)
    result = [float(v) for v in values]
    if expected_length and len(result) != expected_length:
        raise ValueError(f"Expected {expected_length} values, got {len(result)}")
    return result
    
def clean_and_parse_list(value: str) -> List[float]:
    value = value.strip('[]')
    elements = value.split(',')
    parsed_floats = []
    for element in elements:
        parsed_floats.append(parse_float(element))
    return parsed_floats


def parse_single_site(site_dict: Dict[str, Any]) -> Dict[str, Any]:
    parsed_site = {}
    for key, value in site_dict.items():
        if key in ['abc', 'xyz']:
            if isinstance(value, str):
                # Clean and parse the string representation of the list
                parsed_site[key] = clean_and_parse_list(value)
            elif isinstance(value, list):
                # If it's already a list, clean and parse each element
                parsed_site[key] = [parse_float(re.sub(r'[^\d.eE+-]+', '', str(v))) for v in value]
            else:
                print(f"Warning: Unexpected type for {key}. Using [0.0, 0.0, 0.0]")
                parsed_site[key] = [0.0, 0.0, 0.0]
        elif key == 'properties':
            try:
                parsed_site[key] = json.loads(value)
            except:
                parsed_site[key] = {}
        elif key == 'species':
            try:
                parsed_site[key] = json.loads(value)
            except:
                parsed_site[key] = []
        elif key == 'label':
            parsed_site[key] = re.sub(r'[^a-zA-Z]', '', value)         
        else:
            try:
                parsed_site[key] = json.loads(value)
            except json.JSONDecodeError:
                parsed_site[key] = value
    return parsed_site


def parse_sites(sites_string: str) -> List[Dict[str, Any]]:
    """Parse sites from string representation."""
    sites_string = sites_string.strip('[]')
    site_keys = {'abc', 'label', 'properties', 'species', 'xyz', 'nelements'}
    sites = []
    
    site_strings = re.split(r'\},\s*\{', sites_string)
    for site_string in site_strings:
        site_string = '{' + site_string.strip('{}') + '}'
        pattern = r'"(\w+)":\s*((?:\{(?:[^{}]|\{[^{}]*\})*\}|\[(?:[^[\]]|\[(?:[^[\]]|\[[^[\]]*\])*\])*\]|"[^"]*"|-?\d+\.?\d*(?:e[-+]?\d+)?))'
        matches = re.findall(pattern, site_string)
        
        current_site = {}
        for key, value in matches:
            if key in site_keys:
                current_site[key] = value
                
        if current_site:
            sites.append(parse_single_site(current_site))
            
    return sites

def correct_brackets(s: str) -> str:
    """Correct mismatched brackets in string."""
    stack, result = [], []
    in_quotes, escape = False, False
    
    for char in s:
        if not in_quotes:
            if char in '{[':
                stack.append(char)
                result.append(char)
            elif char in '}]':
                if stack:
                    expected = '}' if stack[-1] == '{' else ']'
                    result.append(expected if char != expected else char)
                    stack.pop()
            elif char == '"':
                in_quotes = True
                result.append(char)
            else:
                result.append(char)
        else:
            if char == '"' and not escape:
                in_quotes = False
            elif char == '\\' and not escape:
                escape = True
            else:
                escape = False
            result.append(char)
            
    while stack:
        result.append('}' if stack.pop() == '{' else ']')
        
    return ''.join(result)