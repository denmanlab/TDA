"""
Utilities for TDA 

allows for data input and handling across different OS in the TDA repo.

@emilyekstrum
11/25/25
"""

import os
import zipfile
import pickle as pkl
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import glob
import re


class TDADataManager:
    """
    data management for TDA workflows
    handles cross-OS paths 
    """
    
    def __init__(self, workspace_root: Optional[Union[str, Path]] = None):
        """
        Initialize TDA Data Manager
        
        Args:
            workspace_root: optional path to workspace root. if None, auto-detects.
        """

        self.workspace_root = self._find_workspace_root(workspace_root)
        self.data_paths = self._setup_data_paths()
        self._extract_sample_data_if_needed()
    
    def _find_workspace_root(self, provided_root: Optional[Union[str, Path]] = None) -> Path:
        """Find the TDA repository root directory."""

        if provided_root:
            root = Path(provided_root).resolve()
            if self._is_tda_repo(root):
                return root
        
        # search from current working directory upwards
        current = Path.cwd()
        for path in [current] + list(current.parents):
            if self._is_tda_repo(path):
                return path
        
        # search common locations
        search_paths = [
            Path.cwd().parent / 'TDA',
            Path.cwd() / 'TDA',
            Path.home() / 'repos' / 'TDA',
            Path.home() / 'Documents' / 'TDA',
        ]
        
        for path in search_paths:
            if self._is_tda_repo(path):
                return path
        
        raise FileNotFoundError(
            "Could not find TDA repository root. Make sure you're running from within "
            "the TDA repository or provide the correct workspace_root path."
        )
    
    def _is_tda_repo(self, path: Path) -> bool:
        """Check if a path looks like the TDA repository."""

        if not path.exists():
            return False
        
        # check for characteristic files/folders
        required_items = ['data', 'notebooks']
        optional_items = ['tda_environment.yml', 'README.md']
        
        required_found = all((path / item).exists() for item in required_items)
        optional_found = any((path / item).exists() for item in optional_items)
        
        return required_found and optional_found
    
    def _setup_data_paths(self) -> Dict[str, Path]:
        """Set up all data directory paths."""

        data_root = self.workspace_root / 'data'
        
        return {
            'data_root': data_root,
            'clean_spike_data_zip': data_root / 'clean_spike_data.zip',
            'clean_spike_data_dir': data_root / 'clean_spike_data',
            'cebra_examples': data_root / 'CEBRA_embedding_examples',
            'persistence_examples': data_root / 'persistence_diagram_examples',
            'all_dgms_zip': data_root / 'all_dgms.zip',  # new zip file
            'all_dgms_dir': data_root / 'all_dgms'       # extracted directory
        }
    
    def _extract_sample_data_if_needed(self) -> None:
        """Extract sample data zip files if they exist and haven't been extracted yet."""

        # extract clean_spike_data.zip
        zip_path = self.data_paths['clean_spike_data_zip']
        extract_dir = self.data_paths['clean_spike_data_dir']
        
        if zip_path.exists() and not extract_dir.exists():
            print(f"Extracting sample data from {zip_path.name}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_paths['data_root'])
            except Exception as e:
                warnings.warn(f"Failed to extract data: {e}")
        elif extract_dir.exists():
            print("Spike data already available.")
        
        # extract all_dgms.zip
        self._extract_all_dgms_if_needed()
    
    def _extract_all_dgms_if_needed(self) -> None:
        """Extract all_dgms.zip if it exists and hasn't been extracted yet."""
        
        zip_path = self.data_paths['all_dgms_zip']
        extract_dir = self.data_paths['all_dgms_dir']
        
        if zip_path.exists() and not extract_dir.exists():
            print(f"Extracting persistence diagrams from {zip_path.name}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_paths['data_root'])
                print(f"Extracted to: {extract_dir}")
            except Exception as e:
                warnings.warn(f"Failed to extract all_dgms.zip: {e}")
        elif extract_dir.exists():
            print("All persistence diagrams already available.")
    
    def get_all_persistence_files(self, pattern: str = "*.pkl") -> List[Path]:
        """
        Get all persistence diagram files from both persistence_examples and all_dgms directories.
        
        Args:
            pattern: Glob pattern to match files (default: "*.pkl")
        
        Returns:
            List of file paths from both directories
        """
        
        files = []
        
        # check persistence_examples directory
        persistence_dir = self.data_paths['persistence_examples']
        if persistence_dir.exists():
            files.extend(list(persistence_dir.glob(pattern)))
        
        # check all_dgms directory
        all_dgms_dir = self.data_paths['all_dgms_dir']
        if all_dgms_dir.exists():
            files.extend(list(all_dgms_dir.rglob(pattern)))  # recursive search
        
        # ALSO check for persistence diagram files directly in data root
        # (since all_dgms.zip extracts files directly to data root, not to subdirectory)
        data_root = self.data_paths['data_root']
        if data_root.exists():
            # Look for files that match CEBRA pattern (these are persistence diagrams)
            cebra_files = list(data_root.glob("CEBRA_*.pkl"))
            files.extend(cebra_files)
        
        return sorted(files)
    
    def get_available_datasets(self, pattern: str = "*.pkl") -> List[Tuple[str, Path]]:
        """
        Find all available datasets matching the pattern.
        
        Args:
            pattern: Glob pattern to match files (default: "*.pkl")
        
        Returns:
            List of (filename, filepath) tuples
        """

        datasets = []
        
        # search in clean spike data directory
        clean_data_dir = self.data_paths['clean_spike_data_dir']
        if clean_data_dir.exists():
            for file_path in clean_data_dir.glob(pattern):
                datasets.append((file_path.name, file_path))
        
        return sorted(datasets)
    
    def load_spike_data(self, dataset_name: Optional[str] = None) -> Tuple[List, List, Dict]:
        """
        data loader for spike data.
        
        Args:
            dataset_name: Optional specific dataset name to load
        
        Returns:
            Tuple of (datas, recordings, dataset_info)
        """

        available_datasets = self.get_available_datasets()
        
        if not available_datasets:
            raise FileNotFoundError(
                "No datasets found. Make sure clean_spike_data.zip is extracted or "
                "add .pkl files to the data/clean_spike_data/ folder."
            )
        
        # select dataset
        if dataset_name is None:
            if len(available_datasets) == 1:
                selected_name, selected_path = available_datasets[0]
                print(f"Loading the only available dataset: {selected_name}")
            else:
                print("Multiple datasets available:")
                for i, (name, _) in enumerate(available_datasets, 1):
                    print(f"  {i}. {name}")
                print("\nUsing the first one. To specify, use dataset_name parameter.")
                selected_name, selected_path = available_datasets[0]
        else:
            # find exact or partial match
            matches = [(name, path) for name, path in available_datasets 
                      if dataset_name.lower() in name.lower()]
            if not matches:
                available_names = [name for name, _ in available_datasets]
                raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available_names}")
            selected_name, selected_path = matches[0]
        
        # load the data
        print(f"Loading dataset: {selected_name}")
        try:
            with open(selected_path, 'rb') as f:
                loaded = pkl.load(f)
            
            datas = loaded['datas']
            recordings = loaded['recordings']
            
            dataset_info = {
                'filename': selected_name,
                'path': selected_path,
                'n_sessions': len(recordings),
                'session_names': recordings
            }
            
            print(f"Loaded {len(recordings)} recording sessions")
            print(f"  Sessions: {recordings}")
            if len(datas) > 0:
                print(f"  Data shape: {datas[0].shape}")
            
            return datas, recordings, dataset_info
            
        except Exception as e:
            raise RuntimeError(f"Error loading {selected_name}: {e}")
    
    def find_files(self, 
                   pattern: str = "*.pkl", 
                   directory: str = "persistence_examples",
                   recursive: bool = True) -> List[Path]:
        """
        Find files matching pattern in specified directory.
        
        Args:
            pattern: Glob pattern to match
            directory: Directory name (key from data_paths or relative path)
            recursive: Whether to search recursively
        
        Returns:
            List of matching file paths
        """

        if directory in self.data_paths:
            search_dir = self.data_paths[directory]
        else:
            search_dir = self.data_paths['data_root'] / directory
        
        if not search_dir.exists():
            return []
        
        if recursive:
            pattern = f"**/{pattern}"
        
        return list(search_dir.glob(pattern))
    
    def load_persistence_diagrams(self, filepath: Union[str, Path]) -> List[np.ndarray]:
        """
        Load persistence diagrams from pickle file.
        
        Args:
            filepath: Path to pickle file containing persistence diagrams
        
        Returns:
            List of persistence diagrams [H0, H1, H2, ...]
        """

        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            obj = pkl.load(f)
        
        # handle different pickle formats
        if isinstance(obj, (list, tuple)):
            return [np.asarray(dgm) for dgm in obj]
        
        if isinstance(obj, dict):
            if 'dgms' in obj:
                dgms = obj['dgms']
                return [np.asarray(dgm) for dgm in dgms]
            
            # try numeric keys 
            max_key = max(k for k in obj.keys() if isinstance(k, int)) if obj else -1
            if max_key >= 0:
                return [np.asarray(obj[k]) for k in range(max_key + 1) if k in obj]
        
        raise ValueError(f"Unrecognized persistence diagram format in {filepath}")
    
    def list_all_dgms_files(self) -> List[Path]:
        """
        List all persistence diagram .pkl files from both persistence_examples and all_dgms directories.
        
        Returns:
            List of file paths
        """
        return self.get_all_persistence_files(pattern="*.pkl")
    
    def load_embedding_data(self, filepath: Union[str, Path], force_cpu: bool = True) -> Tuple[Dict, List[str]]:
        """
        Load embedding data from pickle file with CPU compatibility.
        
        Args:
            filepath: Path to pickle file containing embedding data
            force_cpu: Force CPU loading to avoid CUDA issues
        
        Returns:
            Tuple of (session_dict, session_names)
        """
        return load_embedding_data_safe(filepath, force_cpu=force_cpu)
    
    def parse_filename_info(self, filepath: Union[str, Path]) -> Dict[str, str]:
        """
        Parse filename to extract metadata like region, stimulus, dimension, etc.
        
        Args:
            filepath: Path to file
        
        Returns:
            Dictionary with extracted metadata
        """

        filepath = Path(filepath)
        filename = filepath.stem.lower()
        
        # extract region
        region = 'unknown'
        if 'v1' in filename:
            region = 'V1'
        elif 'lgn' in filename:
            region = 'LGN'
        
        # extract dimension
        dimension = 'unknown'
        for dim in ['3d', '8d', '24d', '32d']:
            if dim in filename:
                dimension = dim
                break
        
        # extract stimulus
        stimulus_map = {
            'cex': 'color_exchange',
            'dg': 'drifting_gratings', 
            'cg': 'chromatic_gratings',
            'lf': 'luminance_flash'
        }
        
        stimulus = 'unknown'
        for code, full_name in stimulus_map.items():
            if code in filename or full_name in filename:
                stimulus = full_name
                break
        
        # extract mouse ID
        mouse_id = 'unknown'
        mouse_match = re.search(r'[cd]\d+', filename)  # match C153, d5, etc
        if mouse_match:
            mouse_id = mouse_match.group().upper()
        
        # check for shuffled/random data
        is_shuffled = 'shuffled' in filename
        is_random = 'random' in filename
        
        return {
            'region': region,
            'dimension': dimension,
            'stimulus': stimulus,
            'mouse_id': mouse_id,
            'is_shuffled': is_shuffled,
            'is_random': is_random,
            'filename': filepath.name,
            'stem': filepath.stem
        }
    
    def get_output_path(self, 
                       filename: str, 
                       subdirectory: str = "outputs",
                       create_dirs: bool = True) -> Path:
        """
        Get standardized output path for saving results.
        
        Args:
            filename: Output filename
            subdirectory: Subdirectory within data folder
            create_dirs: Whether to create directories if they don't exist
        
        Returns:
            Full output path
        """

        output_dir = self.data_paths['data_root'] / subdirectory
        
        if create_dirs:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir / filename
    
    def print_summary(self) -> None:
        """Print summary of available data and paths."""

        print("TDA Data Manager Summary")
        print(f"Workspace root: {self.workspace_root}")
        print(f"Data directory: {self.data_paths['data_root']}")
        
        # check each data directory
        for name, path in self.data_paths.items():
            if name == 'data_root':
                continue
            
            status = "-" if path.exists() else "X"
            print(f"  {status} {name}: {path.name}")
            
            if path.exists() and path.is_dir():
                pkl_files = list(path.glob("*.pkl"))
                if pkl_files:
                    print(f"      Contains {len(pkl_files)} .pkl files")
        
        # show available datasets
        datasets = self.get_available_datasets()
        if datasets:
            print(f"\nAvailable spike datasets ({len(datasets)}):")
            for name, _ in datasets[:5]:  # show first 5
                print(f"  • {name}")
            if len(datasets) > 5:
                print(f"  and {len(datasets) - 5} more")


# global instance of data manager - can be imported and used directly
try:
    tda_manager = TDADataManager()
except Exception as e:
    warnings.warn(f"Could not initialize TDA Data Manager: {e}")
    tda_manager = None


# convenience functions for backward compatibility
def setup_data_paths() -> Dict[str, Path]:
    """Legacy function - use TDADataManager instead."""

    if tda_manager is None:
        raise RuntimeError("TDA Data Manager not initialized")
    return tda_manager.data_paths


def load_spike_data_smart(dataset_name: Optional[str] = None) -> Tuple[List, List, Dict]:
    """Legacy function - use TDADataManager.load_spike_data instead."""

    if tda_manager is None:
        raise RuntimeError("TDA Data Manager not initialized")
    return tda_manager.load_spike_data(dataset_name)


def find_available_datasets(data_paths: Dict[str, Path]) -> List[Tuple[str, Path]]:
    """Legacy function - use TDADataManager.get_available_datasets instead."""

    if tda_manager is None:
        raise RuntimeError("TDA Data Manager not initialized")
    return tda_manager.get_available_datasets()


# utility functions for common data handling and loading
def safe_model_device_handling(force_cpu: bool = False) -> str:
    """
    Determine the best device for model training/inference.
    
    Args:
        force_cpu: Force CPU usage regardless of CUDA availability
    
    Returns:
        Device string ('cpu' or 'cuda')
    """

    try:
        import torch
        
        if force_cpu:
            return 'cpu'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'


def load_cebra_model_safe(filepath: Union[str, Path], 
                         force_cpu: bool = False) -> Tuple[Dict, str]:
    """
    Safely load CEBRA model from pickle file with device compatibility.
    
    Args:
        filepath: Path to pickle file containing CEBRA model
        force_cpu: Force model to CPU regardless of CUDA availability
    
    Returns:
        Tuple of (loaded_data, device_info)
    """
    import torch
    
    target_device = safe_model_device_handling(force_cpu)
    device_info = f"Using {target_device.upper()}"
    
    print(f"Loading model: {device_info}")
    
    try:
        # first try normal loading
        with open(filepath, 'rb') as f:
            loaded_data = pkl.load(f)
        print("Successfully loaded model normally")
        
    except RuntimeError as e:
        if "CUDA" in str(e) and "torch.cuda.is_available() is False" in str(e):
            print("CUDA compatibility issue detected, forcing CPU mapping")
            
            # custom unpickler that maps CUDA tensors to CPU
            class CPUUnpickler(pkl.Unpickler):
                def find_class(self, module, name):
                    if module == 'torch.storage' and name == '_load_from_bytes':
                        return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                    else:
                        return super().find_class(module, name)
            
            # other approach: use torch.load directly if the file contains torch objects
            try:
                import io
                with open(filepath, 'rb') as f:
                    # try loading as a torch file first
                    try:
                        loaded_data = torch.load(f, map_location='cpu')
                        print("Successfully loaded using torch.load with CPU mapping")
                    except:
                        #  try custom unpickler
                        f.seek(0)
                        loaded_data = CPUUnpickler(f).load()
                        print("Successfully loaded using custom CPU unpickler")
                        
            except Exception as inner_e:
                #try to reconstruct the data structure
                print(f"Advanced loading failed: {inner_e}")
                print("Recommendation: Re-train the model using the CPU training cell")
                raise RuntimeError(
                    f"Could not load model due to CUDA/CPU compatibility. "
                    f"Original error: {e}\n"
                    f"Please use the 'Train CEBRA - CPU' cell to create new embeddings."
                )
        else:
            # re-raise if it's not a CUDA issue
            raise e
    
    # update model devices if needed and models are present
    if isinstance(loaded_data, dict):
        for session_name in loaded_data.keys():
            if isinstance(loaded_data[session_name], dict) and 'model' in loaded_data[session_name]:
                model = loaded_data[session_name]['model']
                if hasattr(model, 'device_') and model.device_ != target_device:
                    print(f"Moving {session_name} model from {model.device_} to {target_device}")
                    model.device_ = target_device
                
                # also update any torch tensors in embeddings
                if 'embedding' in loaded_data[session_name]:
                    embedding = loaded_data[session_name]['embedding']
                    if hasattr(embedding, 'cpu'):  # It's a torch tensor
                        loaded_data[session_name]['embedding'] = embedding.cpu().numpy()
                        print(f"Converted {session_name} embedding to CPU numpy array")
    
    return loaded_data, device_info


def load_embedding_data_safe(filepath: Union[str, Path], 
                            force_cpu: bool = True) -> Tuple[Dict, List[str]]:
    """
    Safely load embedding data from pickle file with CPU compatibility.
    
    Args:
        filepath: Path to pickle file containing embedding data
        force_cpu: Force CPU loading to avoid CUDA issues
    
    Returns:
        Tuple of (session_dict, session_names)
    """
    try:
        loaded_data, device_info = load_cebra_model_safe(filepath, force_cpu=force_cpu)
        session_names = list(loaded_data.keys())
        return loaded_data, session_names
    except Exception as e:
        print(f"Could not load embedding data: {e}")
        print("Try training new embeddings using the CPU training cell")
        raise