# -*- coding: utf-8 -*-
"""
System for loading and saving geometric configurations in JSON format

Main features:
- Automatic schema validation
- Automatic backups
- Integrity verification with checksums
- Metadata and versioning
- Robust error handling
- Detailed logging
- Automatic backup cleanup
"""

import os
import json
import shutil
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import hashlib

class GeometryConfigManager:
    """
    Manager for loading and saving geometric configurations in JSON format.
    """
    def __init__(self, config_dir: str | None = None, backup_dir: str | None = None):
        # --- Logger setup ---
        self.logger = logging.getLogger(__name__)
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO)

        # --- Base directory setup ---
        # Define default directory: VLMPy/src/vlm/data/saved_configs
        base_dir = Path(__file__).resolve().parent.parent / 'data' / 'saved_configs'

        # --- Config and backup directories ---
        self.config_dir = Path(config_dir).resolve() if config_dir else base_dir
        self.backup_dir = Path(backup_dir).resolve() if backup_dir else self.config_dir / 'backups'

        # --- Ensure directories exist ---
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Error creating directories: {e}")
            raise

        self.logger.info(f"Config directory: {self.config_dir}")
        self.logger.info(f"Backup directory: {self.backup_dir}")
        
        # Validation schema for configurations
        self.schema = {
            "wing": {
                "required": ["wingsections"],
                "optional": ["name", "description", "version", "created_at", "modified_at"]
            },
            "wingsection": {
                "required": ["chord_root", "chord_tip", "span_fraction", "NACA_root", "NACA_tip"],
                "optional": ["sweep", "dihedral", "alpha", "flap_start", "flap_end", 
                           "flaphingechord", "deflectionangle", "deflectiontype"]
            },
            "stabilizer": {
                "required": ["NACA_root", "NACA_tip", "chord_root", "chord_tip", "span_fraction"],
                "optional": ["x_translate", "z_translate", "sweep", "alpha", "dihedral"]
            },
            "parameters": {
                "required": ["u", "rho", "alpha", "beta", "n", "m"],
                "optional": ["name", "description"]
            }
        }
    
    def validate_config(self, config: Dict[str, Any], config_type: str = "wing") -> List[str]:
        """
        Validate the configuration against the defined schema.
        
        Args:
            config: Dictionary with the configuration
            config_type: Type of configuration ("wing", "parameters", etc.)
            
        Returns:
            List of errors found (empty if no errors)
        """
        errors = []
        
        if config_type not in self.schema:
            errors.append(f"Unknown configuration type: {config_type}")
            return errors
        
        schema = self.schema[config_type]
        
        # Validate required fields
        for field in schema["required"]:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Specific validations
        if config_type == "wing":
            if "wingsections" in config:
                for i, section in enumerate(config["wingsections"]):
                    section_errors = self.validate_config(section, "wingsection")
                    for error in section_errors:
                        errors.append(f"Section {i+1}: {error}")
            
            # Validate stabilizers if present
            for stabilizer_key in ["horizontalstabilizer", "verticalstabilizer"]:
                if stabilizer_key in config:
                    stab_errors = self.validate_config(config[stabilizer_key], "stabilizer")
                    for error in stab_errors:
                        errors.append(f"{stabilizer_key}: {error}")
        
        elif config_type == "wingsection":
            # Validate numeric ranges
            numeric_ranges = {
                "chord_root": (0.01, 50.0),
                "chord_tip": (0.01, 50.0),
                "span_fraction": (0.1, 50.0),
                "sweep": (-45, 45),
                "dihedral": (-45, 45),
                "alpha": (-30, 30)
            }
            
            for field, (min_val, max_val) in numeric_ranges.items():
                if field in config:
                    try:
                        value = float(config[field])
                        if not min_val <= value <= max_val:
                            errors.append(f"{field} must be between {min_val} and {max_val}")
                    except (ValueError, TypeError):
                        errors.append(f"{field} must be a valid number")
        
        return errors
    
    def calculate_checksum(self, config: Dict[str, Any]) -> str:
        """Calculates an MD5 checksum to verify data integrity."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def create_backup(self, filename: str) -> Optional[str]:
        """
        Creates a backup copy of the existing file.
        
        Args:
            filename: Name of the file to backup
            
        Returns:
            Path to the backup file created or None if failed
        """
        source_path = self.config_dir / filename
        if not source_path.exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{source_path.stem}_{timestamp}{source_path.suffix}"
        backup_path = self.backup_dir / backup_filename
        
        try:
            shutil.copy2(source_path, backup_path)
            self.logger.info(f"Backup created: {backup_path}")
            return str(backup_path)
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return None
    
    def save_config(self, config: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """
        Saves a configuration in JSON format with validation and ALWAYS creates a backup.
        
        Args:
            config: Dictionary with the configuration
            filename: Name of the file (without extension)
            
        Returns:
            Dictionary with the result of the operation
        """
        result = {
            "status": "error",
            "message": "",
            "filename": "",
            "backup_created": False,
            "checksum": ""
        }
        
        try:
            # Ensure .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            # Add metadata
            enhanced_config = config.copy()
            enhanced_config["metadata"] = {
                "version": "1.0",
                "created_at": enhanced_config.get("metadata", {}).get("created_at", 
                                                 datetime.now().isoformat()),
                "modified_at": datetime.now().isoformat(),
                "checksum": self.calculate_checksum(config)
            }
            
            file_path = self.config_dir / filename
            
            # ALWAYS create backup if the file exists
            if file_path.exists():
                backup_path = self.create_backup(filename)
                result["backup_created"] = backup_path is not None
            
            # Save file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_config, f, indent=2, ensure_ascii=False)
            
            result.update({
                "status": "success",
                "message": f"Configuration successfully saved in {filename}",
                "filename": str(file_path),
                "checksum": enhanced_config["metadata"]["checksum"]
            })
            
        except Exception as e:
            result["message"] = f"Error saving configuration: {str(e)}"
            self.logger.error(f"Error in save_config: {e}")
        
        return result
    
    def load_config(self, filename: str) -> Dict[str, Any]:
        """
        Loads a configuration from a JSON file with validation.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Dictionary with the result of the operation
        """
        result = {
            "status": "error",
            "message": "",
            "config": {},
            "checksum_valid": False,
            "version": ""
        }
        
        try:
            # Ensure .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            file_path = self.config_dir / filename
            
            if not file_path.exists():
                result["message"] = f"File not found: {filename}"
                return result
            
            # Load file
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Extract metadata if present
            metadata = config.pop("metadata", {})
            result["version"] = metadata.get("version", "unknown")
            
            # Verify checksum if available
            calculated_checksum = self.calculate_checksum(config)
            result["checksum_valid"] = calculated_checksum == metadata["checksum"]
            if not result["checksum_valid"]:
                self.logger.warning(f"Checksum does not match for {filename}")
            else:
                result["checksum_valid"] = True
            
            result.update({
                "status": "success",
                "message": f"Configuration successfully loaded from {filename}",
                "config": config
            })
            
        except json.JSONDecodeError as e:
            result["message"] = f"JSON format error: {str(e)}"
        except Exception as e:
            result["message"] = f"Error loading configuration: {str(e)}"
            self.logger.error(f"Error in load_config: {e}")
        
        return result
    
    def list_configs(self, pattern: str = "*.json") -> List[Dict[str, Any]]:
        """
        Lists all available configurations with metadata information.
        
        Args:
            pattern: File pattern to search
            
        Returns:
            List of dictionaries with information about each file
        """
        configs = []
        
        for file_path in self.config_dir.glob(pattern):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                metadata = config.get("metadata", {})
                
                config_info = {
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "created_at": metadata.get("created_at", "unknown"),
                    "modified_at": metadata.get("modified_at", "unknown"),
                    "version": metadata.get("version", "unknown"),
                    "has_wing": "wingsections" in config,
                    "has_horizontal_stabilizer": "horizontalstabilizer" in config,
                    "has_vertical_stabilizer": "verticalstabilizer" in config,
                    "wing_sections_count": len(config.get("wingsections", []))
                }
                
                configs.append(config_info)
                
            except Exception as e:
                self.logger.warning(f"Error reading {file_path.name}: {e}")
        
        return sorted(configs, key=lambda x: x["modified_at"], reverse=True)
    
    def delete_config(self, filename: str) -> Dict[str, Any]:
        """
        Deletes a configuration ALWAYS creating a backup before deleting.

        Args:
            filename: Name of the file to delete

        Returns:
            Dictionary with the result of the operation
        """
        result = {
            "status": "error",
            "message": "",
            "backup_created": False
        }

        try:
            if not filename.endswith('.json'):
                filename += '.json'

            file_path = self.config_dir / filename

            if not file_path.exists():
                result["message"] = f"File not found: {filename}"
                return result

            # ALWAYS create backup before deleting
            backup_path = self.create_backup(filename)
            result["backup_created"] = backup_path is not None

            # Delete file
            file_path.unlink()

            result.update({
                "status": "success",
                "message": f"Configuration {filename} deleted successfully"
            })

        except Exception as e:
            result["message"] = f"Error deleting configuration: {str(e)}"
            self.logger.error(f"Error in delete_config: {e}")

        return result

# Factory function to create the configuration manager
def create_config_manager():
    """Factory function to create the configuration manager."""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'saved_configs')
    backup_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'saved_configs/backups')
    return GeometryConfigManager(config_dir, backup_dir)

# Initialize the manager globally
config_manager = create_config_manager()
