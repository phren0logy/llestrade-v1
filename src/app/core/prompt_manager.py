"""
Prompt management system for LLM interactions.
Handles loading and formatting of prompt templates.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from src.config.prompt_store import get_bundled_dir, get_custom_dir
from src.app.core.bundled_prompts import canonical_prompt_key


class PromptManager:
    """Manages loading and formatting of prompt templates.
    
    Templates are stored in text files in a specified directory.
    Template files can include placeholders in {variable} format.
    """
    
    # Default search precedence: user custom → user bundled
    DEFAULT_PROMPTS_DIRS = [get_custom_dir(), get_bundled_dir()]

    def __init__(self, template_dir: Optional[str | Path] = None):
        """Initialize the prompt manager with a template directory.
        
        Args:
            template_dir: Directory containing prompt template files
        """
        # Support a single directory or default to the user store dirs
        if template_dir is not None:
            self.template_dirs = [Path(template_dir)]
        else:
            self.template_dirs = list(self.DEFAULT_PROMPTS_DIRS)
        self.templates: Dict[str, str] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load all templates from the template directory."""
        if not self.template_dirs:
            return
        for directory in self.template_dirs:
            if not directory.exists():
                continue
            for template_file in directory.glob("*.md"):
                try:
                    # First directory wins (custom overrides bundled)
                    stem = template_file.stem
                    if stem in self.templates:
                        continue
                    with open(template_file, 'r', encoding='utf-8') as f:
                        self.templates[stem] = f.read()
                    logging.debug(f"Loaded template: {stem} from {directory}")
                except Exception as e:
                    logging.error(f"Error loading template {template_file}: {e}")
    
    def get_template(self, name: str, **kwargs: Any) -> str:
        """Get a template by name with optional formatting.
        
        Args:
            name: Name of the template (without extension)
            **kwargs: Variables to format into the template
            
        Returns:
            The template content with variables substituted
            
        Raises:
            KeyError: If template not found
        """
        canonical = canonical_prompt_key(name)
        if canonical not in self.templates:
            raise KeyError(f"Template not found: {name}")
            
        template = self.templates[canonical]
        return template.format(**kwargs)
        
    def get_system_prompt(self) -> str:
        """Get the system prompt.
        
        Returns:
            The system prompt content, or a default if not found
        """
        try:
            return self.get_template("default_system")
        except KeyError:
            # Default system prompt if not found in templates
            logging.warning("Using default system prompt as template not found")
            return (
                "You are an advanced assistant designed to help a forensic psychiatrist. "
                "Your task is to analyze and objectively document case information in a formal clinical style, "
                "maintaining professional psychiatric documentation standards. Distinguish between information "
                "from the subject and objective findings. Report specific details such as dates, frequencies, "
                "dosages, and other relevant clinical data. Document without emotional language or judgment."
            )
    
    def get_prompt_template(self, name: str) -> Dict[str, str]:
        """Get a prompt template that contains both system and user prompts.
        
        This method looks for a template file and parses it to extract
        system and user prompt sections. The template should contain:
        - ## System Prompt section
        - ## User Prompt section
        
        Args:
            name: Name of the template (without extension)
            
        Returns:
            Dictionary with 'system_prompt' and 'user_prompt' keys
            
        Raises:
            KeyError: If template not found
        """
        if name not in self.templates:
            # Try loading a specific prompt file if not already loaded
            prompt_file = self.template_dir / f"{name}_prompt.md"
            if prompt_file.exists():
                try:
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        self.templates[name] = f.read()
                except Exception as e:
                    logging.error(f"Error loading prompt template {prompt_file}: {e}")
                    raise KeyError(f"Template not found: {name}")
            else:
                raise KeyError(f"Template not found: {name}")
        
        template_content = self.templates[name]
        
        # Parse the template to extract system and user prompts
        system_prompt = ""
        user_prompt = ""
        
        # Split by common section markers
        lines = template_content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            # Check for section headers
            if line.strip().lower().startswith('## system prompt') or line.strip().lower().startswith('# system prompt'):
                if current_section == 'user' and current_content:
                    user_prompt = '\n'.join(current_content).strip()
                current_section = 'system'
                current_content = []
            elif line.strip().lower().startswith('## user prompt') or line.strip().lower().startswith('# user prompt'):
                if current_section == 'system' and current_content:
                    system_prompt = '\n'.join(current_content).strip()
                current_section = 'user'
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Capture the last section
        if current_section == 'system' and current_content:
            system_prompt = '\n'.join(current_content).strip()
        elif current_section == 'user' and current_content:
            user_prompt = '\n'.join(current_content).strip()
        
        # If no sections found, treat entire content as user prompt
        if not system_prompt and not user_prompt:
            user_prompt = template_content
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        }


def combine_transcript_with_fragments(transcript_text: str, fragment: str) -> str:
    """
    Combine transcript text with a template fragment.
    
    Args:
        transcript_text: The transcript text to combine with the fragment
        fragment: Template fragment as a string
        
    Returns:
        Combined prompt as a string
    """
    # Add the fragment first, then the transcript with proper wrapping
    combined = f"{fragment}\n\n<transcript>\n{transcript_text}\n</transcript>"
    return combined
