"""Prompt management for LLM integration."""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from string import Template

logger = logging.getLogger(__name__)


class PromptManager:
    """Manager for LLM prompts and templates."""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """Initialize prompt manager.
        
        Args:
            templates_dir: Directory containing prompt templates
        """
        if templates_dir:
            self.templates_dir = Path(templates_dir)
        else:
            # Default to templates directory relative to this file
            self.templates_dir = Path(__file__).parent / "templates"
        
        # Ensure templates directory exists
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded templates
        self._template_cache = {}
        
        # Initialize built-in templates
        self._initialize_builtin_templates()
    
    def _initialize_builtin_templates(self):
        """Initialize built-in prompt templates."""
        # Contextual chunking template
        contextual_chunking_template = """<conversation_context>
$past_messages
$current_messages
</conversation_context>

Here is the message chunk we want to situate within the conversation flow
<message_chunk>
$chunk_content
</message_chunk>

Please give a short succinct context to situate this message chunk within the overall conversation for the purposes of improving search retrieval of the chunk. Consider the conversation flow, topics being discussed, and how this chunk relates to previous messages. Answer only with the succinct context and nothing else."""
        
        # Store built-in templates
        self._builtin_templates = {
            "contextual_chunking": contextual_chunking_template,
            "retrieval_enhancement": """Based on the following conversation context, generate a brief description that would help in retrieving this content later:

Context: $context
Current Content: $content

Description:""",
            "summarization": """Please provide a concise summary of the following content:

$content

Summary:"""
        }
    
    def get_prompt(self, template_name: str, **kwargs) -> str:
        """Get a formatted prompt from a template.
        
        Args:
            template_name: Name of the template
            **kwargs: Variables to substitute in the template
            
        Returns:
            Formatted prompt string
        """
        try:
            # Get template content
            template_content = self._get_template_content(template_name)
            
            # Create Template object and substitute variables
            template = Template(template_content)
            
            # Handle missing variables gracefully
            safe_kwargs = {}
            for key, value in kwargs.items():
                safe_kwargs[key] = str(value) if value is not None else ""
            
            return template.safe_substitute(**safe_kwargs)
            
        except Exception as e:
            logger.error(f"Error formatting prompt template '{template_name}': {e}")
            raise ValueError(f"Error formatting prompt template: {e}")
    
    def _get_template_content(self, template_name: str) -> str:
        """Get template content from file or built-in templates.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template content string
        """
        # Check cache first
        if template_name in self._template_cache:
            return self._template_cache[template_name]
        
        # Try to load from file
        template_file = self.templates_dir / f"{template_name}.txt"
        if template_file.exists():
            try:
                content = template_file.read_text(encoding="utf-8")
                self._template_cache[template_name] = content
                return content
            except Exception as e:
                logger.warning(f"Error reading template file '{template_file}': {e}")
        
        # Fall back to built-in templates
        if template_name in self._builtin_templates:
            content = self._builtin_templates[template_name]
            self._template_cache[template_name] = content
            return content
        
        # Template not found
        raise ValueError(f"Template '{template_name}' not found")
    
    def add_template(self, name: str, content: str, save_to_file: bool = False):
        """Add a new template.
        
        Args:
            name: Template name
            content: Template content
            save_to_file: Whether to save template to file
        """
        self._template_cache[name] = content
        
        if save_to_file:
            template_file = self.templates_dir / f"{name}.txt"
            try:
                template_file.write_text(content, encoding="utf-8")
                logger.info(f"Saved template '{name}' to {template_file}")
            except Exception as e:
                logger.error(f"Error saving template '{name}' to file: {e}")
    
    def list_templates(self) -> Dict[str, str]:
        """List all available templates.
        
        Returns:
            Dictionary mapping template names to their sources
        """
        templates = {}
        
        # Built-in templates
        for name in self._builtin_templates:
            templates[name] = "built-in"
        
        # File-based templates
        if self.templates_dir.exists():
            for template_file in self.templates_dir.glob("*.txt"):
                name = template_file.stem
                templates[name] = "file"
        
        return templates
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get information about a template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Dictionary with template information
        """
        try:
            content = self._get_template_content(template_name)
            
            # Extract variables from template
            template = Template(content)
            # This is a simple way to find template variables
            # More sophisticated parsing could be added if needed
            import re
            variables = re.findall(r'\$(\w+)', content)
            
            return {
                "name": template_name,
                "content_length": len(content),
                "variables": list(set(variables)),
                "source": "file" if (self.templates_dir / f"{template_name}.txt").exists() else "built-in"
            }
        except Exception as e:
            return {
                "name": template_name,
                "error": str(e)
            }
    
    def reload_templates(self):
        """Reload all templates from files."""
        self._template_cache.clear()
        logger.info("Template cache cleared, templates will be reloaded on next access")


# Global prompt manager instance
_global_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance."""
    global _global_prompt_manager
    if _global_prompt_manager is None:
        _global_prompt_manager = PromptManager()
    return _global_prompt_manager
