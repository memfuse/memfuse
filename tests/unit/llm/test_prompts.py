"""Unit tests for prompt management."""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.memfuse_core.llm.prompts.manager import PromptManager, get_prompt_manager


class TestPromptManager:
    """Test cases for PromptManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test templates
        self.temp_dir = tempfile.mkdtemp()
        self.templates_dir = Path(self.temp_dir) / "templates"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test prompt manager
        self.prompt_manager = PromptManager(str(self.templates_dir))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test PromptManager initialization."""
        assert self.prompt_manager.templates_dir == self.templates_dir
        assert self.prompt_manager._template_cache == {}
        assert "contextual_chunking" in self.prompt_manager._builtin_templates
    
    def test_get_builtin_prompt(self):
        """Test getting built-in prompt."""
        prompt = self.prompt_manager.get_prompt(
            "contextual_chunking",
            past_messages="Previous conversation",
            current_messages="Current message",
            chunk_content="Chunk to process"
        )
        
        assert "Previous conversation" in prompt
        assert "Current message" in prompt
        assert "Chunk to process" in prompt
        assert "<conversation_context>" in prompt
        assert "<message_chunk>" in prompt
    
    def test_get_prompt_with_missing_variables(self):
        """Test getting prompt with missing variables."""
        prompt = self.prompt_manager.get_prompt(
            "contextual_chunking",
            past_messages="Previous conversation"
            # Missing current_messages and chunk_content
        )
        
        # Should use safe_substitute, so missing variables become empty
        assert "Previous conversation" in prompt
        assert "$current_messages" in prompt  # Not substituted
        assert "$chunk_content" in prompt     # Not substituted
    
    def test_get_prompt_from_file(self):
        """Test getting prompt from file."""
        # Create test template file
        template_content = "Hello $name, welcome to $place!"
        template_file = self.templates_dir / "test_template.txt"
        template_file.write_text(template_content)
        
        prompt = self.prompt_manager.get_prompt(
            "test_template",
            name="Alice",
            place="MemFuse"
        )
        
        assert prompt == "Hello Alice, welcome to MemFuse!"
    
    def test_get_prompt_file_overrides_builtin(self):
        """Test that file templates override built-in templates."""
        # Create file with same name as built-in template
        custom_content = "Custom contextual chunking: $chunk_content"
        template_file = self.templates_dir / "contextual_chunking.txt"
        template_file.write_text(custom_content)
        
        prompt = self.prompt_manager.get_prompt(
            "contextual_chunking",
            chunk_content="test chunk"
        )
        
        assert prompt == "Custom contextual chunking: test chunk"
    
    def test_get_prompt_nonexistent(self):
        """Test getting non-existent prompt."""
        with pytest.raises(ValueError, match="Template 'nonexistent' not found"):
            self.prompt_manager.get_prompt("nonexistent")
    
    def test_add_template(self):
        """Test adding new template."""
        template_content = "Test template with $variable"
        self.prompt_manager.add_template("new_template", template_content)
        
        prompt = self.prompt_manager.get_prompt("new_template", variable="value")
        assert prompt == "Test template with value"
    
    def test_add_template_save_to_file(self):
        """Test adding template and saving to file."""
        template_content = "File template with $variable"
        self.prompt_manager.add_template(
            "file_template", 
            template_content, 
            save_to_file=True
        )
        
        # Check that file was created
        template_file = self.templates_dir / "file_template.txt"
        assert template_file.exists()
        assert template_file.read_text() == template_content
        
        # Check that template works
        prompt = self.prompt_manager.get_prompt("file_template", variable="test")
        assert prompt == "File template with test"
    
    def test_list_templates(self):
        """Test listing available templates."""
        # Add file template
        template_file = self.templates_dir / "file_template.txt"
        template_file.write_text("File template")
        
        # Add in-memory template
        self.prompt_manager.add_template("memory_template", "Memory template")
        
        templates = self.prompt_manager.list_templates()
        
        # Should include built-in templates
        assert "contextual_chunking" in templates
        assert templates["contextual_chunking"] == "built-in"
        
        # Should include file template
        assert "file_template" in templates
        assert templates["file_template"] == "file"
        
        # Memory template should not appear in list (only cached)
        # unless it was saved to file
    
    def test_get_template_info(self):
        """Test getting template information."""
        info = self.prompt_manager.get_template_info("contextual_chunking")
        
        assert info["name"] == "contextual_chunking"
        assert info["content_length"] > 0
        assert "past_messages" in info["variables"]
        assert "current_messages" in info["variables"]
        assert "chunk_content" in info["variables"]
        assert info["source"] == "built-in"
    
    def test_get_template_info_nonexistent(self):
        """Test getting info for non-existent template."""
        info = self.prompt_manager.get_template_info("nonexistent")
        
        assert info["name"] == "nonexistent"
        assert "error" in info
    
    def test_reload_templates(self):
        """Test reloading templates."""
        # Get a template to populate cache
        self.prompt_manager.get_prompt("contextual_chunking", chunk_content="test")
        assert "contextual_chunking" in self.prompt_manager._template_cache
        
        # Reload templates
        self.prompt_manager.reload_templates()
        assert self.prompt_manager._template_cache == {}
    
    def test_template_caching(self):
        """Test template caching behavior."""
        # First call should load and cache
        prompt1 = self.prompt_manager.get_prompt("contextual_chunking", chunk_content="test1")
        assert "contextual_chunking" in self.prompt_manager._template_cache
        
        # Second call should use cache
        prompt2 = self.prompt_manager.get_prompt("contextual_chunking", chunk_content="test2")
        
        # Content should be different due to different variables
        assert prompt1 != prompt2
        assert "test1" in prompt1
        assert "test2" in prompt2
    
    def test_error_handling_invalid_template_file(self):
        """Test error handling for invalid template file."""
        # Create a directory with the template name (should cause read error)
        invalid_template = self.templates_dir / "invalid_template.txt"
        invalid_template.mkdir()
        
        # Should fall back to built-in templates and raise error
        with pytest.raises(ValueError, match="Template 'invalid_template' not found"):
            self.prompt_manager.get_prompt("invalid_template")


class TestGlobalPromptManager:
    """Test cases for global prompt manager."""
    
    def test_get_prompt_manager_singleton(self):
        """Test that get_prompt_manager returns singleton."""
        manager1 = get_prompt_manager()
        manager2 = get_prompt_manager()
        
        assert manager1 is manager2
    
    def test_get_prompt_manager_functionality(self):
        """Test that global prompt manager works correctly."""
        manager = get_prompt_manager()
        
        prompt = manager.get_prompt(
            "contextual_chunking",
            past_messages="test",
            current_messages="test",
            chunk_content="test"
        )
        
        assert "test" in prompt
        assert "<conversation_context>" in prompt


class TestBuiltinTemplates:
    """Test cases for built-in templates."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.prompt_manager = PromptManager()
    
    def test_contextual_chunking_template(self):
        """Test contextual chunking template."""
        prompt = self.prompt_manager.get_prompt(
            "contextual_chunking",
            past_messages="User: Hello\nAssistant: Hi there",
            current_messages="User: How are you?",
            chunk_content="User: How are you?"
        )
        
        assert "User: Hello" in prompt
        assert "Assistant: Hi there" in prompt
        assert "User: How are you?" in prompt
        assert "conversation flow" in prompt
        assert "search retrieval" in prompt
    
    def test_retrieval_enhancement_template(self):
        """Test retrieval enhancement template."""
        prompt = self.prompt_manager.get_prompt(
            "retrieval_enhancement",
            context="Previous discussion about AI",
            content="Current message about machine learning"
        )
        
        assert "Previous discussion about AI" in prompt
        assert "Current message about machine learning" in prompt
        assert "Description:" in prompt
    
    def test_summarization_template(self):
        """Test summarization template."""
        prompt = self.prompt_manager.get_prompt(
            "summarization",
            content="This is a long piece of content that needs to be summarized for better understanding."
        )
        
        assert "This is a long piece of content" in prompt
        assert "Summary:" in prompt
