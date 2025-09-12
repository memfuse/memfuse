import argparse
import asyncio
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from datasets import load_dataset as hf_load_dataset


@dataclass
class DataLoadingConfig:
    """Configuration for dataset loading operations."""
    dataset_key: str
    num_samples: int = 0
    random_sampling: bool = False
    start_index: int = 0
    load_all: bool = False
    question_types: Optional[List[str]] = None
    question_ids: Optional[List[str]] = None
    
    @property
    def effective_num_samples(self) -> int:
        """Get the effective number of samples (0 if load_all is True)."""
        return 0 if self.load_all else self.num_samples


@dataclass
class LoadingResults:
    """Results from loading operations."""
    success_count: int
    total_count: int
    timing_stats: List[float]
    errors: List[str]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (self.success_count / self.total_count * 100) if self.total_count > 0 else 0.0
    
    @property
    def avg_time(self) -> float:
        """Calculate average loading time per item."""
        return sum(self.timing_stats) / len(self.timing_stats) if self.timing_stats else 0.0


def create_standard_parser(description: str, dataset_name: str, include_question_types: bool = False) -> argparse.ArgumentParser:
    """Create a standardized argument parser for benchmark scripts.
    
    Args:
        description: Parser description
        dataset_name: Name of the dataset (for help text)
        include_question_types: Whether to include --question-types argument
    """
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument(
        "--random",
        action="store_true",
        help="If set, randomly samples from the dataset. Otherwise, uses questions in a deterministic order."
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for deterministic sampling (default: 0). Ignored if --random is set."
    )
    parser.add_argument(
        "--num-questions", 
        type=int, 
        default=10,
        help="Number of questions to process (default: 10)"
    )
    parser.add_argument(
        "--load-all",
        action="store_true",
        help="Process the entire dataset (overrides --num-questions)"
    )
    
    if include_question_types:
        # LME-specific question types
        parser.add_argument(
            "--question-types",
            nargs="+",
            choices=["multi-session", "temporal-reasoning", "knowledge-update", 
                    "single-session-user", "single-session-assistant", "single-session-preference"],
            help="Specify which question types to include. If not specified, all types are included. "
                 "Available types: multi-session (133), temporal-reasoning (133), knowledge-update (78), "
                 "single-session-user (70), single-session-assistant (56), single-session-preference (30). "
                 "Works with both --load-all and --num-questions flags."
        )
    
    return parser


def args_to_config(args: argparse.Namespace, dataset_key: str, question_ids: Optional[List[str]] = None) -> DataLoadingConfig:
    """Convert argument namespace to DataLoadingConfig."""
    return DataLoadingConfig(
        dataset_key=dataset_key,
        num_samples=args.num_questions,
        random_sampling=args.random,
        start_index=args.start_index,
        load_all=args.load_all,
        question_types=getattr(args, 'question_types', None),
        question_ids=question_ids
    )


def load_dataset_from_huggingface(
        config: Dict[str, Any], 
        num_samples: int = 0, 
        random_sampling: bool = False, 
        start_index: int = 0,
        logger: Optional[logging.Logger] = None,
    ) -> List[Dict[str, Any]]:
    """Legacy function for backward compatibility."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    dataset_id = config["dataset_id"]
    data_file = config.get("data_file")
    logger.info(f"Loading dataset from HuggingFace ({dataset_id}, {data_file})...")
    try:
        if data_file:
            ds = hf_load_dataset(dataset_id, data_files=data_file, split="train")
        else:
            ds = hf_load_dataset(dataset_id, split="train")
        data = list(ds)
        logger.info(f"Successfully loaded {len(data)} samples.")
        
        # Apply sampling
        return _apply_sampling(data, num_samples, random_sampling, start_index, logger)
        
    except Exception as ex:
        logger.error(f"Error loading dataset from HuggingFace: {ex}")
        return []


def load_dataset_from_local_file(
        file_path: str,
        num_samples: int = 0, 
        random_sampling: bool = False, 
        start_index: int = 0,
        logger: Optional[logging.Logger] = None
    ) -> List[Dict[str, Any]]:
    """Load dataset from local JSON/JSONL file."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Loading dataset from local file: {file_path}")
    
    try:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Dataset file not found: {file_path}")
            return []
            
        with open(path, 'r', encoding='utf-8') as f:
            # Try JSON array format first
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.error("Expected dataset to be a list of questions.")
                    return []
                logger.info(f"Successfully loaded {len(data)} questions from JSON array format")
            except json.JSONDecodeError:
                # Try JSONL format
                logger.info("JSON array format failed, trying JSONL format...")
                f.seek(0)
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                            continue
                
                if not data:
                    logger.error("No valid JSON objects found in file")
                    return []
                
                logger.info(f"Successfully loaded {len(data)} questions from JSONL format")
        
        # Apply sampling
        return _apply_sampling(data, num_samples, random_sampling, start_index, logger)
        
    except Exception as e:
        logger.error(f"Unexpected error loading dataset: {e}")
        return []


def _apply_sampling(
        data: List[Dict[str, Any]], 
        num_samples: int, 
        random_sampling: bool, 
        start_index: int,
        logger: logging.Logger
    ) -> List[Dict[str, Any]]:
    """Apply sampling logic to dataset."""
    if num_samples > 0 and num_samples < len(data):
        if random_sampling:
            logger.info(f"Using random sampling: {num_samples} from {len(data)} total")
            data = random.sample(data, num_samples)
        else:
            logger.info(f"Using deterministic sampling (from index {start_index})")
            # Ensure start_index is within bounds
            if start_index >= len(data):
                logger.warning(f"Start index {start_index} exceeds dataset size {len(data)}. Using index 0.")
                start_index = 0
            # Handle wrapping around if needed
            if start_index + num_samples > len(data):
                logger.info(f"Requested {num_samples} samples from index {start_index}, but only {len(data) - start_index} samples are available. Wrapping around to the beginning.")
                end_samples = data[start_index:]
                remaining_samples = data[:num_samples - len(end_samples)]
                data = end_samples + remaining_samples
            else:
                data = data[start_index:start_index + num_samples]
    elif start_index > 0:
        logger.info(f"Starting from index {start_index}")
        data = data[start_index:]
    
    logger.info(f"Final dataset size: {len(data)} samples")
    return data


def apply_question_type_filter(
        dataset: List[Dict[str, Any]], 
        question_types: List[str],
        logger: Optional[logging.Logger] = None
    ) -> List[Dict[str, Any]]:
    """Filter dataset by question types."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Filtering questions by types: {', '.join(question_types)}")
    
    filtered_dataset = [
        item for item in dataset 
        if item.get('question_type') in question_types
    ]
    
    logger.info(f"Filtered dataset from {len(dataset)} to {len(filtered_dataset)} questions")
    
    # Log counts by question type
    type_counts = {}
    for item in filtered_dataset:
        qtype = item.get('question_type', 'unknown')
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    for qtype, count in sorted(type_counts.items()):
        logger.info(f"  {qtype}: {count} questions")
    
    return filtered_dataset


async def _call_memfuse(
        query: str,
        choices_length: int,
        model_name: str,
        memory_instance,
        llm_provider: str = "gemini",
        max_retries: int = 20,
        logger: Optional[logging.Logger] = None
    ):
    """Call MemFuse client and parse response."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    from tests.utils.data_models import MultipleChoiceResponse
    import json
    import re
    
    # Create system prompt with response format instructions
    system_prompt = f"""You are answering a multiple choice question. There are {choices_length} choices (0-indexed from 0 to {choices_length-1}). 

Please respond in JSON format with:
- "index": the 0-based index of your chosen answer (integer from 0 to {choices_length-1})
- "reasoning": brief explanation of your choice (string)

Example: {{"index": 2, "reasoning": "Option 2 is correct because..."}}"""

    # Initialize the appropriate client based on provider
    if llm_provider == "gemini":
        from memfuse.llm import AsyncGeminiClient
        from google.genai import types
        
        # Configure HTTP options for proxy if base URL is provided
        http_options = None
        if os.getenv("GEMINI_BASE_URL"):
            http_options = types.HttpOptions(
                base_url=os.getenv("GEMINI_BASE_URL")
            )
        
        client = AsyncGeminiClient(
            memory=memory_instance,
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=http_options
        )
    elif llm_provider == "openai":
        from memfuse.llm import AsyncOpenAI
        client = AsyncOpenAI(
            memory=memory_instance,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif llm_provider == "anthropic":
        from memfuse.llm import AsyncAnthropic
        client = AsyncAnthropic(
            memory=memory_instance,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                delay = 2 ** (attempt - 1)
                logger.info(f"Retrying MemFuse {llm_provider} call (attempt {attempt + 1}/{max_retries + 1}) after {delay}s delay...")
                await asyncio.sleep(delay)
            
            # Call the appropriate client with memory integration
            if llm_provider == "gemini":
                # Combine system and user prompts for Gemini
                combined_prompt = f"System: {system_prompt}\n\nUser: {query}"
                response = await client.models.generate_content_async(
                    model=model_name,
                    contents=combined_prompt
                )
            elif llm_provider == "openai":
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ]
                )
            elif llm_provider == "anthropic":
                response = await client.messages.create(
                    model=model_name,
                    system=system_prompt,
                    messages=[{"role": "user", "content": query}],
                    max_tokens=1000
                )
            
            # Extract response content based on provider
            response_content = ""
            if llm_provider == "gemini":
                if response and response.candidates:
                    if response.candidates[0].content and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if part.text:
                                response_content += part.text
            elif llm_provider == "openai":
                if response and response.choices and response.choices[0].message:
                    response_content = response.choices[0].message.content or ""
            elif llm_provider == "anthropic":
                if response and response.content:
                    for content_item in response.content:
                        if hasattr(content_item, 'text') and content_item.text:
                            response_content += content_item.text
            
            if not response_content:
                if attempt < max_retries:
                    logger.warning(f"Received empty response from MemFuse {llm_provider} client (attempt {attempt + 1}), retrying...")
                    continue
                else:
                    logger.error(f"Received empty response from MemFuse {llm_provider} client after all retries")
                    return MultipleChoiceResponse(
                        index=0,
                        reasoning="Empty response from API after all retries",
                        description="Error: Empty response",
                        retry_count=attempt,
                        failed_after_retries=True
                    )
            
            logger.debug(f"MemFuse {llm_provider} response: {response_content}")
            
            # Parse JSON response
            try:
                # Extract JSON from markdown code blocks if present
                json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
                match = re.search(json_pattern, response_content, re.DOTALL | re.IGNORECASE)
                if match:
                    cleaned_content = match.group(1).strip()
                else:
                    cleaned_content = response_content.strip()
                
                # Ensure it looks like JSON
                if not (cleaned_content.startswith('{') and cleaned_content.endswith('}')):
                    raise json.JSONDecodeError("Not valid JSON format", cleaned_content, 0)
                
                parsed_json = json.loads(cleaned_content)
                
                # Extract index and reasoning from various possible formats
                index = 0
                for key in ["index", "choice", "answer", "selected"]:
                    if key in parsed_json:
                        try:
                            index = int(parsed_json[key])
                            break
                        except (ValueError, TypeError):
                            continue
                
                reasoning = parsed_json.get("reasoning",
                           parsed_json.get("explanation", 
                           parsed_json.get("rationale", "No explanation provided")))
                
                # Validate choice index
                if not (0 <= index < choices_length):
                    logger.warning(f"Model returned choice index {index} which is out of range (0-{choices_length-1}). Defaulting to 0.")
                    index = 0
                
                # Capture retrieval debug info after successful LLM call
                retrieval_debug = get_retrieval_debug_info(llm_provider)
                retrieval_info = ""
                if retrieval_debug:
                    results = retrieval_debug.get("data", {}).get("results", [])
                    retrieval_info = f" | Retrieved: {len(results)} memories"
                    logger.debug(f"Retrieval debug info: {len(results)} memories retrieved")
                
                return MultipleChoiceResponse(
                    index=index,
                    reasoning=str(reasoning),
                    description=f"Response from MemFuse {llm_provider} client{retrieval_info}",
                    retry_count=attempt,
                    failed_after_retries=False
                )
                
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON response: {e}. Attempting text parsing.")
                # Fallback to text parsing
                numbers = re.findall(r'\b\d+\b', response_content)
                index = 0
                if numbers:
                    try:
                        index = int(numbers[0])
                        if not (0 <= index < choices_length):
                            index = 0
                    except ValueError:
                        index = 0
                
                return MultipleChoiceResponse(
                    index=index,
                    reasoning=response_content[:200] + "..." if len(response_content) > 200 else response_content,
                    description=f"Fallback text parsing from MemFuse {llm_provider}",
                    retry_count=attempt,
                    failed_after_retries=False
                )
        
        except Exception as e:
            if attempt < max_retries:
                delay = 2 ** attempt
                logger.warning(f"Error calling MemFuse {llm_provider} client (attempt {attempt + 1}): {e}, retrying in {delay}s...")
                await asyncio.sleep(delay)
                continue
            else:
                logger.error(f"Error calling MemFuse {llm_provider} client after all retries: {e}")
                return MultipleChoiceResponse(
                    index=0,
                    reasoning=f"Error calling MemFuse {llm_provider} client after {max_retries + 1} attempts: {str(e)}",
                    description="Error response",
                    retry_count=max_retries,
                    failed_after_retries=True
                )


def load_benchmark_dataset(
        config: DataLoadingConfig,
        logger: Optional[logging.Logger] = None
    ) -> List[Dict[str, Any]]:
    """Load benchmark dataset with unified interface.
    
    This function handles all dataset loading scenarios:
    - HuggingFace datasets (msc, lme)
    - Local file datasets (locomo)
    - Question type filtering (lme)
    - All sampling options
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Import here to avoid circular imports
    try:
        from .config import DATASET_CONFIGS
    except ImportError:
        logger.error("Cannot import DATASET_CONFIGS. Ensure benchmarks.config is available.")
        return []
    
    if config.dataset_key not in DATASET_CONFIGS:
        logger.error(f"Unknown dataset: {config.dataset_key}")
        return []
    
    dataset_config = DATASET_CONFIGS[config.dataset_key]
    
    # Log loading info
    if config.load_all:
        logger.info(f"Loading {config.dataset_key.upper()} dataset (ALL samples)...")
    else:
        logger.info(f"Loading {config.dataset_key.upper()} dataset ({config.num_samples} samples)...")
    
    if config.random_sampling:
        logger.info("Using random sampling from dataset.")
    else:
        logger.info(f"Using deterministic sampling starting from index {config.start_index}.")
    
    # Load dataset based on type
    if dataset_config["dataset_id"] == "local":
        # Local file loading (LoCoMo)
        dataset = load_dataset_from_local_file(
            dataset_config["data_file"],
            config.effective_num_samples,
            config.random_sampling,
            config.start_index,
            logger
        )
    else:
        # HuggingFace loading (MSC, LME)
        if config.question_types:
            # Load full dataset first for filtering
            full_dataset = load_dataset_from_huggingface(
                dataset_config,
                num_samples=0,  # Load all for filtering
                random_sampling=False,
                start_index=0,
                logger=logger
            )
            
            if not full_dataset:
                return []
            
            # Apply question type filter
            filtered_dataset = apply_question_type_filter(full_dataset, config.question_types, logger)
            
            if not filtered_dataset:
                logger.error("No questions match the specified question types.")
                return []
            
            # Apply sampling to filtered dataset
            if config.load_all:
                dataset = filtered_dataset
            else:
                if config.random_sampling:
                    if len(filtered_dataset) <= config.num_samples:
                        dataset = filtered_dataset
                    else:
                        dataset = random.sample(filtered_dataset, config.num_samples)
                else:
                    start_idx = config.start_index
                    end_idx = start_idx + config.num_samples
                    dataset = filtered_dataset[start_idx:end_idx]
        else:
            # Direct loading without filtering
            dataset = load_dataset_from_huggingface(
                dataset_config,
                config.effective_num_samples,
                config.random_sampling,
                config.start_index,
                logger
            )
    
    if not dataset:
        logger.error("Failed to load dataset.")
        return []
    
    # Apply question ID filtering if specified
    if config.question_ids:
        original_count = len(dataset)
        dataset = [item for item in dataset if item.get('question_id') in config.question_ids]
        filtered_count = len(dataset)
        
        if filtered_count == 0:
            logger.error("No questions match the provided question IDs.")
            return []
        
        logger.info(f"Filtered dataset by question IDs: {filtered_count}/{original_count} questions matched")
        
        # Log which question IDs were not found
        found_ids = {item.get('question_id') for item in dataset}
        missing_ids = set(config.question_ids) - found_ids
        if missing_ids:
            logger.warning(f"Question IDs not found in dataset: {', '.join(sorted(missing_ids))}")
    
    actual_count = len(dataset)
    requested_count = config.effective_num_samples
    
    if requested_count > 0 and actual_count < requested_count:
        logger.warning(f"Only {actual_count} questions available, less than requested {requested_count}.")
    
    logger.info(f"Successfully loaded {actual_count} questions")
    return dataset


def convert_messages_for_memfuse(messages: List[Dict[str, Any]], dataset_type: str) -> List[Dict[str, Any]]:
    """Convert dataset-specific message format to MemFuse format.
    
    Args:
        messages: Original messages from dataset
        dataset_type: Type of dataset (msc, lme, locomo)
    
    Returns:
        Messages in MemFuse format
    """
    if dataset_type == "locomo":
        # LoCoMo format: [{"role": "user", "name": "Speaker", "content": "text"}, ...]
        # MemFuse format: [{"role": "user", "content": "text"}, ...]
        # LoCoMo messages already have speaker names embedded in content (e.g., "[CAROLINE]: text")
        memfuse_messages = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            
            # Skip messages with empty content or role
            if not content or not content.strip() or not role or not role.strip():
                continue
                
            memfuse_msg = {
                "role": role,
                "content": content
            }
            memfuse_messages.append(memfuse_msg)
        return memfuse_messages
    else:
        # MSC and LME need to filter out extra fields and empty messages to match MemFuse format
        memfuse_messages = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            
            # Skip messages with empty content or role
            if not content or not content.strip() or not role or not role.strip():
                continue
                
            memfuse_msg = {
                "role": role,
                "content": content
            }
            memfuse_messages.append(memfuse_msg)
        return memfuse_messages


async def load_dataset_to_memfuse(
        dataset: List[Dict[str, Any]],
        dataset_type: str,
        logger: Optional[logging.Logger] = None,
        start_id: int = 1,
        end_id: Optional[int] = None
    ) -> LoadingResults:
    """Load dataset into MemFuse memory.

    Args:
        dataset: List of questions with haystack sessions
        dataset_type: Type of dataset (msc, lme, locomo)
        logger: Logger instance
        start_id: Starting question ID (1-based) for loading
        end_id: Ending question ID (1-based, inclusive) for loading

    Returns:
        LoadingResults with success statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    from memfuse import AsyncMemFuse
    successfully_loaded_count = 0
    question_times = []
    errors = []

    # Use async context manager to ensure proper resource cleanup
    async with AsyncMemFuse() as memfuse_client:
        # Calculate actual loading range
        if end_id is not None:
            if start_id == 1 and end_id == len(dataset):
                logger.info(f"Starting to load all {len(dataset)} questions into MemFuse...")
            else:
                load_count = end_id - start_id + 1
                logger.info(f"Starting to load {load_count} questions into MemFuse "
                            f"(questions {start_id} to {end_id})...")
        elif start_id > 1:
            skipped_count = start_id - 1
            remaining_count = len(dataset) - skipped_count
            logger.info(f"Starting to load {remaining_count} questions into MemFuse "
                        f"(skipping first {skipped_count})...")
        else:
            logger.info(f"Starting to load {len(dataset)} questions into MemFuse...")

        for i, data_sample in enumerate(dataset, start=1):
            # Skip questions before start_id
            if i < start_id:
                continue

            # Stop if we've reached end_id
            if end_id is not None and i > end_id:
                break

            question_start_time = time.time()
            question_number = i
            logger.info(f"--- Loading Question {question_number}/{len(dataset)} ---")

            question_id = data_sample.get('question_id', f"q_{uuid.uuid4().hex[:8]}")
            question_text = data_sample.get('question')
            haystack_session_ids = data_sample.get('haystack_session_ids', [])
            haystack_sessions_data = data_sample.get('haystack_sessions', [])

            if not all([question_text, haystack_session_ids, haystack_sessions_data]):
                error_msg = f"Question {question_number} (ID: {question_id}): Missing required fields. Skipping."
                logger.error(error_msg)
                errors.append(error_msg)
                continue

            logger.info(f"Question {question_number} (ID: {question_id}): {question_text}")

            user_name_for_test = question_id
            agent_name_for_test = "agent_default"

            try:
                logger.info(f"Adding {len(haystack_sessions_data)} sessions to MemFuse for user '{user_name_for_test}' (Question {question_number})...")

                for _, (session_id, messages_for_session) in enumerate(zip(haystack_session_ids, haystack_sessions_data)):
                    if not messages_for_session:
                        logger.warning(f"Skipping session '{session_id}' for Q{question_number} as it has no messages.")
                        continue

                    logger.info(f"Initializing session: '{session_id}' for Q{question_number}")
                    # Create memory instance (no context manager needed for Memory instances)
                    memory_instance = await memfuse_client.init(
                        user=user_name_for_test,
                        agent=agent_name_for_test,
                        session=session_id
                    )
                    
                    # Convert messages to MemFuse format if needed
                    memfuse_messages = convert_messages_for_memfuse(messages_for_session, dataset_type)

                    logger.info(f"Adding {len(memfuse_messages)} messages to session '{memory_instance.session}' (ID: {memory_instance.session_id}) for Q{question_number}")
                    add_result = await memory_instance.add(memfuse_messages)

                    if add_result.get("status") == "success":
                        logger.info(f"Successfully added messages for Q{question_number}: {add_result.get('data', {}).get('message_ids', [])}")
                    else:
                        error_msg = f"Failed to add messages to session '{session_id}' for Q{question_number}: {add_result}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        continue

                successfully_loaded_count += 1
                question_elapsed_time = time.time() - question_start_time
                question_times.append(question_elapsed_time)
                logger.info(f"⏱️  Question {question_number} loaded in {question_elapsed_time:.2f} seconds")
                logger.info(f"--- Successfully loaded Question {question_number}/{len(dataset)} ---")

            except ConnectionError as e:
                error_msg = f"Connection error with MemFuse server for Q{question_number} (ID: {question_id}): {e}"
                logger.error(error_msg)
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error for Q{question_number} (ID: {question_id}): {e}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
    
    results = LoadingResults(
        success_count=successfully_loaded_count,
        total_count=len(dataset),
        timing_stats=question_times,
        errors=errors
    )
    
    # Log summary
    avg_time = results.avg_time
    logger.info(f"--- LOADING SUMMARY ---")
    logger.info(f"Successfully loaded: {results.success_count}/{results.total_count} questions")
    logger.info(f"Success rate: {results.success_rate:.1f}%")
    logger.info(f"⏱️  Average loading time per question: {avg_time:.2f} seconds")
    logger.info(f"⏱️  Total data loading time: {sum(question_times):.2f} seconds")
    if errors:
        logger.warning(f"Encountered {len(errors)} errors during loading")
        logger.warning("Failed question details:")
        for error in errors:
            logger.warning(f"  - {error}")
    logger.info("Data has been loaded into MemFuse and is ready for querying.")
    
    return results


@dataclass
class BenchmarkResults:
    """Results from benchmark execution."""
    question_results: List[Dict[str, Any]]
    query_times: List[float]
    success_count: int
    total_count: int
    accuracy: float
    total_elapsed_time: float
    total_retries: int = 0
    final_failures: int = 0
    # Retrieval evaluation metrics (for LME dataset)
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_f1: float = 0.0
    retrieval_metrics_available: bool = False
    
    @property
    def success_rate(self) -> float:
        return (self.success_count / self.total_count * 100) if self.total_count > 0 else 0.0


async def _evaluate_single_question(
        semaphore: asyncio.Semaphore,
        memfuse_client,
        data_sample: Dict[str, Any],
        question_number: int,
        total_questions: int,
        top_k: int,
        model_name: str,
        llm_provider: str,
        logger: logging.Logger
    ) -> Dict[str, Any]:
    """Evaluate a single question with concurrency control."""
    async with semaphore:
        logger.info(f"--- Processing Question {question_number}/{total_questions} ---")

        question_id = data_sample.get('question_id', f"q_{uuid.uuid4().hex[:8]}")
        question_text = data_sample.get('question')
        choices = data_sample.get('choices')
        correct_choice_index = data_sample.get('correct_choice_index')

        if not all([question_text, choices, correct_choice_index is not None]):
            logger.error(f"Question {question_number} (ID: {question_id}): Missing required fields for evaluation. Skipping.")
            return {
                "question_id": question_id,
                "question_text": question_text,
                "status": "SKIPPED - Missing data",
                "query_time": 0.0
            }

        logger.info(f"Question {question_number} (ID: {question_id}): {question_text}")

        user_name_for_test = question_id
        agent_name_for_test = "agent_default"

        try:
            # Create memory instance for the current question
            logger.info(f"Initializing MemFuse for Q{question_number}...")
            query_memory_instance = await memfuse_client.init(
                user=user_name_for_test,
                agent=agent_name_for_test,
            )

            # Create the multiple choice question format
            formatted_question = f"{question_text}\n\nChoices:\n"
            for i, choice in enumerate(choices):
                formatted_question += f"{i}. {choice}\n"

            # Call LLM with MemFuse integration (SDK handles memory retrieval automatically)
            logger.info(f"Calling MemFuse {llm_provider} for Q{question_number}...")
            llm_response_model = await _call_memfuse(
                query=formatted_question,
                choices_length=len(choices),
                model_name=model_name,
                memory_instance=query_memory_instance,
                llm_provider=llm_provider,
                max_retries=20,
                logger=logger
            )

            model_choice_idx = llm_response_model.index
            model_explanation = llm_response_model.reasoning
            is_correct = (model_choice_idx == correct_choice_index)

            # Capture retrieval debug info and timing from the SDK
            retrieval_debug = get_retrieval_debug_info(llm_provider)
            retrieval_timing = get_retrieval_timing_info(llm_provider)

            retrieved_memories_count = 0
            query_duration = retrieval_timing or 0.0  # Default to 0 if timing not available

            retrieved_memories_summary = None
            if retrieval_debug:
                results = retrieval_debug.get("data", {}).get("results", [])
                retrieved_memories_count = len(results)
                logger.info(f"MemFuse query for Q{question_number} took {query_duration * 1000:.2f} ms.")
                # Create a summary for logging failed questions
                if not is_correct and results:
                    retrieved_memories_summary = []
                    for result in results[:3]:  # Show first 3 for brevity
                        content = result.get("content", "")[:100]  # Truncate long content
                        score = result.get("score", 0)
                        retrieved_memories_summary.append(f"Score:{score:.3f} Content:{content}...")
                    logger.info(f"Q{question_number} FAILED - Retrieved {retrieved_memories_count} memories: {retrieved_memories_summary}")
                elif not is_correct:
                    logger.info(f"Q{question_number} FAILED - No memories retrieved from RAG")
            else:
                logger.warning(f"No retrieval debug info available for Q{question_number}")

            logger.info(f"--- Q{question_number} RESULT ---")
            logger.info(f"Question: {question_text}")
            logger.info(f"LLM's Choice: {model_choice_idx} ('{choices[model_choice_idx] if 0 <= model_choice_idx < len(choices) else 'Invalid Index'}')")
            logger.info(f"Correct Choice: {correct_choice_index} ('{choices[correct_choice_index]}')")
            logger.info(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
            logger.info(f"LLM's Explanation: {model_explanation}")

            return {
                "question_id": question_id,
                "question_text": question_text,
                "model_choice_idx": model_choice_idx,
                "model_choice_text": choices[model_choice_idx] if 0 <= model_choice_idx < len(choices) else 'Invalid Index',
                "correct_choice_idx": correct_choice_index,
                "correct_choice_text": choices[correct_choice_index],
                "is_correct": is_correct,
                "explanation": model_explanation,
                "top_k": top_k,
                "retrieval_time_ms": query_duration * 1000,
                "retry_count": llm_response_model.retry_count,
                "failed_after_retries": llm_response_model.failed_after_retries,
                "retrieved_memories_count": retrieved_memories_count,
                "retrieval_debug_available": retrieval_debug is not None,
                "query_time": query_duration,
                "memory_instance": query_memory_instance
            }

        except ConnectionError as e:
            error_msg = f"Connection error with MemFuse server for Q{question_number} (ID: {question_id}): {e}"
            logger.error(error_msg)
            return {
                "question_id": question_id,
                "question_text": question_text,
                "status": f"FAILED - ConnectionError: {e}",
                "query_time": 0.0
            }
        except Exception as e:
            error_msg = f"Unexpected error for Q{question_number} (ID: {question_id}): {e}"
            logger.error(error_msg, exc_info=True)
            return {
                "question_id": question_id,
                "question_text": question_text,
                "status": f"FAILED - Exception: {e}",
                "query_time": 0.0
            }


async def run_benchmark_evaluation(
        dataset: List[Dict[str, Any]],
        dataset_name: str,
        top_k: int = 3,
        model_name: str = "deepseek-ai/DeepSeek-V3.1",
        llm_provider: str = "gemini",
        concurrent: int = 1,
        logger: Optional[logging.Logger] = None
    ) -> BenchmarkResults:
    """Run benchmark evaluation on a dataset.

    Args:
        dataset: List of questions with choices and correct answers
        dataset_name: Name of dataset for recording (msc, lme, locomo)
        top_k: Number of top results to retrieve from memory
        model_name: LLM model to use for evaluation
        llm_provider: LLM provider to use ("gemini", "openai", "anthropic")
        concurrent: Number of concurrent evaluations to run (default: 1)
        logger: Logger instance

    Returns:
        BenchmarkResults with evaluation statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Import here to avoid circular imports
    from memfuse import AsyncMemFuse
    from benchmarks.recorder import BenchmarkRecorder
    
    logger.info(f"Starting benchmark evaluation for {len(dataset)} questions...")
    logger.info(f"Using {llm_provider} model: {model_name}, TOP_K: {top_k}, "
                f"Concurrent: {concurrent}")

    # Initialize recorder
    recorder = BenchmarkRecorder(dataset_name=dataset_name, top_k=top_k)

    # Track total elapsed time
    total_start_time = time.perf_counter()

    # Use async context manager for proper resource cleanup
    async with AsyncMemFuse() as memfuse_client:
        # Create semaphore to control concurrency
        semaphore = asyncio.Semaphore(concurrent)

        # Create tasks for concurrent evaluation
        tasks = []
        for i, data_sample in enumerate(dataset):
            question_number = i + 1
            task = _evaluate_single_question(
                semaphore=semaphore,
                memfuse_client=memfuse_client,
                data_sample=data_sample,
                question_number=question_number,
                total_questions=len(dataset),
                top_k=top_k,
                model_name=model_name,
                llm_provider=llm_provider,
                logger=logger
            )
            tasks.append(task)

        # Execute all tasks concurrently
        logger.info(f"Executing {len(tasks)} evaluation tasks with "
                    f"concurrency limit of {concurrent}...")

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        results_summary = []
        all_query_times = []
        all_created_mem_instances = []
        total_retries = 0
        final_failures = 0

        for i, result in enumerate(results_list):
            if isinstance(result, Exception):
                logger.error(f"Task {i+1} failed with exception: {result}")
                results_summary.append({
                    "question_id": f"q_{i+1}_failed",
                    "question_text": "Task failed",
                    "status": f"FAILED - Exception: {result}"
                })
            elif isinstance(result, dict):
                results_summary.append(result)

                # Collect query times and memory instances
                if ("query_time" in result and
                        isinstance(result["query_time"], (int, float))):
                    all_query_times.append(result["query_time"])

                if "memory_instance" in result:
                    all_created_mem_instances.append(result["memory_instance"])

                # Collect retry statistics
                if ("retry_count" in result and
                        isinstance(result["retry_count"], int)):
                    total_retries += result["retry_count"]
                if result.get("failed_after_retries", False):
                    final_failures += 1
        # Calculate total elapsed time
        total_end_time = time.perf_counter()
        total_elapsed_time = total_end_time - total_start_time
        
        # Save raw results
        recorder.record_raw_results(results_summary)
        
        # Calculate and save summary
        valid_results = [item for item in results_summary if 'is_correct' in item]
        correct_count = sum(1 for item in valid_results if item.get('is_correct'))
        accuracy = (correct_count / len(valid_results)) * 100 if len(valid_results) > 0 else 0
        # Convert query times to milliseconds to match the expected format for summary
        all_query_times_ms = [t * 1000 for t in all_query_times]
        recorder.record_summary(all_query_times_ms, accuracy)
        
        # Display the overall summary with percentiles
        if all_query_times:
            p50_time = np.percentile(all_query_times, 50)
            p90_time = np.percentile(all_query_times, 90)
            p95_time = np.percentile(all_query_times, 95)
            
            logger.info(f"P50 Query Time: {p50_time * 1000:.2f}ms, P90 Query Time: {p90_time * 1000:.2f}ms, P95 Query Time: {p95_time * 1000:.2f}ms")
        
        # Log retry statistics
        logger.info(f"Total retries: {total_retries}")
        logger.info(f"Final failures after all retries: {final_failures}")
        
        # Calculate average retrieval metrics for LME and MSC datasets
        avg_precision = avg_recall = avg_f1 = 0.0
        retrieval_metrics_available = False
        
        if dataset_name in ["lme", "msc"]:
            # Extract retrieval metrics from valid results that have them
            results_with_metrics = [
                item for item in valid_results 
                if 'precision' in item and 'recall' in item and 'f1' in item
            ]
            
            if results_with_metrics:
                avg_precision = sum(item['precision'] for item in results_with_metrics) / len(results_with_metrics)
                avg_recall = sum(item['recall'] for item in results_with_metrics) / len(results_with_metrics)
                avg_f1 = sum(item['f1'] for item in results_with_metrics) / len(results_with_metrics)
                retrieval_metrics_available = True
                
                logger.info(f"=== RETRIEVAL EVALUATION METRICS ({dataset_name.upper()}) ===")
                logger.info(f"Questions with retrieval metrics: {len(results_with_metrics)}/{len(valid_results)}")
                logger.info(f"Average Precision: {avg_precision:.3f}")
                logger.info(f"Average Recall: {avg_recall:.3f}")
                logger.info(f"Average F1: {avg_f1:.3f}")
                logger.info(f"========================================")
        
        # Note: Summary printing is now handled in the main scripts
        
        return BenchmarkResults(
            question_results=results_summary,
            query_times=all_query_times,
            success_count=len(valid_results),
            total_count=len(dataset),
            accuracy=accuracy,
            total_elapsed_time=total_elapsed_time,
            total_retries=total_retries,
            final_failures=final_failures,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1=avg_f1,
            retrieval_metrics_available=retrieval_metrics_available
        )


def get_retrieval_debug_info(llm_provider: str = "gemini") -> Optional[Dict[str, Any]]:
    """
    Get retrieval debug information from the last MemFuse query.
    
    This function provides access to the intermediate query response from
    memory.query_session() that occurs within the MemFuse LLM wrappers.
    Useful for debugging whether retrieval is working correctly.
    
    Args:
        llm_provider: The LLM provider to get debug info from ("gemini", "openai", "anthropic")
    
    Returns:
        Dict containing query response with retrieval results, or None if not available.
        The dict typically contains:
        - "data": {"results": [...]} - the retrieved memories
        - Other metadata from the query response
    
    Usage:
        # After calling a MemFuse-enabled LLM
        debug_info = get_retrieval_debug_info("openai")
        if debug_info:
            results = debug_info.get("data", {}).get("results", [])
            print(f"Retrieved {len(results)} memories")
    """
    try:
        if llm_provider == "gemini":
            from memfuse.llm.gemini_adapter import get_last_query_response
            return get_last_query_response()
        elif llm_provider == "openai":
            from memfuse.llm.openai_adapter import get_last_query_response
            return get_last_query_response()
        elif llm_provider == "anthropic":
            from memfuse.llm.anthropic_adapter import get_last_query_response
            return get_last_query_response()
        else:
            return None
    except ImportError:
        # In case the adapter is not available
        return None


def get_retrieval_timing_info(llm_provider: str = "gemini") -> Optional[float]:
    """
    Get retrieval timing information from the last MemFuse query.
    
    Args:
        llm_provider: The LLM provider to get timing info from ("gemini", "openai", "anthropic")
    
    Returns:
        Query duration in seconds if available, None otherwise.
    """
    try:
        if llm_provider == "gemini":
            from memfuse.llm.gemini_adapter import get_last_query_time
            return get_last_query_time()
        elif llm_provider == "openai":
            from memfuse.llm.openai_adapter import get_last_query_time
            return get_last_query_time()
        elif llm_provider == "anthropic":
            from memfuse.llm.anthropic_adapter import get_last_query_time
            return get_last_query_time()
        else:
            return None
    except ImportError:
        # In case the adapter is not available
        return None


def extract_answer_containing_messages(haystack_sessions: List[List[Dict[str, Any]]]) -> List[str]:
    """
    Extract content from messages that contain answers (has_answer=True) from haystack sessions.
    
    Args:
        haystack_sessions: List of sessions, each containing a list of messages
        
    Returns:
        List of message contents that have has_answer=True
    """
    answer_messages = []
    
    for session in haystack_sessions:
        for message in session:
            if message.get("has_answer") is True:
                content = message.get("content", "").strip()
                if content:
                    answer_messages.append(content)
    
    return answer_messages


def find_answer_containing_content(haystack_sessions: List[List[Dict[str, Any]]], answer_text: str) -> Tuple[List[str], List[str]]:
    """
    Find all content that contains the answer, both flagged messages and substring matches.
    
    Args:
        haystack_sessions: List of sessions, each containing a list of messages
        answer_text: The correct answer text to search for
        
    Returns:
        Tuple of (flagged_messages, substring_messages)
        - flagged_messages: Content from messages with has_answer=True
        - substring_messages: Content from other messages containing answer substring
    """
    import re
    
    flagged_messages = []
    substring_messages = []
    
    # Normalize answer text: lowercase, strip, remove punctuation for better matching
    answer_text_normalized = re.sub(r'[^\w\s]', '', answer_text.lower().strip())
    
    for session in haystack_sessions:
        for message in session:
            content = message.get("content", "").strip()
            if not content:
                continue
                
            content_lower = content.lower()
            content_normalized = re.sub(r'[^\w\s]', '', content_lower)
            
            if message.get("has_answer") is True:
                flagged_messages.append(content)
            elif answer_text_normalized in content_normalized:
                substring_messages.append(content)
    
    return flagged_messages, substring_messages


def calculate_enhanced_retrieval_metrics(
    answer_text: str,
    haystack_sessions: List[List[Dict[str, Any]]],
    retrieved_memories: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Calculate enhanced precision, recall, and F1 for memory retrieval using both flagged 
    messages and substring matching for more realistic evaluation.
    
    Args:
        answer_text: The correct answer text
        haystack_sessions: All haystack sessions for the question  
        retrieved_memories: List of memory objects from get_retrieval_debug_info()
        logger: Logger instance
        
    Returns:
        Dict with precision, recall, f1, and detailed debug metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not haystack_sessions:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "flagged_messages_count": 0,
            "substring_messages_count": 0,
            "total_answer_content_count": 0,
            "retrieved_memories_count": len(retrieved_memories),
            "flagged_hits": 0,
            "substring_hits": 0,
            "total_hits": 0
        }
    
    if not retrieved_memories:
        flagged_msgs, substring_msgs = find_answer_containing_content(haystack_sessions, answer_text)
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "flagged_messages_count": len(flagged_msgs),
            "substring_messages_count": len(substring_msgs),
            "total_answer_content_count": len(flagged_msgs) + len(substring_msgs),
            "retrieved_memories_count": 0,
            "flagged_hits": 0,
            "substring_hits": 0,
            "total_hits": 0
        }
    
    # Find all content containing the answer
    flagged_msgs, substring_msgs = find_answer_containing_content(haystack_sessions, answer_text)
    total_answer_content = len(flagged_msgs) + len(substring_msgs)
    
    if total_answer_content == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "flagged_messages_count": 0,
            "substring_messages_count": 0,
            "total_answer_content_count": 0,
            "retrieved_memories_count": len(retrieved_memories),
            "flagged_hits": 0,
            "substring_hits": 0,
            "total_hits": 0
        }
    
    # Check retrieved memories for matches
    flagged_hits = 0
    substring_hits = 0
    import re
    answer_text_lower = answer_text.lower().strip()
    answer_text_normalized = re.sub(r'[^\w\s]', '', answer_text_lower)
    
    for memory in retrieved_memories:
        memory_content = memory.get("content", "").strip()
        if not memory_content:
            continue
            
        memory_content_lower = memory_content.lower()
        memory_content_normalized = re.sub(r'[^\w\s]', '', memory_content_lower)
        
        # Check against flagged messages (bidirectional substring match)
        flagged_match = False
        for flagged_msg in flagged_msgs:
            flagged_msg_lower = flagged_msg.lower().strip()
            flagged_msg_normalized = re.sub(r'[^\w\s]', '', flagged_msg_lower)
            if (flagged_msg_normalized in memory_content_normalized or 
                memory_content_normalized in flagged_msg_normalized):
                flagged_hits += 1
                flagged_match = True
                break
        
        # If no flagged match, check for answer substring (avoid double counting)
        if not flagged_match and answer_text_normalized in memory_content_normalized:
            substring_hits += 1
    
    total_hits = flagged_hits + substring_hits
    
    # Calculate metrics
    recall = total_hits / total_answer_content if total_answer_content > 0 else 0.0
    precision = total_hits / len(retrieved_memories) if retrieved_memories else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flagged_messages_count": len(flagged_msgs),
        "substring_messages_count": len(substring_msgs),
        "total_answer_content_count": total_answer_content,
        "retrieved_memories_count": len(retrieved_memories),
        "flagged_hits": flagged_hits,
        "substring_hits": substring_hits,
        "total_hits": total_hits
    }
    
    logger.debug(f"Enhanced retrieval metrics: {metrics}")
    return metrics


def calculate_retrieval_metrics(
    answer_messages: List[str], 
    retrieved_memories: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 for memory retrieval by matching answer-containing 
    messages with retrieved memories using substring search.
    
    Args:
        answer_messages: List of message contents that contain answers (has_answer=True)
        retrieved_memories: List of memory objects from get_retrieval_debug_info()
        logger: Logger instance
        
    Returns:
        Dict with precision, recall, f1, and additional debug metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not answer_messages:
        # No answer messages to match
        return {
            "precision": 0.0,
            "recall": 0.0, 
            "f1": 0.0,
            "answer_messages_count": 0,
            "retrieved_memories_count": len(retrieved_memories),
            "matched_answer_messages": 0,
            "matched_retrieved_memories": 0
        }
    
    if not retrieved_memories:
        # No memories retrieved
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0, 
            "answer_messages_count": len(answer_messages),
            "retrieved_memories_count": 0,
            "matched_answer_messages": 0,
            "matched_retrieved_memories": 0
        }
    
    # Extract content from retrieved memories
    retrieved_contents = []
    for memory in retrieved_memories:
        content = memory.get("content", "").strip()
        if content:
            retrieved_contents.append(content)
    
    # Find matches using substring search
    matched_answer_messages = 0
    matched_memory_indices = set()
    
    for answer_content in answer_messages:
        answer_found = False
        # Normalize answer content for better matching (lowercase, strip)
        answer_normalized = answer_content.lower().strip()
        
        for i, memory_content in enumerate(retrieved_contents):
            memory_normalized = memory_content.lower().strip()
            
            # Check if answer content appears as substring in memory content
            # or vice versa (in case memory content is truncated)
            if (answer_normalized in memory_normalized or 
                memory_normalized in answer_normalized):
                if not answer_found:
                    matched_answer_messages += 1
                    answer_found = True
                matched_memory_indices.add(i)
    
    matched_retrieved_memories = len(matched_memory_indices)
    
    # Calculate metrics
    recall = matched_answer_messages / len(answer_messages) if answer_messages else 0.0
    precision = matched_retrieved_memories / len(retrieved_contents) if retrieved_contents else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "answer_messages_count": len(answer_messages),
        "retrieved_memories_count": len(retrieved_contents),
        "matched_answer_messages": matched_answer_messages,
        "matched_retrieved_memories": matched_retrieved_memories
    }
    
    logger.debug(f"Retrieval metrics: {metrics}")
    return metrics


async def _evaluate_single_question_with_data_loading(
        semaphore: asyncio.Semaphore,
        memfuse_client,
        data_sample: Dict[str, Any],
        question_number: int,
        total_questions: int,
        dataset_name: str,
        top_k: int,
        model_name: str,
        llm_provider: str,
        skip_data_loading: bool,
        retrieval_verbose: bool,
        logger: logging.Logger
    ) -> Dict[str, Any]:
    """Evaluate a single question with optional data loading."""
    async with semaphore:
        logger.info(f"\n--- Processing Question {question_number}/{total_questions} ---")

        question_id = data_sample.get('question_id', f"q_{uuid.uuid4().hex[:8]}")
        question_text = data_sample.get('question')
        choices = data_sample.get('choices')
        correct_choice_index = data_sample.get('correct_choice_index')

        # Data for loading (if not skipping)
        haystack_session_ids = data_sample.get('haystack_session_ids', [])
        haystack_sessions_data = data_sample.get('haystack_sessions', [])

        if not all([question_text, choices, correct_choice_index is not None]):
            logger.error(f"Question {question_number} (ID: {question_id}): Missing required fields for evaluation. Skipping.")
            return {
                "question_id": question_id,
                "question_text": question_text,
                "status": "SKIPPED - Missing data",
                "query_time": 0.0
            }

        logger.info(f"Question {question_number} (ID: {question_id}): {question_text}")

        try:
            # Step 1: Initialize MemFuse instance for this question
            user_name = question_id  # Use question_id as user name for data isolation
            agent_name = "agent_default"  # Keep default agent name

            logger.info(f"Initializing MemFuse for Q{question_number}...")
            query_memory_instance = await memfuse_client.init(
                user=user_name,
                agent=agent_name,
            )

            # Step 2: Load haystack data if not skipping
            if not skip_data_loading:
                if haystack_sessions_data and haystack_session_ids:
                    logger.info(f"Loading {len(haystack_sessions_data)} haystack sessions for Q{question_number}...")

                    for session_id, messages_for_session in zip(haystack_session_ids, haystack_sessions_data):
                        if not messages_for_session:
                            logger.warning(f"Q{question_number}: Skipping empty session '{session_id}'")
                            continue

                        logger.info(f"Loading session '{session_id}' for Q{question_number}")

                        # Create memory instance for this session
                        session_memory_instance = await memfuse_client.init(
                            user=user_name,
                            agent=agent_name,
                            session=session_id
                        )

                        # Convert messages to MemFuse format
                        memfuse_messages = convert_messages_for_memfuse(messages_for_session, dataset_name)

                        logger.info(f"Adding {len(memfuse_messages)} messages to session '{session_id}'")
                        add_result = await session_memory_instance.add(memfuse_messages)

                        if add_result.get("status") == "success":
                            logger.info(f"Successfully loaded session '{session_id}' for Q{question_number}")
                        else:
                            logger.error(f"Failed to load session '{session_id}' for Q{question_number}: {add_result}")
                            continue

                    logger.info(f"Q{question_number}: Loaded haystack data successfully")
                else:
                    logger.warning(f"Q{question_number}: No haystack sessions data found")

            # Step 3: Test the question with MemFuse integration
            formatted_question = f"{question_text}\n\nChoices:\n"
            for j, choice in enumerate(choices):
                formatted_question += f"{j}. {choice}\n"

            logger.info(f"Testing Q{question_number} with MemFuse {llm_provider}...")
            llm_response_model = await _call_memfuse(
                query=formatted_question,
                choices_length=len(choices),
                model_name=model_name,
                memory_instance=query_memory_instance,
                llm_provider=llm_provider,
                max_retries=20,
                logger=logger
            )

            model_choice_idx = llm_response_model.index
            model_explanation = llm_response_model.reasoning
            is_correct = (model_choice_idx == correct_choice_index)

            # Capture retrieval debug info and timing from the SDK
            retrieval_debug = get_retrieval_debug_info(llm_provider)
            retrieval_timing = get_retrieval_timing_info(llm_provider)

            retrieved_memories_count = 0
            query_duration = retrieval_timing or 0.0  # Default to 0 if timing not available

            # Calculate retrieval metrics for LME and MSC datasets
            retrieval_metrics = {}
            if dataset_name in ["lme", "msc"]:
                haystack_sessions = data_sample.get('haystack_sessions', [])
                if haystack_sessions and retrieval_debug:
                    # Get the correct answer text based on dataset type
                    if dataset_name == "lme":
                        answer_text = choices[correct_choice_index] if correct_choice_index < len(choices) else ""
                    elif dataset_name == "msc":
                        # MSC dataset has the answer in the 'answer' field
                        answer_text = data_sample.get('answer', '')
                    # Safe access to retrieval_debug to avoid NoneType errors
                    retrieved_memories = retrieval_debug.get("data", {}).get("results", []) if retrieval_debug else []

                    # Calculate enhanced metrics (primary)
                    enhanced_metrics = calculate_enhanced_retrieval_metrics(
                        answer_text=answer_text,
                        haystack_sessions=haystack_sessions,
                        retrieved_memories=retrieved_memories,
                        logger=logger
                    )
                    
                    # Also calculate legacy metrics for comparison
                    answer_messages = extract_answer_containing_messages(haystack_sessions)
                    legacy_metrics = calculate_retrieval_metrics(
                        answer_messages, 
                        retrieved_memories,
                        logger
                    )
                    
                    # Use enhanced metrics as primary, but include both for comparison
                    retrieval_metrics = enhanced_metrics.copy()
                    retrieval_metrics.update({
                        "legacy_precision": legacy_metrics["precision"],
                        "legacy_recall": legacy_metrics["recall"], 
                        "legacy_f1": legacy_metrics["f1"]
                    })
                    
                    logger.info(f"Q{question_number} ENHANCED retrieval metrics - "
                              f"Precision: {enhanced_metrics['precision']:.3f}, "
                              f"Recall: {enhanced_metrics['recall']:.3f}, "
                              f"F1: {enhanced_metrics['f1']:.3f}")
                    logger.info(f"Q{question_number} Legacy metrics - "
                              f"Precision: {legacy_metrics['precision']:.3f}, "
                              f"Recall: {legacy_metrics['recall']:.3f}, "
                              f"F1: {legacy_metrics['f1']:.3f}")

            retrieved_memories_summary = None
            retrieved_memories_content = None
            if retrieval_debug:
                results = retrieval_debug.get("data", {}).get("results", [])
                retrieved_memories_count = len(results)
                logger.info(f"MemFuse query for Q{question_number} took {query_duration * 1000:.2f} ms.")
                
                # Capture full retrieved memories content if verbose flag is enabled
                if retrieval_verbose and results:
                    retrieved_memories_content = []
                    for result in results:
                        memory_entry = {
                            "content": result.get("content", ""),
                            "score": result.get("score", 0.0),
                            "metadata": {k: v for k, v in result.items() if k not in ["content", "score"]}
                        }
                        retrieved_memories_content.append(memory_entry)
                    logger.info(f"Q{question_number}: Captured {len(retrieved_memories_content)} retrieved memories for detailed analysis")
                
                # Create a summary for logging failed questions
                if not is_correct and results:
                    retrieved_memories_summary = []
                    for result in results[:3]:  # Show first 3 for brevity
                        content = result.get("content", "")[:100]  # Truncate long content
                        score = result.get("score", 0)
                        retrieved_memories_summary.append(f"Score:{score:.3f} Content:{content}...")
                    logger.info(f"Q{question_number} FAILED - Retrieved {retrieved_memories_count} memories: {retrieved_memories_summary}")
                elif not is_correct:
                    logger.info(f"Q{question_number} FAILED - No memories retrieved from RAG")
            else:
                logger.warning(f"No retrieval debug info available for Q{question_number}")

            logger.info(f"--- Q{question_number} RESULT ---")
            logger.info(f"Question: {question_text}")
            logger.info(f"LLM's Choice: {model_choice_idx} ('{choices[model_choice_idx] if 0 <= model_choice_idx < len(choices) else 'Invalid Index'}')")
            logger.info(f"Correct Choice: {correct_choice_index} ('{choices[correct_choice_index]}')")
            logger.info(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
            logger.info(f"LLM's Explanation: {model_explanation}")

            result = {
                "question_id": question_id,
                "question_text": question_text,
                "model_choice_idx": model_choice_idx,
                "model_choice_text": choices[model_choice_idx] if 0 <= model_choice_idx < len(choices) else 'Invalid Index',
                "correct_choice_idx": correct_choice_index,
                "correct_choice_text": choices[correct_choice_index],
                "is_correct": is_correct,
                "explanation": model_explanation,
                "top_k": top_k,
                "retrieval_time_ms": query_duration * 1000,
                "retry_count": llm_response_model.retry_count,
                "failed_after_retries": llm_response_model.failed_after_retries,
                "retrieved_memories_count": retrieved_memories_count,
                "retrieval_debug_available": retrieval_debug is not None,
                "query_time": query_duration,
                # Add retrieval metrics
                **retrieval_metrics
            }
            
            # Add retrieved memories content if available
            if retrieved_memories_content is not None:
                result["retrieved_memories_content"] = retrieved_memories_content
                
            return result

        except ConnectionError as e:
            error_msg = f"Connection error with MemFuse server for Q{question_number} (ID: {question_id}): {e}"
            logger.error(error_msg)
            return {
                "question_id": question_id,
                "question_text": question_text,
                "status": f"FAILED - ConnectionError: {e}",
                "query_time": 0.0
            }
        except Exception as e:
            error_msg = f"Unexpected error for Q{question_number} (ID: {question_id}): {e}"
            logger.error(error_msg, exc_info=True)
            return {
                "question_id": question_id,
                "question_text": question_text,
                "status": f"FAILED - Exception: {e}",
                "query_time": 0.0
            }


async def run_question_by_question_evaluation(
        dataset: List[Dict[str, Any]],
        dataset_name: str,
        dataset_type: str,
        top_k: int = 3,
        model_name: str = "deepseek-ai/DeepSeek-V3.1",
        llm_provider: str = "openai",
        skip_data_loading: bool = False,
        concurrent: int = 1,
        retrieval_verbose: bool = False,
        logger: Optional[logging.Logger] = None
    ) -> BenchmarkResults:
    """Run benchmark evaluation with question-by-question data loading and testing.
    
    This function loads haystack data for each question individually and immediately 
    tests it, providing better isolation and debugging capabilities.
    
    Args:
        dataset: List of questions with haystack sessions, choices and correct answers
        dataset_name: Name of dataset for recording (msc, lme, locomo)  
        dataset_type: Type of dataset (msc, lme, locomo)
        top_k: Number of top results to retrieve from memory
        model_name: LLM model to use for evaluation
        llm_provider: LLM provider to use ("gemini", "openai", "anthropic")
        skip_data_loading: Skip loading haystack data (assumes already loaded)
        logger: Logger instance
    
    Returns:
        BenchmarkResults with evaluation statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Import here to avoid circular imports
    from memfuse import AsyncMemFuse
    from benchmarks.recorder import BenchmarkRecorder
    
    logger.info(f"Starting question-by-question benchmark evaluation for {len(dataset)} questions...")
    logger.info(f"Using {llm_provider} model: {model_name}, TOP_K: {top_k}, Concurrent: {concurrent}")
    logger.info(f"Data loading: {'SKIPPED' if skip_data_loading else 'ENABLED'}")
    
    # Initialize recorder
    recorder = BenchmarkRecorder(dataset_name=dataset_name, top_k=top_k)
    
    # Tracking variables
    all_query_times = []
    correct_answers_count = 0
    results_summary = []
    total_retries = 0
    final_failures = 0
    
    # Track total elapsed time
    total_start_time = time.perf_counter()
    
    # Use async context manager for proper resource cleanup
    async with AsyncMemFuse() as memfuse_client:
        # Create semaphore to control concurrency
        semaphore = asyncio.Semaphore(concurrent)

        # Create tasks for concurrent evaluation
        tasks = []
        for i, data_sample in enumerate(dataset):
            question_number = i + 1
            task = _evaluate_single_question_with_data_loading(
                semaphore=semaphore,
                memfuse_client=memfuse_client,
                data_sample=data_sample,
                question_number=question_number,
                total_questions=len(dataset),
                dataset_name=dataset_name,
                top_k=top_k,
                model_name=model_name,
                llm_provider=llm_provider,
                skip_data_loading=skip_data_loading,
                retrieval_verbose=retrieval_verbose,
                logger=logger
            )
            tasks.append(task)

        # Execute all tasks concurrently
        logger.info(f"Executing {len(tasks)} evaluation tasks with "
                    f"concurrency limit of {concurrent}...")

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results_list:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                results_summary.append({
                    "question_id": "unknown",
                    "question_text": "unknown",
                    "status": f"FAILED - Exception: {result}"
                })
                continue

            results_summary.append(result)

            # Collect statistics
            if result.get("is_correct") is not None:
                if result["is_correct"]:
                    correct_answers_count += 1

            if "query_time" in result:
                all_query_times.append(result["query_time"])

            # Collect retry statistics
            if ("retry_count" in result and
                    isinstance(result["retry_count"], int)):
                total_retries += result["retry_count"]
            if result.get("failed_after_retries", False):
                final_failures += 1

        # Calculate total elapsed time
        total_end_time = time.perf_counter()
        total_elapsed_time = total_end_time - total_start_time
        
        # Save raw results
        recorder.record_raw_results(results_summary)
        
        # Calculate and save summary
        valid_results = [item for item in results_summary if 'is_correct' in item]
        correct_count = sum(1 for item in valid_results if item.get('is_correct'))
        accuracy = (correct_count / len(valid_results)) * 100 if len(valid_results) > 0 else 0
        # Convert query times to milliseconds to match the expected format for summary
        all_query_times_ms = [t * 1000 for t in all_query_times]
        recorder.record_summary(all_query_times_ms, accuracy)
        
        # Display the overall summary with percentiles
        if all_query_times:
            p50_time = np.percentile(all_query_times, 50)
            p90_time = np.percentile(all_query_times, 90)
            p95_time = np.percentile(all_query_times, 95)
            
            logger.info(f"P50 Query Time: {p50_time * 1000:.2f}ms, P90 Query Time: {p90_time * 1000:.2f}ms, P95 Query Time: {p95_time * 1000:.2f}ms")
        
        # Log retry statistics
        logger.info(f"Total retries: {total_retries}")
        logger.info(f"Final failures after all retries: {final_failures}")
        
        # Calculate average retrieval metrics for LME dataset
        avg_precision = avg_recall = avg_f1 = 0.0
        retrieval_metrics_available = False
        
        if dataset_name == "lme":
            # Extract retrieval metrics from valid results that have them
            results_with_metrics = [
                item for item in valid_results 
                if 'precision' in item and 'recall' in item and 'f1' in item
            ]
            
            if results_with_metrics:
                avg_precision = sum(item['precision'] for item in results_with_metrics) / len(results_with_metrics)
                avg_recall = sum(item['recall'] for item in results_with_metrics) / len(results_with_metrics)
                avg_f1 = sum(item['f1'] for item in results_with_metrics) / len(results_with_metrics)
                retrieval_metrics_available = True
                
                logger.info(f"=== RETRIEVAL EVALUATION METRICS (LME) ===")
                logger.info(f"Questions with retrieval metrics: {len(results_with_metrics)}/{len(valid_results)}")
                logger.info(f"Average Precision: {avg_precision:.3f}")
                logger.info(f"Average Recall: {avg_recall:.3f}")
                logger.info(f"Average F1: {avg_f1:.3f}")
                logger.info(f"========================================")
        
        return BenchmarkResults(
            question_results=results_summary,
            query_times=all_query_times,
            success_count=len(valid_results),
            total_count=len(dataset),
            accuracy=accuracy,
            total_elapsed_time=total_elapsed_time,
            total_retries=total_retries,
            final_failures=final_failures,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1=avg_f1,
            retrieval_metrics_available=retrieval_metrics_available
        )