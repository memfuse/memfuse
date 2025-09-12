# Dataset configurations
DATASET_CONFIGS = {
    "msc": {
        "name": "MSC (Multiple-choice Situation Comprehension)",
        "dataset_id": "Percena/msc-memfuse-mc10",
        "data_file": "data/msc_memfuse_mc10.json",
        "description": "Multiple-choice Situation Comprehension dataset",
        "question_type": "conversation"
    },
    "lme": {
        "name": "LME (LongMemEval)",
        "dataset_id": "Percena/lme-mc10",
        "data_file": "data/lme_s_mc10.json",
        "description": "LongMemEval dataset",
        "question_type": "factual"
    },
    "locomo": {
        "name": "LoCoMo (Long Conversation Memory)",
        "dataset_id": "Percena/locomo-mc10",
        "data_file": "data/locomo_mc10.json",
        "description": "Long Conversation Memory dataset",
        "question_type": "conversation"
    }
}