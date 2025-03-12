# import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# from dotenv import load_dotenv
# from huggingface_hub import login
# load_dotenv()  # Load environment variables from .env file
# hf_token = os.getenv("HF_TOKEN")
# login(hf_token)

# # Import necessary libraries
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from datasets import load_dataset
# from trl import SFTConfig, SFTTrainer, setup_chat_format
# import torch

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps" if torch.backends.mps.is_available() else "cpu"
# )

# # Load the model and tokenizer
# model_name = "HuggingFaceTB/SmolLM2-135M"
# model = AutoModelForCausalLM.from_pretrained(
#     pretrained_model_name_or_path=model_name
# ).to(device)
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
# # Set up the chat format
# model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

# # Set our name for the finetune to be saved &/ uploaded to
# finetune_name = "SmolLM2-FT-MVP"
# finetune_tags = ["smol-course", "module_1"]
# ds = load_dataset(path="HuggingFaceTB/smoltalk", name="everyday-conversations")

# # Configure the SFTTrainer
# sft_config = SFTConfig(
#     dataset_text_field="messages",
#     output_dir="./sft_output",
#     max_steps=1000,  # Adjust based on dataset size and desired training duration
#     per_device_train_batch_size=4,  # Set according to your GPU memory capacity
#     learning_rate=5e-5,  # Common starting point for fine-tuning
#     logging_steps=10,  # Frequency of logging training metrics
#     save_steps=100,  # Frequency of saving model checkpoints
#     evaluation_strategy="steps",  # Evaluate the model at regular intervals
#     eval_steps=50,  # Frequency of evaluation
#     use_mps_device=(
#         True if device == "mps" else False
#     ),  # Use MPS for mixed precision training
#     hub_model_id=finetune_name,  # Set a unique name for your model
# )

# # Initialize the SFTTrainer
# trainer = SFTTrainer(
#     model=model,
#     args=sft_config,
#     train_dataset=ds["train"],
#     tokenizer=tokenizer,
#     eval_dataset=ds["test"],
# )
# trainer.train()
# print("here")
# from transformers import ViTForImageClassification


'''
##################################### ORPO #############################################################
'''

# import torch
# import os
# from datasets import load_dataset
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
# )
# from trl import ORPOConfig, ORPOTrainer, setup_chat_format
# import bitsandbytes
# from trl import ORPOConfig, ORPOTrainer

# dataset = load_dataset(path="trl-lib/ultrafeedback_binarized")

# model_name = "HuggingFaceTB/SmolLM2-135M"

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps" if torch.backends.mps.is_available() else "cpu"
# )

# # Model to fine-tune
# model = AutoModelForCausalLM.from_pretrained(
#     pretrained_model_name_or_path=model_name,
#     torch_dtype=torch.float32,
# ).to(device)
# model.config.use_cache = False
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model, tokenizer = setup_chat_format(model, tokenizer)

# # Set our name for the finetune to be saved &/ uploaded to
# finetune_name = "SmolLM2-FT-ORPO-test"
# finetune_tags = ["smol-course", "module_1"]


# orpo_args = ORPOConfig(
#     # Small learning rate to prevent catastrophic forgetting
#     learning_rate=8e-6,
#     # Linear learning rate decay over training
#     lr_scheduler_type="linear",
#     # Maximum combined length of prompt + completion
#     max_length=1024,
#     # Maximum length for input prompts
#     max_prompt_length=512,
#     # Controls weight of the odds ratio loss (λ in paper)
#     beta=0.1,
#     # Batch size for training
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     # Helps with training stability by accumulating gradients before updating
#     gradient_accumulation_steps=4,
#     # Memory-efficient optimizer for CUDA, falls back to adamw_torch for CPU/MPS
#     optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
#     # Number of training epochs
#     num_train_epochs=5,
#     # When to run evaluation
#     evaluation_strategy="steps",
#     # Evaluate every 20% of training
#     eval_steps=0.2,
#     # Log metrics every step
#     logging_steps=1,
#     # Gradual learning rate warmup
#     warmup_steps=10,
#     # Disable external logging
#     report_to="none",
#     # Where to save model/checkpoints
#     output_dir="./results/",
#     # Enable MPS (Metal Performance Shaders) if available
#     use_mps_device=device == "mps",
#     hub_model_id=finetune_name,
# )
# trainer = ORPOTrainer(
#     model=model,
#     args=orpo_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     processing_class=tokenizer,
# )
# trainer.train()

##################################### ORPO #############################################################




'''
##################################### LORA #############################################################
'''


# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# import torch
# import torch.nn as nn
# import bitsandbytes as bnb
# from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


# import torch
# import transformers
# from datasets import load_dataset
# from peft import LoraConfig, get_peft_model
# from peft import PeftModel, PeftConfig
# from transformers import AutoModelForCausalLM, AutoTokenizer

# peft_model_id = "ybelkada/opt-6.7b-lora"
# config = PeftConfig.from_pretrained(peft_model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     config.base_model_name_or_path,
#     return_dict=True,
#     load_in_8bit=True,
#     device_map="auto",
# )
# tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# # Load the Lora model
# model = PeftModel.from_pretrained(model, peft_model_id)
# print("here")


##################################### LORA Training #############################################################


from datasets import load_dataset

# TODO: define your dataset and config using the path and name parameters
dataset = load_dataset(path="HuggingFaceTB/smoltalk", name="everyday-conversations")
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M"

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# Set up the chat format
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

# Set our name for the finetune to be saved &/ uploaded to
finetune_name = "SmolLM2-FT-Lora-test"
finetune_tags = ["smol-course", "module_1"]

from peft import LoraConfig

# TODO: Configure LoRA parameters
# r: rank dimension for LoRA update matrices (smaller = more compression)
rank_dimension = 6
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
lora_alpha = 8
# lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)
lora_dropout = 0.05

peft_config = LoraConfig(
    r=rank_dimension,  # Rank dimension - typically between 4-32
    lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
    lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
    bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
    target_modules="all-linear",  # Which modules to apply LoRA to
    task_type="CAUSAL_LM",  # Task type for model architecture
)

# Training configuration
# Hyperparameters based on QLoRA paper recommendations
args = SFTConfig(
    # Output settings
    output_dir=finetune_name,  # Directory to save model checkpoints
    # Training duration
    num_train_epochs=5,  # Number of training epochs
    # Batch size settings
    per_device_train_batch_size=2,  # Batch size per GPU
    gradient_accumulation_steps=2,  # Accumulate gradients for larger effective batch
    # Memory optimization
    gradient_checkpointing=True,  # Trade compute for memory savings
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    # Optimizer settings
    optim="adamw_torch_fused",  # Use fused AdamW for efficiency
    learning_rate=2e-4,  # Learning rate (QLoRA paper)
    max_grad_norm=0.3,  # Gradient clipping threshold
    # Learning rate schedule
    warmup_ratio=0.03,  # Portion of steps for warmup
    lr_scheduler_type="constant",  # Keep learning rate constant after warmup
    # Logging and saving
    logging_steps=10,  # Log metrics every N steps
    save_strategy="epoch",  # Save checkpoint every epoch
    # Precision settings
    bf16=True,  # Use bfloat16 precision
    # Integration settings
    push_to_hub=False,  # Don't push to HuggingFace Hub
    report_to="none",  # Disable external logging
)

max_seq_length = 1512  # max sequence length for model and packing of the dataset

# Create SFTTrainer with LoRA configuration
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    peft_config=peft_config,  # LoRA configuration
    max_seq_length=max_seq_length,  # Maximum sequence length
    tokenizer=tokenizer,
    packing=True,  # Enable input packing for efficiency
    dataset_kwargs={
        "add_special_tokens": False,  # Special tokens handled by template
        "append_concat_token": False,  # No additional separator needed
    },
)

trainer.train()
print("completed!")