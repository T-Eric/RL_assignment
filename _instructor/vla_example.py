"""
Bonus: Simplified VLA (Vision-Language-Action) training example.

This script demonstrates how to fine-tune a vision-language model (Qwen2.5-VL)
to predict drone navigation actions from first-person-view (FPV) images.

Pipeline overview:
  1. Collect trajectories via PPO (you already did this)
  2. Replay trajectories in AirSim to capture FPV images at each waypoint
  3. Build a dataset: (FPV image, current_pose, target_coord) -> action_delta
  4. Fine-tune Qwen2.5-VL with LoRA using SFTTrainer
  5. Evaluate action prediction error on in-sample and out-of-sample data

Prerequisites:
  pip install transformers trl peft accelerate qwen-vl-utils bitsandbytes

NOTE: This is a SIMPLIFIED example. You will need to:
  - Implement FPV image capture from AirSim (see capture_fpv_hint below)
  - Prepare your dataset in the expected format
  - Adjust hyperparameters for your compute budget
"""

import os
import json
import random
import pickle
import numpy as np
import torch
from functools import partial
from PIL import Image

# ============================================================
# 1. Action Tokenizer — discretizes continuous actions into tokens
# ============================================================

class ActionTokenizer:
    """Discretizes continuous action values into token IDs."""

    def __init__(self, n_bins, tokenizer, min_action, max_action):
        self.n_bins = n_bins
        self.tokenizer = tokenizer
        self.min_action = min_action
        self.max_action = max_action
        self.bins = np.linspace(min_action, max_action, n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        self.vocab_size = len(self.tokenizer)

    def tokenize(self, action):
        action = np.clip(action, self.min_action, self.max_action)
        discretized = np.digitize(action, self.bins)
        return self.tokenizer.batch_decode(
            (self.vocab_size - discretized).tolist())

    def detokenize(self, action_token_ids):
        discretized = self.vocab_size - np.array(action_token_ids)
        discretized = np.clip(discretized - 1, 0, len(self.bin_centers) - 1)
        return self.bin_centers[discretized]


# ============================================================
# 2. Data formatting — converts samples to chat format for Qwen2.5-VL
# ============================================================

SYSTEM_PROMPT = (
    "You are a Vision-Language-Action model specialized in interpreting "
    "first-person-view drone images. Your task is to analyze the provided "
    "image, the current pose of the drone (x, y, yaw), along with the "
    "coordinate of a target building (x, y) that the drone is supposed to "
    "approach, and then respond with a pose delta (dx, dy) that the drone "
    "should achieve in the next step."
)

USER_TEMPLATE = "Current pose: {current_pose}\nTarget coordinate: {target_coordinate}\n"


def format_sample(sample):
    """Convert a data sample into Qwen2.5-VL chat format."""
    query = USER_TEMPLATE.format(
        current_pose=sample["current_pose"],
        target_coordinate=sample["target_coordinate"],
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": sample["image_path"]},
            {"type": "text", "text": query},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": sample["action_tokens"]},
        ]},
    ]


# ============================================================
# 3. Collate function for training
# ============================================================

def find_assistant_content_indexes(token_ids):
    """Find (start, end) ranges of assistant responses in token_ids."""
    # Qwen2.5 chat template markers
    START_SEQ = [151644, 77091, 198]   # <|im_start|>assistant\n
    END_SEQ = [151645, 198]            # <|im_end|>\n
    ranges = []
    for i in range(len(token_ids) - 2):
        if token_ids[i:i+3] == START_SEQ:
            for j in range(i + 3, len(token_ids) - 1):
                if token_ids[j:j+2] == END_SEQ:
                    ranges.append((i + 3, j + 2))
                    break
    return ranges


def collate_fn(examples, processor):
    from qwen_vl_utils import process_vision_info
    texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]
    image_inputs = [process_vision_info(ex)[0] for ex in examples]
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    labels_list = []
    for ids in batch["input_ids"].tolist():
        label_ids = [-100] * len(ids)
        for start, end in find_assistant_content_indexes(ids):
            label_ids[start:end] = ids[start:end]
        labels_list.append(label_ids)
    batch["labels"] = torch.tensor(labels_list, dtype=torch.int64)
    return batch


# ============================================================
# 4. Training
# ============================================================

def train_vla(dataset_pkl_path, model_id="Qwen/Qwen2.5-VL-7B-Instruct",
              output_dir="./vla_checkpoints", num_epochs=5, batch_size=4,
              num_bins=256):
    """
    Fine-tune Qwen2.5-VL for action prediction.

    Args:
        dataset_pkl_path: path to pickle with keys:
            'train_dataset', 'val_dataset' — lists of formatted chat samples
            'action_percentile_1', 'action_percentile_99' — for ActionTokenizer
            'num_bins' — number of discretization bins
        model_id: HuggingFace model identifier
        output_dir: where to save checkpoints
        num_epochs: training epochs
        batch_size: per-device batch size
        num_bins: action discretization bins
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig, get_peft_model

    with open(dataset_pkl_path, "rb") as f:
        data = pickle.load(f)

    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    for i in range(num_bins):
        processor.tokenizer.add_tokens(f"[action_{i}]")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.resize_token_embeddings(len(processor.tokenizer))

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=2e-5,
        lr_scheduler_type="constant",
        logging_steps=10,
        eval_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=1000,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=partial(collate_fn, processor=processor),
        processing_class=processor.tokenizer,
    )

    trainer.train()
    print("Training complete!")


# ============================================================
# 5. Hints for FPV image capture from AirSim
# ============================================================

def capture_fpv_hint():
    """
    To capture FPV images from AirSim for your trajectories:

    1. Start the AirSim simulation (env_airsim_16)
    2. For each trajectory waypoint (x, y, yaw):
       - Teleport drone: client.simSetVehiclePose(airsim.Pose(
             airsim.Vector3r(x, y, -15),  # NED: z is negative up
             airsim.to_quaternion(0, 0, yaw)), True)
       - Capture image: responses = client.simGetImages([
             airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
       - Save: img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
               img = img.reshape(responses[0].height, responses[0].width, 3)
    3. Build dataset pickle with format:
       Each sample = format_sample({
           "image_path": "path/to/frame.png",
           "current_pose": [x, y, yaw],
           "target_coordinate": [target_cx, target_cy],
           "action_tokens": "<tokenized dx dy>",
       })
    """
    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset pickle")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output_dir", default="./vla_checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    train_vla(args.dataset, args.model, args.output_dir, args.epochs, args.batch_size)
