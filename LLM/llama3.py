"""Extract volume statistics from LLaMA-3 8B weight matrices."""

import os

import torch
from transformers import LlamaForCausalLM

MODEL_PATH = ""
OUTPUT_DIR = ""
OUTPUT_FILENAME = "llama3_8b_volume.txt"
NUM_LAYERS = 32
CUDA_DEVICE = 1

TARGET_WEIGHT_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def write_volume_stats(output_file, matrix):
    """Write diagonal sum, mean, and variance of the Gram matrix."""
    gram_matrix = torch.matmul(matrix.T, matrix)
    diagonal = torch.diag(gram_matrix)
    diag_sum = torch.sum(diagonal).item()
    diag_mean = torch.mean(diagonal).item()
    diag_var = torch.var(diagonal, unbiased=False).item()

    print(
        f"Diag_sum: {diag_sum}  diag_mean: {diag_mean}  diag_var: {diag_var}"
    )
    print("=" * 50)
    output_file.write(f"sum_mean_var,{diag_sum},{diag_mean},{diag_var}\n")


def main():
    torch.cuda.set_device(CUDA_DEVICE)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, shape: {param.size()}")

    with open(output_path, "w", encoding="utf-8") as output_file:
        for name, param in model.named_parameters():
            for layer_idx in range(NUM_LAYERS):
                for suffix in TARGET_WEIGHT_SUFFIXES:
                    param_name = f"model.layers.{layer_idx}.{suffix}"
                    if name != param_name:
                        continue

                    parts = name.split(".")
                    output_file.write(f"{parts[2]}_{parts[4]},")
                    write_volume_stats(output_file, param.data)


if __name__ == "__main__":
    main()
