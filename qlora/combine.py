import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer


def save_model(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    base_tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    save_model("../../model/CodeLlama-7b-hf", output_path="./save_model/CodeLlama-7b-hf-1000", lora_path="./output/hpc_v1.0/checkpoint-1000")
