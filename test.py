import torch
from torch.cuda.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline


def prompt_format(system_prompt, instruction):
    prompt = f"""{system_prompt}

 ####### Instruction:
{instruction}

 %%%%%%% Response:
"""
    return prompt

system_prompt = """
You're an AI Large Language Model developed(created) by an AI developer named Tuấn, the architecture of you is decoder-based LM, your task are to think loudly step by step before give a good and relevant response
to the user request, answer in the language the user preferred.

The AI has been trained to answer questions, provide recommendations, and help with decision making. The AI thinks outside the box and follows the user requests
"""
instruction = "Xin chào"

formatted_prompt = prompt_format(system_prompt, instruction)
print(formatted_prompt)

model_name = "1TuanPham/T-Llama"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             use_cache=True,
                                             device_map="auto"
                                             )
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
streamer = TextStreamer(tokenizer, skip_special_tokens=True)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, streamer=streamer)

with autocast():
  output_default = pipe(formatted_prompt, pad_token_id=50256, max_new_tokens=128)
  print(output_default)
