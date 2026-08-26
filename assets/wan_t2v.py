import torch, os
from diffusers import WanPipeline, AutoModel
from diffusers.utils import export_to_video
from transformers import UMT5EncoderModel, BitsAndBytesConfig as TB
from diffusers import BitsAndBytesConfig as DB

M = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
NEG = "text, captions, subtitles, watermark, logo, letters, numbers, distorted faces, deformed hands, extra fingers, flicker, cartoon, morphing"

te = UMT5EncoderModel.from_pretrained(M, subfolder="text_encoder", quantization_config=TB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16), torch_dtype=torch.float16)
tr = AutoModel.from_pretrained(M, subfolder="transformer", quantization_config=DB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16), torch_dtype=torch.float16)
pipe = WanPipeline.from_pretrained(M, text_encoder=te, transformer=tr, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.vae.enable_tiling()

SHOTS = [
    ("shot0_chaos",
     "Handheld cinematic shot inside a busy modern office in warm evening light, semi-realistic high-end 3D animation style like a premium video game cinematic. Stressed employees buried in administrative work: a tired man slumped behind tall stacks of paper invoices, a woman flipping anxiously between dense spreadsheets on two monitors, another employee scrolling an overflowing email inbox rubbing his eyes, desks cluttered with teetering piles of paper. Slow push-in camera, overwhelmed frustrated expressions, shallow depth of field, vertical composition, no on-screen text anywhere"),
    ("shot4_relief",
     "Smooth slow dolly shot through a calm, organized modern office in bright warm morning light, semi-realistic high-end 3D animation style like a premium video game cinematic. Relaxed happy employees: two colleagues smiling while collaborating at a clean desk, a man talking cheerfully on a headset, a woman writing simple abstract shapes on a whiteboard, desks clear and tidy with single neat document trays. Far in the soft-focus background a blonde woman in a dark green blazer stands watching the team, satisfied. Shallow depth of field, vertical composition, no on-screen text anywhere"),
]

for name, prompt in SHOTS:
    out_path = f"/content/{name}.mp4"
    if os.path.exists(out_path):
        print("skip", name, flush=True)
        continue
    print("generating", name, flush=True)
    out = pipe(prompt=prompt, negative_prompt=NEG, height=832, width=480, num_frames=97, num_inference_steps=30, guidance_scale=5.0, generator=torch.Generator().manual_seed(31)).frames[0]
    export_to_video(out, out_path, fps=24)
    print("DONE", out_path, flush=True)
print("ALL DONE", flush=True)
