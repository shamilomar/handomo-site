import torch, os, sys, gc
from diffusers import WanImageToVideoPipeline, WanPipeline, AutoModel
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel, BitsAndBytesConfig as TB
from diffusers import BitsAndBytesConfig as DB

M = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
RAW = "https://raw.githubusercontent.com/shamilomar/handomo-site/main/assets/"
NEG = "text, captions, subtitles, watermark, logo, letters, numbers, distorted face, deformed hands, extra fingers, flicker, cartoon, morphing"

W_, H_ = 704, 1248
FRAMES, STEPS = 97, 30

def build():
    te = UMT5EncoderModel.from_pretrained(M, subfolder="text_encoder", quantization_config=TB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16), torch_dtype=torch.float16)
    tr = AutoModel.from_pretrained(M, subfolder="transformer", quantization_config=DB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16), torch_dtype=torch.float16)
    pipe = WanImageToVideoPipeline.from_pretrained(M, text_encoder=te, transformer=tr, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.vae.enable_tiling()
    return pipe

SHOTS = [
    ("shot1_aisle", "scene1_aisle_walk.png",
     "The blonde woman in the dark green blazer walks forward down the office aisle toward the camera, camera slowly tracking backward ahead of her, natural confident stride, stressed employees at paper-piled desks glance up at her as she passes, soft warm cinematic lighting, shallow depth of field, consistent character throughout, no deformation or drift"),
    ("shot2_transform", "scene2_transformation.png",
     "Side profile tracking shot, the blonde woman in the dark green blazer strides from right to left through the office, camera panning with her, behind her the desks stay tidy and employees relaxed, ahead of her loose paper invoices quietly settle into neat stacks as she approaches, subtle elegant motion, no glow effects, consistent character throughout, no deformation"),
    ("shot5_final", "character_v1.png",
     "The blonde woman in the dark green blazer stands still and looks directly into the camera with a subtle confident smile, very gentle breathing motion, soft blink, hair moves slightly, background office softly blurred behind her, camera almost static with a very slow push in, consistent character, no deformation"),
]

if __name__ == "__main__":
    pipe = build()
    for name, imgfile, prompt in SHOTS:
        out_path = f"/content/{name}.mp4"
        if os.path.exists(out_path):
            print("skip", name, flush=True)
            continue
        print("generating", name, flush=True)
        img = load_image(RAW + imgfile).resize((W_, H_))
        try:
            out = pipe(image=img, prompt=prompt, negative_prompt=NEG, height=H_, width=W_, num_frames=FRAMES, num_inference_steps=STEPS, guidance_scale=5.0, generator=torch.Generator().manual_seed(11)).frames[0]
        except torch.cuda.OutOfMemoryError:
            print("OOM at 704x1248, retrying 480x832", flush=True)
            gc.collect(); torch.cuda.empty_cache()
            img2 = load_image(RAW + imgfile).resize((480, 832))
            out = pipe(image=img2, prompt=prompt, negative_prompt=NEG, height=832, width=480, num_frames=FRAMES, num_inference_steps=STEPS, guidance_scale=5.0, generator=torch.Generator().manual_seed(11)).frames[0]
        export_to_video(out, out_path, fps=24)
        print("DONE", out_path, flush=True)
    print("ALL DONE", flush=True)
