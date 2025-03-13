#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A standalone Tkinter app that uses a pretrained BigGAN generator (via pytorch_pretrained_biggan)
and CLIP to generate images from a text prompt via CLIP-guided latent optimization.
Both the latent noise vector and a continuous class vector (via softmax on logits) are optimized 
to maximize the CLIP similarity between the generated image and the text prompt.
This version reduces the optimization steps for quicker intermediate results.
Licensed under the MIT License.
"""

import os
import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import time

import torch
import torch.nn.functional as F
from torchvision import transforms

from pytorch_pretrained_biggan import BigGAN
import clip

# Set device (GPU is recommended for speed).
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load a pretrained BigGAN model. Use a valid model name.
print("Loading BigGAN 'biggan-deep-256' weights...")
biggan = BigGAN.from_pretrained('biggan-deep-256', cache_dir='./biggan_cache').to(device)
print("BigGAN loaded.")

# Load CLIP model.
print("Loading CLIP model...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
print("CLIP loaded.")

def define_parameters():
    # For faster results, we use fewer steps.
    params = {
        "prompt": "A golden retriever in a park",
        "num_steps": 50,            # Reduced number of optimization steps for quick feedback.
        "lr": 0.05,
        "truncation": 0.4,
        "output_dir": "biggan_clip_output",
        "display_size": 256
    }
    return params

def postprocess_biggan_output(img_tensor):
    # BigGAN returns images as tensors in [0, 1]. Convert to PIL image.
    to_pil = transforms.ToPILImage()
    img = to_pil(img_tensor.squeeze(0).cpu())
    return img

def optimize_biggan(prompt, params, update_callback, log_callback):
    os.makedirs(params["output_dir"], exist_ok=True)
    
    start_time = time.time()
    log_callback("Starting optimization...")
    
    # Encode the text prompt with CLIP.
    with torch.no_grad():
        text_tokens = clip.tokenize([prompt]).to(device)
        text_features = clip_model.encode_text(text_tokens).detach()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Initialize latent vector z and class logits.
    z = torch.randn(1, 128, device=device, requires_grad=True)
    y_logits = torch.zeros(1, 1000, device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([z, y_logits], lr=params["lr"])
    
    for step in range(1, params["num_steps"] + 1):
        optimizer.zero_grad()
        y = F.softmax(y_logits, dim=-1)
        generated = biggan(z, y, params["truncation"])
        pil_img = postprocess_biggan_output(generated)
        clip_input = clip_preprocess(pil_img).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(clip_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        loss = -F.cosine_similarity(image_features, text_features).mean()
        loss.backward()
        optimizer.step()
        
        if step % 5 == 0:
            elapsed = time.time() - start_time
            log_callback(f"Step {step}/{params['num_steps']} | Loss: {loss.item():.4f} | Elapsed: {elapsed:.1f}s")
            with torch.no_grad():
                current_img = postprocess_biggan_output(generated)
            update_callback(current_img)
    
    with torch.no_grad():
        final_generated = biggan(z, F.softmax(y_logits, dim=-1), params["truncation"])
    final_img = postprocess_biggan_output(final_generated)
    out_path = os.path.join(params["output_dir"], f"generated_{int(time.time())}.png")
    final_img.save(out_path)
    log_callback(f"Final image saved to {out_path}")
    return final_img

class BigGANCLIPApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BigGAN+CLIP Generator (Quick Demo)")
        self.geometry("900x800")
        
        self.log_text = ScrolledText(self, height=8)
        self.log_text.pack(fill=tk.X, padx=10, pady=5)
        
        prompt_frame = tk.Frame(self)
        prompt_frame.pack(padx=10, pady=5, fill=tk.X)
        tk.Label(prompt_frame, text="Prompt:").pack(side=tk.LEFT)
        self.prompt_entry = tk.Entry(prompt_frame, width=80)
        self.prompt_entry.pack(side=tk.LEFT, padx=5)
        params = define_parameters()
        self.prompt_entry.insert(0, params["prompt"])
        
        btn_frame = tk.Frame(self)
        btn_frame.pack(padx=10, pady=5)
        self.gen_btn = tk.Button(btn_frame, text="Generate Image", command=self.start_generation)
        self.gen_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = tk.Button(btn_frame, text="Stop", command=self.stop_generation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.img_label = tk.Label(self)
        self.img_label.pack(padx=10, pady=10)
        
        self.params = define_parameters()
        os.makedirs(self.params["output_dir"], exist_ok=True)
        self.log("Loaded default parameters.")
        
        self.running = False
        self.gen_thread = None
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
    
    def stop_generation(self):
        self.running = False
        self.gen_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log("Stopping generation... (current step may complete)")
    
    def start_generation(self):
        self.gen_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.running = True
        prompt = self.prompt_entry.get()
        self.log(f"Starting optimization for prompt: '{prompt}'")
        self.gen_thread = threading.Thread(target=self.run_optimization, args=(prompt,))
        self.gen_thread.start()
    
    def update_image(self, pil_img):
        disp_img = pil_img.resize((self.params["display_size"], self.params["display_size"]))
        self.tk_img = ImageTk.PhotoImage(disp_img)
        self.img_label.config(image=self.tk_img)
        self.img_label.image = self.tk_img
    
    def run_optimization(self, prompt):
        try:
            optimized_img = optimize_biggan(
                prompt,
                self.params,
                update_callback=self.update_image,
                log_callback=self.log
            )
            self.update_image(optimized_img)
        except Exception as e:
            self.log("Error during optimization: " + str(e))
        finally:
            self.running = False
            self.gen_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

if __name__ == "__main__":
    app = BigGANCLIPApp()
    app.mainloop()
