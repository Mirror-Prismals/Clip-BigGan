#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An improved standalone Tkinter app that uses a pretrained BigGAN generator (via pytorch_pretrained_biggan)
and CLIP to generate images from a text prompt via CLIP-guided latent optimization.
Both the latent noise vector and a continuous class vector (via softmax on logits) are optimized 
to maximize the CLIP similarity between the generated image and the text prompt.
This version uses longer, steadier optimization with improved regularization and gradient clipping
to help reduce psychedelic color artifacts. It also supports aspect ratio options: square (1:1),
landscape (16:9), and portrait (9:16). Additionally, the number of optimization steps and learning rate
can be adjusted at runtime.
Licensed under the MIT License.
"""

import os
import threading
import tkinter as tk
from tkinter import ttk
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

def define_default_parameters():
    params = {
        "prompt": "A golden retriever in a park",
        "num_steps": 250,            # Default number of optimization steps.
        "lr": 0.003,                 # Default learning rate.
        "truncation": 0.4,
        "output_dir": "biggan_clip_output",
        "base_display_size": 256,    # Base size for display; will be adjusted by aspect ratio.
        "aspect_ratio": "square"     # Options: "square", "16:9", "9:16"
    }
    return params

def get_display_size(aspect_ratio, base_size):
    """
    Returns a tuple (width, height) for display based on the desired aspect ratio.
    """
    if aspect_ratio == "square":
        return (base_size, base_size)
    elif aspect_ratio == "16:9":
        return (base_size, int(base_size * 9 / 16))
    elif aspect_ratio == "9:16":
        return (int(base_size * 9 / 16), base_size)
    else:
        return (base_size, base_size)

def adjust_aspect_ratio(pil_img, aspect_ratio):
    """
    Adjusts a square PIL image to the desired aspect ratio by center-cropping.
    For a square image (e.g., 256x256):
      - "16:9" crops vertically.
      - "9:16" crops horizontally.
    """
    if aspect_ratio == "square":
        return pil_img
    width, height = pil_img.size
    if aspect_ratio == "16:9":
        target_height = int(width * 9 / 16)
        if target_height < height:
            top = (height - target_height) // 2
            return pil_img.crop((0, top, width, top + target_height))
        else:
            return pil_img.resize((width, target_height))
    elif aspect_ratio == "9:16":
        target_width = int(height * 9 / 16)
        if target_width < width:
            left = (width - target_width) // 2
            return pil_img.crop((left, 0, left + target_width, height))
        else:
            return pil_img.resize((target_width, height))
    else:
        return pil_img

def postprocess_biggan_output(img_tensor):
    """
    Converts BigGAN's output tensor (assumed to be in [0,1]) to a PIL image.
    This function helps ensure proper conversion to avoid color distortions.
    """
    to_pil = transforms.ToPILImage()
    img = to_pil(img_tensor.squeeze(0).cpu())
    return img

def optimize_biggan(prompt, params, update_callback, log_callback, should_stop):
    os.makedirs(params["output_dir"], exist_ok=True)
    start_time = time.time()
    log_callback("Starting optimization...")

    # Encode the text prompt with CLIP.
    with torch.no_grad():
        text_tokens = clip.tokenize([prompt]).to(device)
        text_features = clip_model.encode_text(text_tokens).detach()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Initialize latent vector and class logits.
    z = torch.randn(1, 128, device=device, requires_grad=True)
    y_logits = torch.zeros(1, 1000, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([z, y_logits], lr=params["lr"])

    # Use stronger regularization to help keep the latent space near its prior.
    lambda_z = 0.0015
    lambda_y = 0.0015

    for step in range(1, params["num_steps"] + 1):
        if should_stop():
            log_callback("Optimization stopped by user.")
            break

        optimizer.zero_grad()
        y = F.softmax(y_logits, dim=-1)
        generated = biggan(z, y, params["truncation"])
        pil_img = postprocess_biggan_output(generated)
        
        # Adjust image to desired aspect ratio.
        adjusted_img = adjust_aspect_ratio(pil_img, params["aspect_ratio"])
        
        # Feed the adjusted image into CLIP.
        clip_input = clip_preprocess(adjusted_img).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(clip_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        cosine_sim = F.cosine_similarity(image_features, text_features).mean()
        loss = 1 - cosine_sim
        loss = loss + lambda_z * torch.mean(z ** 2) + lambda_y * torch.mean(y_logits ** 2)

        loss.backward()
        # Tighter gradient clipping for stability.
        torch.nn.utils.clip_grad_norm_([z, y_logits], max_norm=0.5)
        optimizer.step()

        # Clamp latent vector more tightly.
        z.data.clamp_(-1, 1)

        if step % 5 == 0:
            elapsed = time.time() - start_time
            log_callback(f"Step {step}/{params['num_steps']} | Loss: {loss.item():.4f} | Cosine: {cosine_sim.item():.4f} | Elapsed: {elapsed:.1f}s")
            with torch.no_grad():
                current_img = postprocess_biggan_output(generated)
                current_adjusted = adjust_aspect_ratio(current_img, params["aspect_ratio"])
            update_callback(current_adjusted)

    with torch.no_grad():
        final_generated = biggan(z, F.softmax(y_logits, dim=-1), params["truncation"])
    final_img = postprocess_biggan_output(final_generated)
    final_adjusted = adjust_aspect_ratio(final_img, params["aspect_ratio"])
    out_path = os.path.join(params["output_dir"], f"generated_{int(time.time())}.png")
    final_adjusted.save(out_path)
    log_callback(f"Final image saved to {out_path}")
    return final_adjusted

class BigGANCLIPApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BigGAN+CLIP Generator (Steady Optimization)")
        self.geometry("950x950")

        self.log_text = ScrolledText(self, height=8)
        self.log_text.pack(fill=tk.X, padx=10, pady=5)

        options_frame = tk.Frame(self)
        options_frame.pack(padx=10, pady=5, fill=tk.X)

        # Prompt
        tk.Label(options_frame, text="Prompt:").grid(row=0, column=0, sticky="w")
        self.prompt_entry = tk.Entry(options_frame, width=60)
        self.prompt_entry.grid(row=0, column=1, padx=5, pady=5)
        defaults = define_default_parameters()
        self.prompt_entry.insert(0, defaults["prompt"])

        # Aspect Ratio
        tk.Label(options_frame, text="Aspect Ratio:").grid(row=1, column=0, sticky="w")
        self.aspect_ratio_var = tk.StringVar(value="square")
        aspect_menu = ttk.Combobox(options_frame, textvariable=self.aspect_ratio_var, state="readonly", width=10)
        aspect_menu['values'] = ("square", "16:9", "9:16")
        aspect_menu.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # Number of Steps
        tk.Label(options_frame, text="Num Steps:").grid(row=2, column=0, sticky="w")
        self.steps_entry = tk.Entry(options_frame, width=10)
        self.steps_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.steps_entry.insert(0, str(defaults["num_steps"]))

        # Learning Rate
        tk.Label(options_frame, text="Learning Rate:").grid(row=3, column=0, sticky="w")
        self.lr_entry = tk.Entry(options_frame, width=10)
        self.lr_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        self.lr_entry.insert(0, str(defaults["lr"]))

        btn_frame = tk.Frame(self)
        btn_frame.pack(padx=10, pady=5)
        self.gen_btn = tk.Button(btn_frame, text="Generate Image", command=self.start_generation)
        self.gen_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = tk.Button(btn_frame, text="Stop", command=self.stop_generation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.img_label = tk.Label(self)
        self.img_label.pack(padx=10, pady=10)

        # Set default parameters and output directory.
        self.params = define_default_parameters()
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
        self.prompt_entry.config(state=tk.DISABLED)
        self.steps_entry.config(state=tk.DISABLED)
        self.lr_entry.config(state=tk.DISABLED)
        self.running = True
        prompt = self.prompt_entry.get()
        # Update aspect ratio parameter.
        self.params["aspect_ratio"] = self.aspect_ratio_var.get()
        # Update num_steps and learning rate based on user input.
        try:
            self.params["num_steps"] = int(self.steps_entry.get())
        except ValueError:
            self.log("Invalid number of steps; using default.")
            self.params["num_steps"] = define_default_parameters()["num_steps"]
        try:
            self.params["lr"] = float(self.lr_entry.get())
        except ValueError:
            self.log("Invalid learning rate; using default.")
            self.params["lr"] = define_default_parameters()["lr"]

        self.log(f"Starting optimization for prompt: '{prompt}' with aspect ratio '{self.params['aspect_ratio']}', "
                 f"steps: {self.params['num_steps']}, lr: {self.params['lr']}")
        self.gen_thread = threading.Thread(target=self.run_optimization, args=(prompt,))
        self.gen_thread.start()

    def update_image(self, pil_img):
        # Adjust the displayed image size based on selected aspect ratio.
        disp_size = get_display_size(self.params["aspect_ratio"], self.params["base_display_size"])
        disp_img = pil_img.resize(disp_size)
        self.tk_img = ImageTk.PhotoImage(disp_img)
        self.img_label.config(image=self.tk_img)
        self.img_label.image = self.tk_img

    def run_optimization(self, prompt):
        try:
            optimized_img = optimize_biggan(
                prompt,
                self.params,
                update_callback=self.update_image,
                log_callback=self.log,
                should_stop=lambda: not self.running
            )
            self.update_image(optimized_img)
        except Exception as e:
            self.log("Error during optimization: " + str(e))
        finally:
            self.running = False
            self.gen_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.prompt_entry.config(state=tk.NORMAL)
            self.steps_entry.config(state=tk.NORMAL)
            self.lr_entry.config(state=tk.NORMAL)

if __name__ == "__main__":
    app = BigGANCLIPApp()
    app.mainloop()
