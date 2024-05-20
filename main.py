import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Define the generate function
def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]

    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)

# Initialize the main application window
app = tk.Tk()
app.geometry('532x622')  # Set the size of the window
app.title("Stable Bud")  # Set the title of the window
ctk.set_appearance_mode("dark")  # Set the appearance mode of customtkinter

# Create the entry widget using CTkEntry with supported arguments
prompt = ctk.CTkEntry(master=app, height=40, width=512, text_color="black", fg_color="white")
prompt.place(x=10, y=10)  # Place the entry widget at the specified position

# Create the button widget using CTkButton with supported arguments
trigger = ctk.CTkButton(master=app, height=40, width=120, text="Generate", text_color="white", fg_color="blue", command=generate)
trigger.place(x=206, y=60)  # Place the button widget at the specified position

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, variant="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

# Start the Tkinter main loop to run the application
app.mainloop()
