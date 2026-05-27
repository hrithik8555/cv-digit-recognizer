import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np

# Import the model from the project architecture
from src.models.cnn import SimpleCNN

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = SimpleCNN().to(device)

# Attempt to load trained weights. 
# If they don't exist yet, it will just make random predictions.
try:
    model.load_state_dict(torch.load('model.pth', map_location=device))
    print("Model weights loaded successfully.")
except FileNotFoundError:
    print("Warning: 'model.pth' not found. The model will make random predictions.")
    print("Please train your model and save it as 'model.pth' in the project root.")

model.eval()

def predict_digit(img):
    if img is None:
        return "Please draw a digit."

    # Handle the dictionary structure from Gradio 4 Sketchpad
    if isinstance(img, dict):
        img_data = img['composite']
    else:
        img_data = img

    # Convert to PIL Image and Grayscale
    if isinstance(img_data, np.ndarray):
        img_pil = Image.fromarray(img_data).convert('L')
    else:
        img_pil = img_data.convert('L')
        
    # MNIST models are trained on digits that are white on a black background.
    # We need to invert the image if the user drew black ink on a white canvas.
    # Check the top-left pixel (background) to see if it's mostly white
    if img_pil.getpixel((0, 0)) > 128:
        img_pil = ImageOps.invert(img_pil)

    # Standard MNIST Transformations
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        # Normalize with MNIST mean and std
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])

    # Preprocess the image
    tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        # Apply Softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # Map predictions to class labels (0-9)
    predictions = {str(i): float(probabilities[i]) for i in range(10)}
    return predictions

# Build the custom Gradio Blocks UI
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.header-box {
    text-align: center; 
    max-width: 800px; 
    margin: 2rem auto; 
    padding-bottom: 2rem; 
    border-bottom: 1px solid #e5e7eb;
}
.title-text {
    font-size: 3rem; 
    font-weight: 800; 
    color: #1e293b; 
    margin-bottom: 0;
}
.subtitle-text {
    font-size: 1.2rem; 
    color: #64748b; 
    margin-top: 10px;
}
"""

with gr.Blocks() as demo:
    gr.HTML(f'''
    <div class="header-box">
        <h1 class="title-text">Neural Canvas</h1>
        <p class="subtitle-text">
            A Convolutional Neural Network trained on MNIST.<br>
            Draw a digit (<b>0-9</b>) below and watch the model's confidence shift in real-time.
        </p>
    </div>
    ''')
    
    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Sketchpad(
                type="pil", 
                label="Drawing Area"
            )
            gr.Markdown("*Tip: Draw near the center and try to fill the space for better accuracy!*")
                
        with gr.Column(scale=1):
            output_label = gr.Label(num_top_classes=4, label="Network Confidence")

    # Real-time updates as the user draws
    canvas.change(fn=predict_digit, inputs=canvas, outputs=output_label)

if __name__ == "__main__":
    demo.launch(share=False, css=custom_css, theme=gr.themes.Monochrome(primary_hue="blue", neutral_hue="slate"))
