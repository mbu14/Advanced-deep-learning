import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from Lab3PythonInsteadOfJupyterCuzItWorksBetterAndCuzILoveCarlLisper.py import Vocabulary

# IMPORTANT: If running this as a completely separate Python file, 
# you must import your model classes from the previous script. 
# (e.g., from my_training_script import CNNtoRNN, Vocabulary)

def setup_inference(checkpoint_path, embed_size=256, hidden_size=256, num_layers=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {checkpoint_path}...")
    
    # Tell PyTorch this specific custom class is safe to unpickle
    torch.serialization.add_safe_globals([Vocabulary])
    
    # Load the checkpoint (weights_only=True is the default now)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint["vocab"]

    # Initialize model with the same architecture
    model = CNNtoRNN(
        embed_size=embed_size,git,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        num_layers=num_layers
    ).to(device)

    # Load weights and set to evaluation moded
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    return model, vocab, device

def decode_caption(tokens, vocab):
    """Converts the model's predicted tokens back into a readable string."""
    words = []
    for token in tokens:
        # Handle case where token is already a string
        if isinstance(token, str):
            if token == "<EOS>": break
            if token not in ["<SOS>", "<PAD>"]:
                words.append(token)
            continue
            
        # Handle case where token is an integer index
        word = vocab.itos.get(token, "<UNK>")
        if word == "<EOS>":
            break
        if word not in ["<SOS>", "<PAD>"]:
            words.append(word)

    return " ".join(words)

def predict_caption(image_path, model, vocab, device):
    """Loads an image, generates a caption, and displays the result."""
    # Setup the evaluation transform
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    display_image = image  # Keep original for matplotlib
    
    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Generate caption
    with torch.no_grad():
        tokens = model.caption_image(image_tensor, vocab)

    caption = decode_caption(tokens, vocab)

    # Display image and caption
    plt.figure(figsize=(8, 8))
    plt.imshow(display_image)
    plt.title(caption, fontsize=14, wrap=True)
    plt.axis("off")
    plt.show()

    return caption

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Point this to the path we dynamically generated in the training script
    CHECKPOINT_PATH = os.path.expanduser("~/D7047E_Lab3_Models/BestModel.pth")
    
    # 1. Load the model and vocab
    model, vocab, device = setup_inference(CHECKPOINT_PATH)
    print("Model loaded successfully!")
    
    # 2. Test it on an image
    # Note: Replace this string with a valid path to an image on your machine 
    # if you aren't running this in the same notebook where train_imgs is defined.
    test_image_path = "/path/to/your/test/image.jpg" 
    
    if os.path.exists(test_image_path):
        generated_caption = predict_caption(test_image_path, model, vocab, device)
        print(f"Generated Output: {generated_caption}")
    else:
        print(f"Waiting for a valid image path to test! Update 'test_image_path'.")