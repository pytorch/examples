import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.models import vgg16
import torchvision.transforms as transforms

# Upoading reference and target image
reference_image = Image.open('ref1-0.jpg').convert('RGB')
target_image = Image.open('tar2-0.jpg').convert('RGB')

# Converting images to Tensor format
reference_tensor = ToTensor()(reference_image)
target_tensor = ToTensor()(target_image)

# Check the size of the images and resize them if necessary
if reference_tensor.dim() != 3 or target_tensor.dim() != 3:
    print("Image sizes are mismatched. Use RGB images.")
    exit()

if reference_tensor.shape != target_tensor.shape:
    print("Reference and target images must be equal in size.")
    exit()

# Reshape dimensions
reference_tensor = reference_tensor.unsqueeze(0)  # Add batch size
target_tensor = target_tensor.unsqueeze(0)

# Moving image tensors to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reference_tensor = reference_tensor.to(device)
target_tensor = target_tensor.to(device)

# Loading the VGG16 model
model = vgg16(pretrained=True)
model = model.to(device)

# Processing images appropriately for the VGG16 model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to VGG16 input size
    transforms.ToTensor(),  # Converting tensor format
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing
])

reference_tensor = preprocess(reference_image).unsqueeze(0).to(device)
target_tensor = preprocess(target_image).unsqueeze(0).to(device)

# Similarity Calculating Function
def calculate_similarity(reference_tensor, target_tensor):
    # Feature maps of reference and target images
    reference_features = model.features(reference_tensor)
    target_features = model.features(target_tensor)

    # Similarity Calculating
    similarity = F.cosine_similarity(reference_features, target_features).mean()
    
    return similarity

# Calculating of Similarity
similarity_score = calculate_similarity(reference_tensor, target_tensor)

# Similarity percenatage
similarity_percentage = similarity_score.item() * 100

print(f"Benzerlik Skoru: {similarity_percentage:.2f}%")
