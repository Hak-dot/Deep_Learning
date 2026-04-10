import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# --- 1. ĐỊNH NGHĨA LẠI CẤU TRÚC MODEL (Bắt buộc phải giống lúc train) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):   
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# --- 2. HÀM LOAD MODEL ---
@st.cache_resource # Giúp load model 1 lần duy nhất để tiết kiệm RAM
def load_my_model():
    model = SimpleCNN()
    # Load trọng số từ file .pth của bạn
    model.load_state_dict(torch.load('simple_cnn_cats_dogs.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# --- 3. GIAO DIỆN WEB ---
st.title("Ứng dụng nhận diện Chó & Mèo")
st.write("Tải một tấm ảnh lên để xem mô hình dự đoán là gì nhé!")

uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh vừa upload
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
    st.write("")
    st.write("Đang nhận diện...")

    # Tiền xử lý ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)

    # Dự đoán
    model = load_my_model()
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        
    # Xuất kết quả
    classes = ['Mèo 🐱', 'Chó 🐶']
    result = classes[predicted.item()]
    
    st.success(f"Dự đoán: **{result}**")