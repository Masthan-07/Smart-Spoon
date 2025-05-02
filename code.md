import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    img = Image.open(image_path)
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
    return output.argmax(dim=1).item()
  import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Simulated user preferences
data = {
    "salt_level": [0.1, 0.2, 0.4, 0.6, 0.9],
    "umami_level": [0.3, 0.5, 0.7, 0.2, 0.8]
}
df = pd.DataFrame(data)

kmeans = KMeans(n_clusters=2)
df['cluster'] = kmeans.fit_predict(df[["salt_level", "umami_level"]])

plt.scatter(df.salt_level, df.umami_level, c=df.cluster)
plt.xlabel("Salt Simulation Intensity")
plt.ylabel("Umami Simulation Intensity")
plt.title("Taste Profile Clustering")
plt.show()
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
example = "This spoon really helps with my salt-free diet!"
print(classifier(example))
