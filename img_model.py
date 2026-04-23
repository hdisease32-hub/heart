import numpy as np
import torch
from check import ECGResNet, CLASSES, THRESHOLDS

device = torch.device('cpu')
model = ECGResNet(n_classes=5)
model.load_state_dict(torch.load('checkpoints/cardi_model.pt', map_location=device))
model.eval()

ecg = np.load('ecg_digitized.npy')
x = torch.FloatTensor(ecg.T).unsqueeze(0)

with torch.no_grad():
    import torch.nn.functional as F
    probs = torch.sigmoid(model(x)).cpu().numpy()[0]

for cls, prob in zip(CLASSES, probs):
    print(f'{cls:<8} {prob:.3f}  {"YES" if prob >= THRESHOLDS[cls] else "no"}')
