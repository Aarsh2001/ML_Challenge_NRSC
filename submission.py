import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms as T

#cuda 
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_final_mask(model, image , mean=[0.485, 0.456, 0.406],
                       std = [0.229 , 0.224 ,0.225]):
  model.eval()
  t= T.Compose([T.ToTensor() ,T.Normalize(mean, std)])
  image = t(image)
  model.to(device) ; image = image.to(device)
  with torch.no_grad():

    image = image.unsqueeze(0)

    output = model(image)
    masked = torch.argmax(output , dim =1)
    masked = masked.cpu().squeeze(0).numpy()
  return masked



model = torch.load('./model_ARG_BEST.pth')

img = cv2.imread('./test_data/test/mosaic_test.jpg')
i=750
j=750
index = 1

final_output = np.zeros((3750,3750))

while i <= 3750:
    while j <=3750:
        cropped_image = img[i-750:i, j-750:j]
        im = cv2.cvtColor(cropped_image , cv2.COLOR_BGR2RGB)
        kernel_sharp = np.array(([-2, -2, -2], [-2, 17, -2], [-2, -2, -2]), dtype='int')
        im = cv2.filter2D(im, -1, kernel_sharp)
        im = cv2.resize(im, (512 , 512) , interpolation = cv2.INTER_NEAREST)

        masked_image = predict_final_mask(model,im)
        path = os.path.join('./output/','%s.png'%index)
        masked_image = cv2.resize(masked_image,(750,750),interpolation = cv2.INTER_NEAREST)
        final_output[i-750:i, j-750:j] = masked_image
        j=j+750
        index=index+1
    i=i+750
    j=750
plt.figure(figsize = (15,12) )
plt.imshow(final_output)
cv2.imwrite('./output/output.png',final_output)
np.save('./output/out_imds.npy',final_output)
