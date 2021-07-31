import cv2
import numpy as np
import os
import torch
from torchvision import transforms as T

#cuda
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):
  """
  Fuction to load the model

  Input :- Path of the model 

  Returns:- Torch Model
  """
  model = torch.load(path)
  return model

def predict_mask(model, image , mean=[0.485, 0.456, 0.406],std = [0.229 , 0.224 ,0.225]):
  """
  Function to predict mask of the given image 

  INPUT :-
  Model :- model file you want to load for making prediction
  image :- image to mask
  mean,std(optional) :- To normalize image
  
  Returns :- predicted mask numpy array
  """
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


def final_mask(img_path,model):
  """
  Fuction to combine all the predicted masks

  Input :- 
  img_path :- path of image on which you wanna make prediction
  model :- torch model

  Returns:- None

  Saves output at ./out/
  """
  img = cv2.imread(img_path)
  i=750
  j=750
  index = 1
  final_output = np.zeros((3750,3750))
  #loop through images
  while i <= 3750:
      while j <=3750:
          cropped_image = img[i-750:i, j-750:j]
          im = cv2.cvtColor(cropped_image , cv2.COLOR_BGR2RGB)
          kernel_sharp = np.array(([-2, -2, -2], [-2, 17, -2], [-2, -2, -2]), dtype='int')
          im = cv2.filter2D(im, -1, kernel_sharp)
          im = cv2.resize(im, (512 , 512) , interpolation = cv2.INTER_NEAREST)

          masked_image = predict_mask(model,im)
          path = os.path.join('./output/','%s.png'%index)
          masked_image = cv2.resize(masked_image,(750,750),interpolation = cv2.INTER_NEAREST)
          final_output[i-750:i, j-750:j] = masked_image
          j=j+750
          index=index+1
      i=i+750
      j=750
  #saving output 
  np.save('./submission_output/out_imgds.npy',final_output)
  final_output = final_output*255
  cv2.imwrite('./submission_output/output.png',final_output)


if __name__ == "__main__":
  model = load_model('./model.pth')
  final_mask('./test_data/mosaic_test.jpg',model)


