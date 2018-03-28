def get_preprocessed_img(img_path, image_size):
  import cv2
  img = cv2.imread(img_path)
  img = cv2.resize(img, (image_size, image_size))
  img = img.astype('float32')
  img /= 255
  return img

def show(image):
  import cv2
  cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
  cv2.imshow("view", image)
  cv2.waitKey(0)
  cv2.destroyWindow("view")
  
def one_hot(indices, depth):
  import numpy as np
  result = []
  for indice in indices:
    temp = np.zeros(depth)
    temp[indice] = 1
    result.append(temp)
  return result
    