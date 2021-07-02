from PIL import Image
from torchvision import transforms
def image_process(image_path, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
  img = Image.open(image_path).convert('RGB')

  if keep_ratio:
    w, h = img.size
    ratio = w / float(h)
    imgW = int(np.floor(ratio * imgH))
    imgW = max(imgH * min_ratio, imgW)

  img = img.resize((imgW, imgH), Image.BILINEAR)
  img = transforms.ToTensor()(img)
  img.sub_(0.5).div_(0.5)

  return img