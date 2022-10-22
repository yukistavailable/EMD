from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


from utils.image_processing import read_split_image
from utils.bytesIO import PickledImageProvider, bytes_to_file
from torch.utils.data import DataLoader


class ContentDatasetFromObj(data.Dataset):
    def __init__(
            self,
            obj_path,
    ):
        super(ContentDatasetFromObj, self).__init__()
        self.image_provider = PickledImageProvider(obj_path)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        item = self.image_provider.examples[index]
        ground_truth, content = self.process(item[1])
        return ground_truth, content

    def __len__(self):
        return len(self.image_provider.examples)

    def process(self, img_bytes):
        """
            process byte stream to training content_data entry
        """
        image_file = bytes_to_file(img_bytes)
        img = Image.open(image_file)
        try:
            img_A, img_B = read_split_image(img)
            # img_A = transforms.ToTensor()(img_A)
            # img_B = transforms.ToTensor()(img_B)
            # img_A = self.transform(img_A)
            # img_B = self.transform(img_B)

            img_A = self.to_tensor(img_A)
            img_B = self.to_tensor(img_B)
            c, w, h = img_B.shape
            char_num = int(h / w)
            img_B = torch.reshape(img_B, (char_num, w, w))
            return img_A, img_B

        finally:
            image_file.close()


class StyleDatasetFromObj(data.Dataset):
    def __init__(
            self,
            obj_path,
    ):
        super(StyleDatasetFromObj, self).__init__()
        self.image_provider = PickledImageProvider(obj_path)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        item = self.image_provider.examples[index]
        style = self.process(item[1])
        return style

    def __len__(self):
        return len(self.image_provider.examples)

    def process(self, img_bytes):
        """
            process byte stream to training content_data entry
        """
        image_file = bytes_to_file(img_bytes)
        img = Image.open(image_file)
        try:
            img = self.to_tensor(img)

            c, w, h = img.shape
            char_num = int(h / w)
            img = torch.reshape(img, (char_num, w, w))
            return img

        finally:
            image_file.close()
