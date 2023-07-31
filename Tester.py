import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from Config import Config
from Utils import Utils
from ImageData import ImageDataset
from DomainAdverseNetwork import FeatureExtractor, CategoryClassifier

class Tester():
    def __init__(self):
        Utils.initialization()
        # Data Related
        self.test_loader = DataLoader(ImageDataset.get_testset(), batch_size=128, shuffle=False)
        # model related
        self.feature_extractor = FeatureExtractor().to(Config.device)
        self.category_classifier = CategoryClassifier().to(Config.device)
        load_name = "04090801"
        self.feature_extractor.load_state_dict(torch.load(os.path.join(Config.save_path, f"feature_extractor_{load_name}.ckpt")))
        self.category_classifier.load_state_dict(torch.load(os.path.join(Config.save_path, f"category_classifier_{load_name}.ckpt")))

    def infer(self):
        with torch.no_grad():
            prediction = []
            for image_b, _ in tqdm(self.test_loader):
                image_b = image_b.to(Config.device)
                features_b = self.feature_extractor(image_b)
                category_b = self.category_classifier(features_b)
                label_b = torch.argmax(category_b, dim=1)
                prediction.append(label_b)
            prediction = torch.cat(prediction, dim=0)
            self.save_prediction_to_csv(prediction)

    def save_prediction_to_csv(self, prediction):
        with open(os.path.join(Config.output_path, f"prediction_{Config.time_string}.csv"), "w") as f:
            f.write("id,label\n")
            for i, label in enumerate(prediction):
                f.write(f"{i},{label}\n")

if __name__ == "__main__":
    tester = Tester()
    tester.infer()
