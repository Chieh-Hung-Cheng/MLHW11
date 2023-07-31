import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from Config import Config
from Utils import Utils
from ImageData import ImageDataset
from DomainAdverseNetwork import FeatureExtractor, CategoryClassifier, DomainClassifier


class Trainer:
    def __init__(self):
        Utils.initialization()
        # Data Related
        self.source_loader = DataLoader(ImageDataset.get_trainset(), batch_size=Config.batch_size, shuffle=True)
        self.target_loader = DataLoader(ImageDataset.get_testset(), batch_size=Config.batch_size, shuffle=True)
        # Model Related
        self.feature_extractor = FeatureExtractor().to(Config.device)
        self.category_classifier = CategoryClassifier().to(Config.device)
        self.domain_classifier = DomainClassifier().to(Config.device)
        self.feature_optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=Config.learning_rate)
        self.category_optimizer = torch.optim.Adam(self.category_classifier.parameters(), lr=Config.learning_rate)
        self.domain_optimizer = torch.optim.Adam(self.domain_classifier.parameters(), lr=Config.learning_rate)
        self.upper_criterion = torch.nn.CrossEntropyLoss()
        self.lower_criterion = torch.nn.BCEWithLogitsLoss()
        # Progress bar
        self.progress_bar = tqdm(range(Config.epochs))

    def train_loop(self):
        for epoch in self.progress_bar:
            mean_domain_loss, mean_feature_category_loss, hit_rate = self.train_one_epoch()
            self.summarize(mean_domain_loss, mean_feature_category_loss, hit_rate)

    def train_one_epoch(self):
        # Train mode
        self.feature_extractor.train()
        self.category_classifier.train()
        self.domain_classifier.train()
        # Stats related
        zip_len = 0
        total_num = 0
        total_hit = 0
        total_feature_category_loss = 0
        total_domain_loss = 0
        for (source_image_b, source_label_b), (target_image_b, _) in zip(self.source_loader, self.target_loader):
            source_image_b = source_image_b.to(Config.device)  # (N, 1, 32, 32)
            source_label_b = source_label_b.to(Config.device)  # (N, 1)
            target_image_b = target_image_b.to(Config.device)  # (N, 1, 32, 32)
            # Train Domain Classifier
            # Domain data and labels
            self.domain_classifier.zero_grad()
            data_mix_images = torch.cat([source_image_b, target_image_b], dim=0).to(Config.device)  # (2N, 1, 32, 32)
            data_mix_labels = torch.zeros([len(source_image_b) + len(target_image_b), 1]).to(Config.device)  # (2N, 1)
            data_mix_labels[:len(source_image_b), 0] = 1
            # Forward Pass
            pred_features_mix = self.feature_extractor(data_mix_images)
            pred_domain_mix = self.domain_classifier(pred_features_mix.detach())
            # Backward Pass
            loss = self.lower_criterion(pred_domain_mix, data_mix_labels)
            loss.backward()
            self.domain_optimizer.step()
            # Statistics
            total_domain_loss += loss.item()

            # Train Feature Extractor and Category Classifier
            self.feature_extractor.zero_grad()
            self.category_classifier.zero_grad()
            pred_features_source = pred_features_mix[:len(source_image_b)]
            pred_class_source = self.category_classifier(pred_features_source)
            pred_domain_mix = self.domain_classifier(pred_features_mix)
            loss = self.upper_criterion(pred_class_source, source_label_b) - Config.lamb * self.lower_criterion(
                pred_domain_mix, data_mix_labels)
            loss.backward()
            self.feature_optimizer.step()
            self.category_optimizer.step()
            total_feature_category_loss += loss.item()

            # Statistics
            total_num += len(source_image_b)
            total_hit += (pred_class_source.argmax(dim=1) == source_label_b).sum().item()
            zip_len += 1

        return total_domain_loss / zip_len, total_feature_category_loss / zip_len, total_hit / total_num


    def summarize(self, mean_domain_loss, mean_feature_category_loss, hit_rate):
        torch.save(self.feature_extractor.state_dict(), os.path.join(Config.save_path, f"feature_extractor_{Config.time_string}.ckpt"))
        torch.save(self.category_classifier.state_dict(), os.path.join(Config.save_path, f"category_classifier_{Config.time_string}.ckpt"))
        torch.save(self.domain_classifier.state_dict(), os.path.join(Config.save_path, f"domain_classifier_{Config.time_string}.ckpt"))
        self.progress_bar.set_postfix({"train_domain_loss": f"{mean_domain_loss:.4f}",
                                     "train_feature_category_loss": f"{mean_feature_category_loss:.4f}",
                                     "train_accuracy": f"{hit_rate:.2%}",})


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_loop()

