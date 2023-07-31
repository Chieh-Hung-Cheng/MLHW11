import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class Config:
    # Time & Randomness
    time_string = None
    seed = 48763

    # Paths
    base_path = None
    data_path = r"D:\ML_Dataset\HW11"
    save_path = None
    output_path = None

    # Load Models
    load_ckpt = False
    load_name = None

    # Dataset / DataLoader
    train_loader = None
    valid_loader = None
    test_loader = None
    valid_ratio = 0.1
    num_worker = 4

    # Training Related
    learning_rate = 1e-3
    epochs = 300
    batch_size = 32
    early_stop = 50
    valid_cycle = 2
    warmup_steps = 10
    lamb = 0.4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    optimizer = None
    scheduler = None
    criterion = None



if __name__ == "__main__":
    print(Config.seed)
