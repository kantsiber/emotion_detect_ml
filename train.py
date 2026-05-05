import os
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.EmoModel_base import EmoModel
from datasets.AudioDataset import AudioDataset
from torch.utils.tensorboard import SummaryWriter
from utils.utils import parse_args, read_config, copy_json
import shutil
from pathlib import Path


args = parse_args()
config = read_config(args.config_path)
os.makedirs(config.check_points_path + f"/model_check_points/", exist_ok=False)
copy_json(args.config_path, config.check_points_path + "/config.json")
current_file = Path(__file__).resolve()
destination = Path(config.check_points_path + '/train.py')
shutil.copy2(current_file, destination)

writer = SummaryWriter(log_dir=config.check_points_path + "/logs")

train_dataset = AudioDataset(
    csv_file=config.train_csv_path,
    prefix_path=config.prefix_for_file_path,
    target_sample_rate=config.target_sample_rate
)
val_dataset = AudioDataset(
    csv_file=config.test_csv_path,
    prefix_path=config.test_prefix_for_file_path,
    target_sample_rate=config.target_sample_rate
)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmoModel(num_classes=config.num_classes)
model = model.to(device)

# loss_fun = nn.MSELoss()
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)
epochs = 10_000

for epoch in tqdm(range(1, epochs), desc="Обучение модели"):
    model.train()
    run_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    print("Обучаем модель")
    for batch_idx, (mel_specs, labels) in tqdm(enumerate(train_dataloader)):
        mel_specs = mel_specs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(mel_specs)
        loss = loss_fun(outputs, labels)
        loss.backward()
        optimizer.step()

        run_loss += loss.item()

        pred = outputs.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if batch_idx % 10 == 0:
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('Accuracy/train_batch', (pred == labels).sum().item() / labels.size(0),
                              epoch * len(train_dataloader) + batch_idx)

    model.eval()
    val_run_loss = 0.0
    val_correct = 0
    val_total = 0

    val_all_preds = []
    val_all_labels = []

    print("Валидируем модель")
    with torch.no_grad():
        for batch_idx, (mel_specs, labels) in tqdm(enumerate(val_dataloader)):
            mel_specs = mel_specs.to(device)
            labels = labels.to(device)
            outputs = model(mel_specs)
            loss = loss_fun(outputs, labels)

            val_run_loss += loss.item()

            pred = outputs.argmax(1)
            val_correct += (pred == labels).sum().item()
            val_total += labels.size(0)

            val_all_preds.extend(pred.cpu().numpy())
            val_all_labels.extend(labels.cpu().numpy())

    avg_loss = run_loss / len(train_dataloader)
    accuracy = correct / total * 100

    val_avg_loss = val_run_loss / len(val_dataloader)
    val_accuracy = val_correct / val_total * 100

    train_conf_matrix = confusion_matrix(all_labels, all_preds)
    val_conf_matrix = confusion_matrix(val_all_labels, val_all_preds)

    train_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    train_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    val_precision = precision_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)
    val_recall = recall_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)

    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Loss/val', val_avg_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    writer.add_scalar('Precision/train', train_precision, epoch)
    writer.add_scalar('Precision/val', val_precision, epoch)
    writer.add_scalar('Recall/train', train_recall, epoch)
    writer.add_scalar('Recall/val', val_recall, epoch)

    tqdm.write(f'Эпохи: {epoch}')
    tqdm.write(f'Тренировка: Средний Train Loss: {avg_loss}, Train Точность: {accuracy:.2f}%')
    tqdm.write(f'Валидация: Средний Val Loss: {val_avg_loss}, Val Точность {val_accuracy:.2f}%')
    tqdm.write(f'Матрица ошибок train: {train_conf_matrix}')
    tqdm.write(f'Train Recall: {train_recall:.2f}, Train Precision: {train_precision:.2f}')
    tqdm.write(f'Матрица ошибок val: {val_conf_matrix}')
    tqdm.write(f'Val Recall: {val_recall:.2f}, Val Precision: {val_precision:.2f}')

    epoch_num = epoch
    if epoch_num % 1 == 0:
        torch.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'accuracy': accuracy,
            'val_loss': val_avg_loss,
            'val_accuracy': val_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'val_precision': val_precision,
            'val_recall': val_recall,
        }, config.check_points_path + f"/model_check_points/check_point_{epoch_num}.pth")

writer.close()