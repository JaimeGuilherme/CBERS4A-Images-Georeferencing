import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from glob import glob

from components.dataset import RoadIntersectionDataset
from components.unet import UNet
from components.metrics import calculate_metrics
from components.utils import save_model, load_checkpoint
from components.losses import FocalLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def buscar_melhor_threshold(model, val_loader, device, thresholds=np.linspace(0.1, 0.9, 17)):
    model.eval()
    best_t = 0.5
    best_iou = -1
    sigmoid = torch.nn.Sigmoid()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Coletando predições", leave=False):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(masks.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    for t in thresholds:
        bin_preds = (all_preds > t).astype(np.uint8)
        iou, _, _ = calculate_metrics(bin_preds, all_targets)
        if iou > best_iou:
            best_iou = iou
            best_t = t

    return best_t, best_iou


def avaliar_modelo(model, dataloader, device, threshold=0.5):
    model.eval()
    ious, precisions, recalls = [], [], []

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Avaliando", leave=False)
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device).float()

            outputs = torch.sigmoid(model(images))
            preds = (outputs > threshold).int()

            iou, precision, recall = calculate_metrics(preds, masks)
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)

    return np.mean(ious), np.mean(precisions), np.mean(recalls)


def encontrar_ultimo_checkpoint(pasta_checkpoints):
    arquivos = glob(os.path.join(pasta_checkpoints, "checkpoint_epoch_*.pth"))
    if not arquivos:
        return None
    arquivos.sort(key=os.path.getmtime)
    return arquivos[-1]


if __name__ == "__main__":
    caminho_train_img = "dataset_separated/train/images"
    caminho_train_mask = "dataset_separated/train/masks"

    caminho_val_img = "dataset_separated/val/images"
    caminho_val_mask = "dataset_separated/val/masks"

    caminho_test_img = "dataset_separated/test/images"
    caminho_test_mask = "dataset_separated/test/masks"

    caminho_checkpoint = "checkpoints"
    epocas = 2
    batch = 5
    lr = 1e-4
    salvar_checkpoint_a_cada = 5

    os.makedirs(caminho_checkpoint, exist_ok=True)

    train_dataset = RoadIntersectionDataset(caminho_train_img, caminho_train_mask)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    val_loader = None
    if os.path.exists(caminho_val_img) and os.path.exists(caminho_val_mask):
        val_dataset = RoadIntersectionDataset(caminho_val_img, caminho_val_mask)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    test_loader = None
    if os.path.exists(caminho_test_img) and os.path.exists(caminho_test_mask):
        test_dataset = RoadIntersectionDataset(caminho_test_img, caminho_test_mask)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    focal_loss = FocalLoss(alpha=0.8, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_iou = 0.0
    best_threshold = 0.5
    start_epoch = 0

    ultimo_ckpt = encontrar_ultimo_checkpoint(caminho_checkpoint)
    if ultimo_ckpt:
        print(f"🔄 Encontrado checkpoint para retomar: {ultimo_ckpt}")
        checkpoint_data = load_checkpoint(ultimo_ckpt, model, optimizer)
        start_epoch = checkpoint_data['epoch'] + 1
        best_iou = checkpoint_data.get('best_iou', 0.0)
        best_threshold = checkpoint_data.get('best_threshold', 0.5)
    else:
        print("🚀 Nenhum checkpoint encontrado. Iniciando treinamento do zero.")

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    ema_val_loss = None
    alpha = 0.3

    writer = SummaryWriter(log_dir="runs/Deteccao_Cruzamento_Rodovias")
    writer.add_text("Hiperparametros", 
                    f"epochs={epocas}, batch_size={batch}, lr={lr}, "
                    f"loss=FocalLoss(alpha=0.8, gamma=2), model=U-Net, "
                    f"early_stopping=EMA(alpha={alpha}, patience={patience})")

    stopped_epoch = None

    for epoch in range(start_epoch, epocas):
        print(f"\nEpoch {epoch+1}/{epocas}")
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc="Treinando", leave=False)
        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).float()

            outputs = model(images)
            loss = focal_loss(outputs, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        print(f"Loss médio treino: {avg_train_loss:.4f}")
        writer.add_scalar("Loss/treino", avg_train_loss, epoch)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(DEVICE)
                    masks = masks.to(DEVICE).float()
                    outputs = model(images)
                    loss = focal_loss(outputs, masks)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f"Loss médio validação: {val_loss:.4f}")
            writer.add_scalar("Loss/validacao", val_loss, epoch)

            if ema_val_loss is None:
                ema_val_loss = val_loss
            else:
                ema_val_loss = alpha * val_loss + (1 - alpha) * ema_val_loss

            writer.add_scalar("Loss/val_EMA", ema_val_loss, epoch)

            if ema_val_loss > best_val_loss:
                patience_counter += 1
                print(f"⚠️ Val_loss(EMA) piorou ({ema_val_loss:.4f} > {best_val_loss:.4f}) "
                      f"| patience {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("🛑 Parando treino antecipadamente: perda suavizada de validação começou a subir.")
                    stopped_epoch = epoch + 1
                    writer.add_scalar("EarlyStopping/Stopped_epoch", stopped_epoch, epoch)
                    break
            else:
                patience_counter = 0
                best_val_loss = ema_val_loss

            best_t_epoca, iou_val_threshold_search = buscar_melhor_threshold(model, val_loader, DEVICE)
            iou_val, prec_val, rec_val = avaliar_modelo(model, val_loader, DEVICE, threshold=best_t_epoca)
            print(f"Melhor threshold: {best_t_epoca:.2f} | IoU: {iou_val:.4f} | Precisão: {prec_val:.4f} | Recall: {rec_val:.4f}")

            writer.add_scalar("Metricas/val_iou", iou_val, epoch)
            writer.add_scalar("Metricas/val_precision", prec_val, epoch)
            writer.add_scalar("Metricas/val_recall", rec_val, epoch)
            writer.add_scalar("Metricas/val_best_threshold", best_t_epoca, epoch)

            if (epoch + 1) % salvar_checkpoint_a_cada == 0:
                checkpoint_path = os.path.join(caminho_checkpoint, f"checkpoint_epoch_{epoch+1}.pth")
                save_model(model, checkpoint_path, optimizer=optimizer, epoch=epoch,
                           best_iou=best_iou, best_threshold=best_t_epoca)
                print(f"💾 Checkpoint salvo em {checkpoint_path}")

            if val_loss < best_val_loss:
                best_val_loss = ema_val_loss
                best_threshold = best_t_epoca
                best_model_path = os.path.join(caminho_checkpoint, "best_model.pth")
                save_model(model, best_model_path, optimizer=optimizer, epoch=epoch,
                           best_iou=iou_val, best_threshold=best_threshold)
                print(f"🔁 Novo melhor modelo salvo em {best_model_path}")

    print("\n✅ Treinamento finalizado.")

    if stopped_epoch is not None:
        print(f"📌 Early stopping ativado na época {stopped_epoch}.")

    if test_loader is not None:
        print("\n🔍 Avaliando modelo no conjunto de TESTE...")
        iou_test, prec_test, rec_test = avaliar_modelo(model, test_loader, DEVICE, threshold=best_threshold)
        print(f"Test IoU: {iou_test:.4f} | Precisão: {prec_test:.4f} | Recall: {rec_test:.4f}")

        writer.add_scalar("Metricas/test_iou", iou_test)
        writer.add_scalar("Metricas/test_precision", prec_test)
        writer.add_scalar("Metricas/test_recall", rec_test)

    writer.close()
