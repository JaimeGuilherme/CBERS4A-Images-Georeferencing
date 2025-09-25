import torch, os

def save_model(model, path, optimizer=None, epoch=None,
               best_iou=None, best_threshold=None, best_val_loss=None):
    checkpoint = {
        'model_state_dict': model.state_dict()
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if best_iou is not None:
        checkpoint['best_iou'] = best_iou
    if best_threshold is not None:
        checkpoint['best_threshold'] = best_threshold
    if best_val_loss is not None:
        checkpoint['best_val_loss'] = best_val_loss

    torch.save(checkpoint, path)


def load_checkpoint(caminho, model, optimizer=None):
    if not os.path.exists(caminho):
        raise FileNotFoundError(caminho)
    checkpoint = torch.load(caminho, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def load_checkpoint_raw(caminho, map_location='cpu'):
    if not os.path.exists(caminho):
        raise FileNotFoundError(caminho)
    return torch.load(caminho, map_location=map_location, weights_only=False)
