import torch, os
def save_model(model, path):
    torch.save({'model_state_dict': model.state_dict()}, path)
def load_checkpoint(caminho, model, optimizer=None):
    if not os.path.exists(caminho):
        raise FileNotFoundError(caminho)
    checkpoint = torch.load(caminho, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return checkpoint
