# evaluate.py
import torch
from models.model1 import Model1
from data.dataset_loader import get_dataloader

def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch['inputs'])
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == batch['labels']).sum().item()
            total_samples += batch['labels'].size(0)

    accuracy = total_correct / total_samples
    print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    # Load model
    model = Model1(input_dim=28, output_dim=10)
    model.load_state_dict(torch.load('experiments/model1_exp/checkpoint.pth'))

    # Load test data
    test_loader = get_dataloader('data/test_dataset/', batch_size=32)

    # Evaluate
    evaluate(model, test_loader)