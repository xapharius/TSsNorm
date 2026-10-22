# Integrations

Each folder runs σN-Ens inside a training framework. The framework is used as a library. None of
its files change.

| folder | framework | experiments |
|--------|-----------|-------------|
| [`torch_uncertainty/`](torch_uncertainty/) | [TorchUncertainty](https://github.com/torch-uncertainty/torch-uncertainty) 0.7 | CNNs on CIFAR-10 and CIFAR-100 |
| [`torch_uncertainty_oodeval/`](torch_uncertainty_oodeval/) | TorchUncertainty's OodEval branch | ViTs on CIFAR-100 and ImageNet, with OpenOOD |

Each folder has its own `requirements.txt`. Install each in its own environment.

```bash
cd torch_uncertainty
python train.py fit --config configs/cifar10/resnet18.yaml --data.root ./data
```
