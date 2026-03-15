# CSNets
Neural network implementation from scratch in C#.

## What's Implemented

- **Networks**: Feedforward (fully connected), configurable depth and width
- **Layers**: Dense layer with batching support
- **Neurons**: Standard neuron, momentum neuron (SGD + momentum)
- **Activations**: Sigmoid, ReLU *(Tanh stub)*
- **Loss**: Mean Squared Error, *(Cross-Entropy stub)*
- **Initializations**: He, Random uniform
- **Training**: Single-sample SGD, mini-batch gradient accumulation
- **Data**: MNIST loader, function approximation project

---

## Roadmap

### Activations & Loss
- [ ] Tanh activation
- [ ] Cross-Entropy loss
- [ ] Softmax output layer

### Optimizers
- [ ] Adam optimizer
- [ ] RMSProp
- [ ] Learning rate scheduling (decay, warmup)

### Regularization
- [ ] L2 weight decay
- [ ] Dropout layer
- [ ] Batch normalization

### Architecture
- [ ] Convolutional layers (Conv2D, pooling)
- [ ] Recurrent layer (RNN / LSTM)
- [ ] Residual / skip connections

### Initializations
- [ ] Xavier / Glorot initialization
- [ ] Orthogonal initialization

### Training & Evaluation
- [ ] Validation split and early stopping
- [ ] Accuracy / F1 metrics
- [ ] Model save/load (weights serialization)

### Tooling
- [ ] Training progress logging / loss curves
- [ ] Configurable network builder (JSON/YAML config)
