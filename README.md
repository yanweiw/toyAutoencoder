# Toy AutoEncoder Puzzle

This is an autoencoder where input and output are tensors of size **(1, 28, 28)**.  


`torch` needed for loading the model
```
m = model.Autoencoder()
m.load_state_dict(torch.load('weights.pth'))
```

See what you can do with it.