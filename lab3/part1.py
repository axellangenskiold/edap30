#%% 0. Imports

# TODO

# %% 1. Hyperparameters

# TODO

#%% 2. Prepare the dataset

# TODO

#%% 3. Model implementation

class FlowerNet(nn.Module):
    # TODO: Implement
    pass

# %% 4. Prediction function

def predict_classes(model, x):
    # TODO: Implement
    pass

# %% 5. Evaluate

def evaluate(model, dl, labels):
    # TODO: Implement
    pass

# TODO: Test your evaluation over an unitialized model

# %% 6. Main train loop
def infinite_iter(dl):
    while True:
        for item in dl:
            yield item

# TODO: Implement your train function
def train(model, optimizer, train_dl, val_dl, num_epochs, name):
    # TODO: Implement
    pass

# %% 7. Train baseline
model = FlowerNet().to(device)

# TODO: Write training code and save your results

# %% 8. Evaluate

# TODO: Do full evaluation over all datasets

# %% 9. Attempt Image Augmentation, train for twice as long

# TODO: Write augmentation code, train and save your results.

# %% 10. Evaluate augmented model

# TOOD: Evalute on the test-set

# %% 11. Attempt another model for transfer-learning

# TODO: Implement your new model
    
# %% 12. Train your new model

# TODO: Reuse your train method and train on this new model

# %% 13. Evaluate the new model

# TODO: Evaluate your new model on the test dataset
