import numpy as np
import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score)
from sklearn.preprocessing import OneHotEncoder
from torchattacks import (
    FGSM, BIM, CW, RFGSM, PGDL2,
    TPGD, FAB, DeepFool, SparseFool)
from torchvision import datasets, transforms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# to make the results are consistent and reproducible
torch.manual_seed(123)


def load_image(train_path: str, test_path: str):
    """
    This function is to transform the images
    through imported 'transform' function.
    It return tranformed images of data train and test
    """

    # It can be customized as well
    train_transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()])

    return (
        datasets.ImageFolder(train_path, transform=train_transforms),
        datasets.ImageFolder(test_path, transform=train_transforms))


def sampling(train_path: str, test_path: str):
    """
    This function is to perform a sampling
    technique from related dataset.
    """

    # Load image to train_set & test_set variable
    train_set, test_set = load_image(
        train_path=train_path,
        test_path=test_path)

    num_train = len(train_set)
    indices_train = list(range(num_train))
    np.random.shuffle(indices_train)

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices_train)

    # Shuffle the test set
    num_test = len(test_set)
    indices_test = list(range(num_test))
    np.random.shuffle(indices_test)

    # Normalized the value of test data
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

    return train_set, test_set, train_sampler, test_sampler


def image_processing(
        train_path: str,
        test_path: str,
        batch_train: int,
        batch_test: int):
    """
    This function is a final preprocessing step
    it call one previous function to do a sampling technique.
    It return preprocessed data which is
    ready for model training.
    """

    # Put sampled data to several variables
    # by calling 'sampling' function as well
    train_set, test_set, train_sampler, test_sampler = sampling(
        train_path=train_path,
        test_path=test_path)

    return (
        torch.utils.data.DataLoader(
            train_set, sampler=train_sampler, batch_size=batch_train),
        torch.utils.data.DataLoader(
            test_set, sampler=test_sampler, batch_size=batch_test))


def cnn_model(torch_model):
    """
    This function is to declare the CNN model
    and add customized layer on a top of CNN model.
    It return customized model, loss function, and optimizer
    """

    # Assign the torch_model, get model name
    # and set the nodes value for specific model
    model = torch_model
    nodes = 1024

    # Architecture from top layer of the CNN
    classifier = torch.nn.Sequential(
        torch.nn.Linear(nodes, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 2),
        torch.nn.LogSoftmax(dim=1))

    for name, param in model.named_parameters():
        param.requires_grad = True if name.startswith('fc') else False

    model.fc = classifier
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.003)

    # Set loss function and optimizer
    error = torch.nn.CrossEntropyLoss()

    try:
        model.cuda()
    except:
        pass

    return model, error, optimizer


def model_training(
    batch_size: int,
    data_train,
    num_epochs: int,
    model,
    loss_func,
    optimizer):
    
    """
    To train the CNN models.
    It rerturning a trained model and loss values during training.
    """

    global device
    Train_loss = []

    # Start the epoch iteration
    for epoch in range(num_epochs):
        train_loss = 0.
        model.train()

        # Traning model per batch image
        for images, labels in data_train:
            images, labels = images.to(device), labels.to(device)
            train = images.view(-1, 3, 100, 100)
            outputs = model(train)

            optimizer.zero_grad()
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size

        # Show the loss value per epoch
        print("Epoch: {}, Loss: {:.4f}".
              format(
                  epoch + 1,
                  train_loss / len(data_train.dataset)))

        # Add the loss value per epoch to main list
        Train_loss.append(train_loss / len(data_train.dataset))

    return model, Train_loss


def model_testing(data_test, model):
    """
    To test the performance of model.
    It return confusion matrix, classification report
    and area under curve score.
    """

    global device
    actual, predict, predicton_label_loader = [], [], []
    model.eval()

    with torch.no_grad():
        # Evaluate trained model on the test data
        for images, labels in data_test:
            images, labels = images.to(device), labels.to(device)
            actual.extend(labels.data.tolist())

            test = images.view(-1, 3, 100, 100)
            outputs = model(test)
            predicted = torch.max(outputs, 1)[1]
            predict.extend(predicted.data.tolist())
            predicton_label_loader.append(predicted)

    # Do one hot encoding to get AUC score
    encoder = OneHotEncoder(sparse=False)
    actual_encoded, predict_encoded = np.array(actual), np.array(predict)
    actual_encoded, predict_encoded = actual_encoded.reshape(
        len(actual_encoded), 1), predict_encoded.reshape(
        len(predict_encoded), 1)
    actual_encoded, predict_encoded = encoder.fit_transform(
        actual_encoded), encoder.fit_transform(predict_encoded)

    return (
        predicton_label_loader,
        confusion_matrix(actual, predict),
        classification_report(actual, predict),
        roc_auc_score(
            actual_encoded,
            predict_encoded,
            multi_class='ovo',
            average='macro'))


def model_attacking(data_test, model):
    """
    This function is to attack the trained model
    by testing it to peturbated test data.
    """

    # Iterate over adversarial attack methods
    for method in FGSM, BIM, CW, RFGSM, PGDL2, TPGD, DeepFool:
        correct, total = 0, 0
        atk = method(model)

        # Attack the model
        for images, labels in data_test:
            images = atk(images, labels).cuda()
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum()

        # Show the accuracy of attacked model
        print("Method used :", method,
              '\nRobust accuracy :', (100 * float(correct) / total))

def adversarial_training(
    batch_size: int,
    data_train,
    num_epochs: int,
    model,
    loss_func,
    optimizer,
    adv_method):
    
    """
    To train the CNN models.
    It rerturning a trained model and loss values during training.
    """

    global device
    Train_loss = []
    atk = adv_method(model)

    # Start the epoch iteration
    for epoch in range(num_epochs):
        train_loss = 0.
        model.train()

        # Traning model per batch image
        for images, labels in data_train:
            images= atk(images, labels).cuda()
            labels = labels.to(device)
            train = images.view(-1, 3, 100, 100)
            outputs = model(train)

            optimizer.zero_grad()
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size

        # Show the loss value per epoch
        print("Epoch: {}, Loss: {:.4f}".
              format(
                  epoch + 1,
                  train_loss / len(data_train.dataset)))

        # Add the loss value per epoch to main list
        Train_loss.append(train_loss / len(data_train.dataset))

    return model, Train_loss