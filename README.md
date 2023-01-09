# Tabular-Playground-Series---Aug-2022

### 1. Specification of dependencies

using .ipynb

change paths to your `model.pt` , `train.csv` and `test.csv` in *Path* block

    MODEL_PATH = './model1.pt'
    TRAIN_PATH = './train.csv'
    TEST_PATH = './test.csv'

### 2. Training code

#### Hyperparameters :

    epoch = 200 # epoch numbers
    batch_size = 16 # batch size
    save_best = False   # save hightest val accuracy or not
#### Model:
    class Model(nn.Module):
      def __init__(self, input_shape):
          super(Model, self).__init__()
          self.fc1 = nn.Linear(input_shape, 32)
          self.fc2 = nn.Linear(32, 64)
          self.fc3 = nn.Linear(64, 1)

      def forward(self, x):
          x = torch.relu(self.fc1(x))
          x = torch.relu(self.fc2(x))
          x = torch.sigmoid(self.fc3(x))
          return x
### 3. Evaluation code

Predict result and save result to `submission.csv`

    losses = []
    accur = []
    for i in tqdm(range(NUM_EPOCHS)):
      for j, (x_train, y_train) in enumerate(train_dl):
          # print(x_train)
          #calculate output
          output = model(x_train)

          #calculate loss
          loss = loss_fn(output, y_train.reshape(-1, 1))

          #accuracy
          #backprop
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      if i % 1 == 0:
          losses.append(loss)
          print("epoch {}\tloss : {}".format(i, loss))

### 4. Pre-trained models
This is my model with private score **0.59136**

[Download Link](https://drive.google.com/file/d/10ZgUfNDuQ_1giiO8QVQcfFVSY_Ysx3Tb/view?usp=sharing)

### 5. Result
Private score **0.59136**
