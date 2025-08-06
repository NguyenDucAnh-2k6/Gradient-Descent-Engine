import numpy as np
import matplotlib.pyplot as plt
class LinearRegressionSGD:
    def __init__(self, learning_rate=0.01, epochs=100, verbose=False, loss_history=[]):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.loss_history=loss_history
        self.w = None  # Weight vector
    def fit(self, X, y):
        N, d = X.shape
        self.w = np.zeros((d, 1))  # Initialize weights to zeros
        for epoch in range(self.epochs):
            for i in range(N):
                xi = X[i].reshape(1, -1)  # Shape: (1, d)
                yi = y[i].reshape(1, -1)  # Shape: (1, 1)

                # Prediction
                y_pred = xi @ self.w

                # Gradient computation
                grad = xi.T @ (y_pred - yi)

                # Weight update
                self.w -= self.learning_rate * grad
            loss = np.mean((X @ self.w - y) ** 2)
            self.loss_history.append(loss)
            if self.verbose and epoch % 10 == 0:
                
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
            indices=np.arange(X.shape[0])
            np.random.shuffle(indices)
            X=X[indices]
            y=y[indices]
            noise = 0.01 * np.random.randn(*self.w.shape)
            self.w += noise  # small random noise to weights
            if epoch%30==0:
                self.w+=np.random.randn(*self.w.shape)
        return self.loss_history

    def predict(self, X):
        return X @ self.w

    def get_weights(self):
        return self.w

# === Data Generation ===
N = 200
X = np.random.rand(N, 1)
y = 1 + 3.5 * X + 0.2 * np.random.rand(N, 1)

# Add bias term
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

# === Model Training ===
model = LinearRegressionSGD(learning_rate=0.05, epochs=1000, verbose=True)
loss_history=model.fit(Xbar, y)

# === Results ===
w_learned = model.get_weights()
print("Learned weights:\n", w_learned)
plt.plot(loss_history)
plt.title('Loss function on epochs')
plt.xticks(np.arange(0, len(loss_history), 50))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
