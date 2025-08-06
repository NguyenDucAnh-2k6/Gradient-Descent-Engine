import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  
y = 4 + 3 * X + np.random.randn(100, 1)
def sgd(X,y, learning_rate=0.1, epochs=300, batch_size=1):
    m=len(X)
    theta=np.random.randn(2,1)
    '''one=np.ones((X.shape[0],1))
    X_bias=np.concatenate((one, X), axis=1)'''
    X_bias = np.c_[np.ones((m, 1)), X]
    loss_history=[]
    for epoch in range(epochs):
        indices=np.random.permutation(m)
        X_shuffled=X_bias[indices]
        y_shuffled=y[indices]
        for i in range(0,m,batch_size):
            X_batch=X_shuffled[i:i+batch_size]
            y_batch=y_shuffled[i:i+batch_size]
            gradients=2/batch_size*X_batch.T.dot(X_batch.dot(theta)-y_batch)
            theta-=learning_rate*gradients
        predictions=X_bias.dot(theta)
        loss=np.mean((predictions-y)**2)
        loss_history.append(loss)
        if epoch%10==0:
            print(f"Epoch {epoch}, loss={loss:.6f}")
    return loss_history, theta
loss_history, theta=sgd(X,y,0.1,1000, 1)
print(theta)
plt.plot(loss_history)
plt.title('Training loss on epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(np.arange(0, len(loss_history), 100))
plt.show()




