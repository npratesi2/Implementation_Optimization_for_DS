import numpy as np

class NeuralNetwork:
    """
    Shallow Multi-Layer Perceptron (MLP) for multi-class classification
    using Mean Squared Error (MSE) loss and Lasso (L1) regularization.
    
    Architecture:
      - Input dimension: d_in
      - Hidden layer: m neurons with hyperbolic tangent (tanh) activation
      - Output layer: K neurons with linear activation
      - Regularization: L1-norm penalty lambda * ||theta||_1
    """

    def __init__(self, d_in: int, m: int, K: int, lam: float = 1e-3):
        """
        Initialize the neural network architecture dimensions and regularization parameter.

        Parameters:
            d_in (int): Number of input features.
            m (int): Number of hidden units.
            K (int): Number of output classes.
            lam (float): L1 regularization hyperparameter.
        """
        self.d_in = d_in
        self.m = m
        self.K = K
        self.lam = lam
        
        # Total number of trainable parameters: n = m*(d_in + 1) + K*(m + 1)
        self.n_params = self.m * (self.d_in + 1) + self.K * (self.m + 1)

    def unpack_parameters(self, theta: np.ndarray):
        """
        Unpacks the 1D flat parameter vector theta into weight matrices and bias vectors.

        Parameters:
            theta (np.ndarray): 1D array of shape (n_params,).

        Returns:
            tuple: (W1, b1, W2, b2)
                   W1 of shape (m, d_in)
                   b1 of shape (m,)
                   W2 of shape (K, m)
                   b2 of shape (K,)
        """
        idx = 0
        
        # Input-to-hidden weights W1: shape (m, d_in)
        W1 = theta[idx : idx + self.m * self.d_in].reshape(self.m, self.d_in)
        idx += self.m * self.d_in
        
        # Hidden layer bias b1: shape (m,)
        b1 = theta[idx : idx + self.m]
        idx += self.m
        
        # Hidden-to-output weights W2: shape (K, m)
        W2 = theta[idx : idx + self.K * self.m].reshape(self.K, self.m)
        idx += self.K * self.m
        
        # Output layer bias b2: shape (K,)
        b2 = theta[idx : idx + self.K]
        
        return W1, b1, W2, b2

    def forward(self, theta: np.ndarray, X: np.ndarray):
        """
        Executes the forward propagation pass over the input matrix X.

        Parameters:
            theta (np.ndarray): Parameter vector of shape (n_params,).
            X (np.ndarray): Input feature matrix of shape (N, d_in).

        Returns:
            tuple: (y_hat, a1, (W1, b1, W2, b2))
                   y_hat: Linear model predictions of shape (N, K).
                   a1: Hidden layer activations after tanh of shape (N, m).
                   (W1, b1, W2, b2): Unpacked parameter blocks.
        """
        W1, b1, W2, b2 = self.unpack_parameters(theta)
        
        # Layer 1: affine transformation followed by tanh activation
        z1 = X @ W1.T + b1          # shape: (N, m)
        a1 = np.tanh(z1)            # shape: (N, m)
        
        # Layer 2: linear output units
        y_hat = a1 @ W2.T + b2      # shape: (N, K)
        
        return y_hat, a1, (W1, b1, W2, b2)

    def compute_loss(self, theta: np.ndarray, X: np.ndarray, Y: np.ndarray):
        """
        Computes the regularized objective value:
            f(theta) = L(theta) + lambda * ||theta||_1
        where L(theta) is the normalized Mean Squared Error (MSE).

        Parameters:
            theta (np.ndarray): Parameter vector of shape (n_params,).
            X (np.ndarray): Input data matrix of shape (N, d_in).
            Y (np.ndarray): Target one-hot matrix of shape (N, K).

        Returns:
            float: Total regularized objective value.
        """
        y_hat, _, _ = self.forward(theta, X)
        loss_mse = np.sum((y_hat - Y) ** 2) / X.shape[0]
        reg_l1 = self.lam * np.sum(np.abs(theta))
        return loss_mse + reg_l1

    def oracle(self, theta: np.ndarray, X: np.ndarray, Y: np.ndarray):
        """
        First-order computational oracle: evaluates both the objective function value
        f(theta) and an explicit canonical subgradient vector g(theta) in partial f(theta).

        Calls self.forward to obtain predictions and activations, avoiding duplicated matrix operations.

        Parameters:
            theta (np.ndarray): Parameter vector of shape (n_params,).
            X (np.ndarray): Input data matrix of shape (N, d_in).
            Y (np.ndarray): Target one-hot matrix of shape (N, K).

        Returns:
            tuple: (f_val, g_val)
                   f_val (float): Evaluated objective function value.
                   g_val (np.ndarray): 1D subgradient array of shape (n_params,).
        """
        N = X.shape[0]
        
        # 1. Forward pass: reuse self.forward
        y_hat, a1, (W1, b1, W2, b2) = self.forward(theta, X)
        
        # 2. Objective evaluation
        residuals = y_hat - Y
        loss_mse = np.sum(residuals ** 2) / N
        reg_l1 = self.lam * np.sum(np.abs(theta))
        f_val = loss_mse + reg_l1
        
        # 3. Backward pass: explicit analytical derivatives
        # Output layer sensitivities: dL/dz2 = 2 * (y_hat - Y) / N
        delta2 = 2.0 * residuals / N                # (N, K)
        grad_W2 = delta2.T @ a1                     # (K, m)
        grad_b2 = np.sum(delta2, axis=0)            # (K,)
        
        # Hidden layer sensitivities: delta1 = ((delta2 @ W2) * (1 - a1^2))
        delta1 = (delta2 @ W2) * (1.0 - a1 ** 2)    # (N, m)
        grad_W1 = delta1.T @ X                      # (m, d_in)
        grad_b1 = np.sum(delta1, axis=0)            # (m,)
        
        # 4. Assemble full loss gradient
        grad_loss = np.concatenate([
            grad_W1.ravel(),
            grad_b1,
            grad_W2.ravel(),
            grad_b2
        ])
        
        # 5. Canonical subgradient of the L1 penalty: lambda * sign(theta) with sign(0) = 0
        subgrad_reg = self.lam * np.sign(theta)
        g_val = grad_loss + subgrad_reg
        
        return f_val, g_val

    def predict(self, theta: np.ndarray, X: np.ndarray):
        """
        Determines predicted class labels via winner-takes-all rule (argmax).
        """
        y_hat, _, _ = self.forward(theta, X)
        return np.argmax(y_hat, axis=1)

    def compute_accuracy(self, theta: np.ndarray, X: np.ndarray, Y: np.ndarray):
        """
        Computes classification accuracy percentage on a given dataset.
        """
        predictions = self.predict(theta, X)
        targets = np.argmax(Y, axis=1)
        return np.mean(predictions == targets) * 100.0