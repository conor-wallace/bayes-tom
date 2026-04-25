import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors


class KNNBehaviorModel:
    """
    kNN conditional likelihood model for teammate behavior:
      P(a_obs | s_obs)
    Works for continuous or discrete actions.
    """
    def __init__(self, k=50, continuous_actions=True, sigma_a=0.2, sigma_s=None, eps=1e-6):
        self.k = k
        self.continuous_actions = continuous_actions
        self.sigma_a = sigma_a
        self.sigma_s = sigma_s  # if None, uniform weighting among neighbors
        self.eps = eps

        self.X = None
        self.A = None
        self.nn = None
        self.mu = None
        self.std = None

    def fit(self, X, A):
        """
        X: (N, d) state features
        A: (N, da) continuous actions, OR (N,) discrete action ids
        """
        X = np.asarray(X)
        self.A = np.asarray(A)

        # standardize features
        self.mu = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        Xn = (X - self.mu) / self.std
        self.X = Xn

        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(Xn)), algorithm="auto")
        self.nn.fit(Xn)

    def _weights_from_state_dists(self, dists):
        if self.sigma_s is None:
            return np.ones_like(dists)
        return np.exp(-(dists**2) / (2 * self.sigma_s**2))

    def likelihood(self, x, a_obs):
        """
        x: (d,) state feature
        a_obs: observed teammate action
        returns: scalar likelihood-like value
        """
        x = np.asarray(x)
        xn = (x - self.mu) / self.std
        dists, idxs = self.nn.kneighbors(xn.reshape(1, -1), return_distance=True)
        dists = dists[0]
        idxs = idxs[0]

        w = self._weights_from_state_dists(dists)

        if not self.continuous_actions:
            # discrete actions: probability mass among neighbors (+ smoothing)
            a_neighbors = self.A[idxs]
            count = np.sum(a_neighbors == a_obs)
            p = (count + 1.0) / (len(a_neighbors) + np.unique(self.A).size)  # Laplace
            return max(float(p), self.eps)

        # continuous actions: Gaussian kernel in action space
        a_neighbors = self.A[idxs]  # (k, da)
        a_obs = np.asarray(a_obs)
        diffs = a_neighbors - a_obs.reshape(1, -1)
        sq = np.sum(diffs**2, axis=1)
        ka = np.exp(-sq / (2 * self.sigma_a**2))

        p = np.sum(w * ka) / (np.sum(w) + 1e-12)
        return max(float(p), self.eps)

    def predict(self, x):
        """
        Predict the most likely action given state x.
        For discrete actions: return mode from neighbors.
        For continuous actions: return weighted mean from neighbors.
        
        Args:
            x: (d,) state feature
            
        Returns:
            Predicted action
        """
        x = np.asarray(x)
        xn = (x - self.mu) / self.std
        dists, idxs = self.nn.kneighbors(xn.reshape(1, -1), return_distance=True)
        dists = dists[0]
        idxs = idxs[0]

        if not self.continuous_actions:
            # discrete: return most common action
            a_neighbors = self.A[idxs]
            unique, counts = np.unique(a_neighbors, return_counts=True)
            return unique[np.argmax(counts)]
        
        # continuous: weighted average
        w = self._weights_from_state_dists(dists)
        a_neighbors = self.A[idxs]
        return np.average(a_neighbors, axis=0, weights=w)

    def eval(self, X_test, A_test):
        """
        Evaluate the likelihood model on test data.
        
        Args:
            X_test: (N, d) test state features
            A_test: (N, da) or (N,) test actions
            
        Returns:
            dict with metrics:
                - 'avg_log_likelihood': average log-likelihood
                - 'accuracy': prediction accuracy (discrete actions only)
        """
        X_test = np.asarray(X_test)
        A_test = np.asarray(A_test)
        
        likelihoods = []
        predictions = []
        
        for i in range(len(X_test)):
            x = X_test[i]
            a = A_test[i]
            
            # Compute likelihood
            lik = self.likelihood(x, a)
            likelihoods.append(lik)
            
            # Get prediction
            pred = self.predict(x)
            predictions.append(pred)
        
        likelihoods = np.array(likelihoods)
        log_likelihoods = np.log(likelihoods)
        
        metrics = {
            'avg_log_likelihood': float(np.mean(log_likelihoods)),
            'std_log_likelihood': float(np.std(log_likelihoods)),
        }
        
        # For discrete actions, compute accuracy
        if not self.continuous_actions:
            predictions = np.array(predictions)
            accuracy = np.mean(predictions == A_test)
            metrics['accuracy'] = float(accuracy)
        else:
            # For continuous actions, compute MSE
            predictions = np.array(predictions)
            mse = np.mean(np.sum((predictions - A_test)**2, axis=-1))
            metrics['mse'] = float(mse)
        
        return metrics

    def save(self, filepath):
        """
        Save the fitted kNN behavior model to a file.
        
        Args:
            filepath: Path to save the model (e.g., 'model.pkl')
        """
        state = {
            'k': self.k,
            'continuous_actions': self.continuous_actions,
            'sigma_a': self.sigma_a,
            'sigma_s': self.sigma_s,
            'eps': self.eps,
            'X': self.X,
            'A': self.A,
            'mu': self.mu,
            'std': self.std,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath):
        """
        Load a fitted kNN behavior model from a file.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            KNNBehaviorModel: Loaded model instance
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Create instance with saved hyperparameters
        model = cls(
            k=state['k'],
            continuous_actions=state['continuous_actions'],
            sigma_a=state['sigma_a'],
            sigma_s=state['sigma_s'],
            eps=state['eps']
        )
        
        # Restore fitted state
        model.X = state['X']
        model.A = state['A']
        model.mu = state['mu']
        model.std = state['std']
        
        # Rebuild NearestNeighbors from saved data
        if model.X is not None:
            model.nn = NearestNeighbors(n_neighbors=min(model.k, len(model.X)), algorithm="auto")
            model.nn.fit(model.X)
        
        return model
