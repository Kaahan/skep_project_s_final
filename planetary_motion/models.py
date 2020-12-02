import numpy as np
import os


class Regressor:
    """Regressor for planetary motion prediction"""
    def __init__(self):
        self.w, self.freqs = None, None

    def save(self, name):
        """Save weights and frequencies to .npy files"""
        os.makedirs('./models/', exist_ok=True)
        np.save('./models/' + name + "_weights", self.w)
        np.save('./models/' + name + "_freqs", self.freqs)

    def load(self, name):
        """Load weights and generate frequencies"""
        self.w = np.load('./models/' + name + '_weights.npy')
        self.freqs = np.load('./models/' + name + '_freqs.npy')

    def predict(self, t):
        """Predict complex position given time"""
        pass

    def animate(self, t, circles, dot):
        """Animation loop for fourier series visualization"""
        pass


class RidgeSVD(Regressor):
    def __init__(self):
        super().__init__()

    def train(self, X, y, lam=1.0, num_weights=None):
        """Compute ridge regression using compact SVD
        :param X: Data matrix (fourier features)
        :param y: Truths (complex coordinates)
        :param lam: regularization weighting
        :param num_weights: if given, select top num_weights weights and frequencies

        :returns weights and frequencies
        """
        n, d = X.shape[0], X.shape[1]
        u, s, vh = np.linalg.svd(X, full_matrices=False)
        # ridge regression in the complex coordinate system
        w = vh.conj().T @ np.linalg.inv(np.diag(s.conj().T) @ np.diag(s) + lam * np.eye(s.shape[0])) \
            @ np.diag(s).conj().T @ u.conj().T @ y

        # generate frequencies
        freqs = np.array([np.power(np.e, 2 * np.pi * 1j * (i / d)) for i in range(d)])

        if num_weights:
            # select largest weights and corresponding frequencies
            top_feats = (-np.abs(w)).argsort()[:num_weights]
            w, freqs = w[top_feats], freqs[top_feats]
            ff = X[:, top_feats]
        else:
            # sort for looking goods sake
            indices = (-np.abs(w)).argsort()
            w, freqs = w[indices], freqs[indices]
            ff = X[:, indices]

        # print(f"L2 Loss: {np.linalg.norm(ff @ w - y, 2)}")

        self.w, self.freqs = w, freqs

    def predict(self, t):
        """Predict position of planet; assumes that data was using days as time scale

        :param t: time in days - [0, 366]

        :returns complex rectangular position and real (x, y) relative to Earth (in AU)
        """
        rect_pos = self.w @ np.power(self.freqs, t)
        real_pos = (rect_pos.real, rect_pos.imag)

        return rect_pos, real_pos

    def animate(self, t, circles, dot):
        """ Animation function for matplotlib fourier series animations

        :param t: time in days - [0, 366]
        :param circles: list of circle artists
        :param dot: final location marker

        :return: circle and dot positions
        """
        vals = self.w * np.power(self.freqs, t)
        cs = np.cumsum(vals)
        centers = cs
        for circle, center in zip(circles, centers):
            circle.center = [center.real, center.imag]
        dot.set_data(cs[-1].real, cs[-1].imag)
        return circles + [dot]


class OLS(Regressor):
    def __init__(self):
        super().__init__()

    def train(self, X, y, num_weights=None):
        """ordinary least squares wrapper
        :param X: Data matrix (fourier features)
        :param y: Truths (complex coordinates)
        :param num_weights: if given, select top num_weights weights and frequencies

        :returns weights and frequencies
        """
        n, d = X.shape[0], X.shape[1]
        w = np.linalg.lstsq(X, y)[0]
        # based on our construction of the fourier feature matrix
        freqs = np.array([np.power(np.e, 2 * np.pi * 1j * (i / d)) for i in range(d)])

        if num_weights:
            # select largest weights and corresponding frequencies
            top_feats = (-np.abs(w)).argsort()[:num_weights]
            w, freqs = w[top_feats], freqs[top_feats]
            ff = X[:, top_feats]
        else:
            # sort for looking goods sake
            indices = (-np.abs(w)).argsort()
            w, freqs = w[indices], freqs[indices]
            ff = X[:, indices]

        # print(f"L2 Loss: {np.linalg.norm(ff @ w - y, 2)}")

        self.w, self.freqs = w, freqs

    def predict(self, t):
        """Predict position of planet; assumes that data was using days as time scale

        :param t: time in days - [0, 366]

        :returns complex rectangular position and real (x, y) relative to Earth (in AU)
        """
        rect_pos = self.w @ np.power(self.freqs, t)
        real_pos = (rect_pos.real, rect_pos.imag)

        return rect_pos, real_pos

    def animate(self, t, circles, dot):
        """ Animation function for matplotlib fourier series animations

        :param t: time in days - [0, 366]
        :param circles: list of circle artists
        :param dot: final location marker

        :return: circle and dot positions
        """
        vals = self.w * np.power(self.freqs, t)
        cs = np.cumsum(vals)
        centers = cs
        for circle, center in zip(circles, centers):
            circle.center = [center.real, center.imag]
        dot.set_data(cs[-1].real, cs[-1].imag)
        return circles + [dot]


class DFT(Regressor):
    def __init__(self):
        super().__init__()

    def train(self, X, y, s_int=20, num_weights=None):
        """simple discrete fourier transform wrapper; assumes that the data is sorted by time!
        :param X: data
        :param y: Truths (complex coordinates)
        :param s_int: Sampling interval (reduces the number of fourier coeffecients)
        :param num_weights: if given, select top num_weights weights and frequencies

        :returns weights and frequencies
        """
        w = np.fft.fft(y[::s_int])
        freqs = np.fft.fftfreq(w.shape[0])

        if num_weights:
            # select largest weights and corresponding frequencies
            top_feats = (-np.abs(w)).argsort()[:num_weights]
            w, freqs = w[top_feats], freqs[top_feats]
        else:
            indices = (-np.abs(w)).argsort()
            w, freqs = w[indices], freqs[indices]

        self.w, self.freqs = w, freqs

    def predict(self, t):
        """Predict position of planet; assumes that data was using days as time scale

        :param t: time in days - [0, 366]

        :returns complex rectangular position and real (x, y) relative to Earth (in AU)
        """
        rect_pos = self.w @ np.power(self.freqs, t)
        real_pos = (rect_pos.real, rect_pos.imag)

        return rect_pos, real_pos

    def animate(self, t, circles, dot):
        """ Animation function for matplotlib fourier series animations

        :param t: time in days - [0, 366]
        :param circles: list of circle artists
        :param dot: final location marker

        :return: circle and dot positions
        """
        n = self.w.shape[0]
        vals = self.w / n * np.exp(1j * np.multiply.outer(t, n * self.freqs))
        cs = np.cumsum(vals)
        centers = cs[:-1]
        for circle, center in zip(circles, centers):
            circle.center = [center.real, center.imag]
        dot.set_data(cs[-1].real, cs[-1].imag)
        return circles + [dot]
