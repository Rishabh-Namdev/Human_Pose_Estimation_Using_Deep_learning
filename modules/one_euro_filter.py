import math

class LowPassFilter:
    def __init__(self):
        """
        Initialize a LowPassFilter instance.
        """
        self.x_previous = None

    def __call__(self, x, alpha=0.5):
        """
        Apply low-pass filtering to the input value.

        Parameters:
        - x (float): The input value to be filtered.
        - alpha (float): Smoothing factor.

        Returns:
        float: The filtered output.
        """
        if self.x_previous is None:
            self.x_previous = x
            return x

        x_filtered = alpha * x + (1 - alpha) * self.x_previous
        self.x_previous = x_filtered
        return x_filtered

class OneEuroFilter:
    def __init__(self, freq=15, mincutoff=1, beta=0.05, dcutoff=1):
        """
        Initialize a OneEuroFilter instance.

        Parameters:
        - freq (float): Cutoff frequency of the filter.
        - mincutoff (float): Minimum cutoff frequency.
        - beta (float): Coefficient for smoothing changes.
        - dcutoff (float): Cutoff frequency for derivative computation.
        """
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.filter_x = LowPassFilter()
        self.filter_dx = LowPassFilter()
        self.x_previous = None
        self.dx = None

    def __call__(self, x):
        """
        Apply filtering to the input value.

        Parameters:
        - x (float): The input value to be filtered.

        Returns:
        float: The filtered output.
        """
        if self.dx is None:
            self.dx = 0
        else:
            self.dx = (x - self.x_previous) * self.freq

        dx_smoothed = self.filter_dx(self.dx, get_alpha(self.freq, self.dcutoff))
        cutoff = self.mincutoff + self.beta * abs(dx_smoothed)
        x_filtered = self.filter_x(x, get_alpha(self.freq, cutoff))
        self.x_previous = x

        return x_filtered

if __name__ == '__main__':
    # Example usage
    one_euro_filter = OneEuroFilter(freq=15, beta=0.1)
    for val in range(10):
        input_value = val + (-1)**(val % 2)
        filtered_value = one_euro_filter(input_value)
        print(filtered_value, input_value)
