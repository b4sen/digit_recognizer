import numpy as np
class Accuracy:
    '''
    Average classification accuracy.
    '''

    def __init__(self):
        '''
        Ctor.
        '''
        self.prediction = None
        self.target = None
        self.reset()

    def reset(self):
        '''
        Resets the internal state.
        '''

        # TODO implement
        self.prediction = np.empty(0, dtype=int)
        self.target = np.empty(0, dtype=int)

    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        # TODO implement
        if not prediction.shape[0] == target.shape[0]:  # Checks if s is the same for both ndarrays
            raise ValueError(f"(Prediction shape[0] {prediction.shape[0]} "
                             f"does not equal target shape[0] {target.shape[0]})")

        if not target.ndim == 1:
            raise ValueError(f"(Target shape[1] {target.shape[1]} is not 1)")

        if not np.isin(target, range(prediction.shape[1])).all():
            raise ValueError(f"(Values in target array are out of range)")

        #prediction_highest = np.argmax(prediction, axis=1)
        prediction_highest = np.apply_along_axis(np.argmax, 1, prediction)
        self.prediction = np.append(self.prediction, prediction_highest)
        self.target = np.append(self.target, target)

    def __str__(self):
        '''
        Return a string representation of the performance.
        '''

        # TODO implement
        # return something like "accuracy: 0.395"
        return f"accuracy: {self.accuracy():0.3f}"

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        # TODO implement
        # on this basis implementing the other methods is easy (one line)
        if len(self.target) == 0:
            return 0
        else:
            return sum(self.prediction == self.target) / len(self.target)