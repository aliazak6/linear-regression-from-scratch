class LinearRegression():
    def __init__(self, learning_rate = 0.000005, epoch = 1000):
        self.learning_rate = learning_rate
        self.epoch = epoch

    # Function for model training
    def fit(self, x_train, y_train, z_train):
        
        # Minor check for decent input
        assert len(x_train) == len(y_train) == len(z_train) , "X,Y,Z must be in same size"

        # no_of_training_examples
        self.n = len(x_train)
        
        # weight initialization
        self.m1  = 1
        self.m2  = 2
        self.b   = 0

        # gradient descent learning
        for i in range(self.epoch):
            self.update_weights(x_train,y_train,z_train)

        return self


    def update_weights(self,x_train,y_train,z_train):
        z_pred = self.predict(x_train,y_train)

        # calculate gradients
        temp = 0
        for i in range(self.n):
            temp += (z_pred[i]-z_train[i])*x_train[i]
        dm1 = 2*temp/self.n
        temp = 0
        for i in range(self.n):
            temp += (z_pred[i]-z_train[i])*y_train[i]
        dm2 = 2*temp/self.n
        temp = 0
        for i in range(self.n):
            temp += (z_pred[i]-z_train[i])
        db  = 2*temp/self.n

        # update weights
        self.m1 = self.m1 - self.learning_rate * dm1
        self.m2 = self.m2 - self.learning_rate * dm2
        self.b  = self.b  - self.learning_rate * db

        return self


    def predict(self, x, y):
        z = []
        for i in range(len(x)):
            z.append(round(self.m1 * x[i] + self.m2 * y[i] + self.b)) # rounding to int.
        return z

model = LinearRegression()

    
