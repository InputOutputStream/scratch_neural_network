import pickle
import time
# import random


class Math:
    def sqrt(self, S, x0=1, tolerance=1e-7):
        xn = x0
        while True:
            xn1 = 0.5 * (xn + S / xn)
            if abs(xn1 - xn) < tolerance:
                break
            xn = xn1
        return xn
    
    def log(self, x, base):
        if x <= 0 or base <= 0 or base == 1:
            raise ValueError("Input must be positive and base must be greater than 1")
        result = 0
        while x >= base:
            x /= base
            result += 1
        return result
    
    def pow(self, x, n):
        if n == 0:
            return 1
        elif n < 0:
            return 1 / (x * self.pow(x, abs(n) - 1))
        else:
            return x * self.pow(x, n - 1)
        
    def exp(self, x):
        e = 2.718281828459045  
        return e ** x

    def ln(self, x):
        if x <= 0:
            raise ValueError("Input must be positive")
        return x * 1 / sum(1 / (n * x - n) for n in range(1, 9))
    
    def log10(self, x):
        ln_x = self.ln(x)  
        return ln_x / self.ln(10) 

    def log2(self, x):
        ln_x = self.ln(x) 
        return ln_x / self.ln(2) 

    def calculate_n(self, x, y, z):
        if z:
            n = self.log2(y) + self.log10(x)
        else:
            n = self.ln(z * x ** 2 + y ** 3)
        return n

    def ravel(self, vector):
        return [item for sublist in vector for item in sublist]

    def mean(self, vector, axis=0):
        if axis == 0:
            return sum(vector) / len(vector)
        elif axis == 1:
            return sum(vector) / len(vector[0])

    def standard_deviation(self, vector, axis=0):
        mean_value = self.mean(vector, axis)
        deviations = [(x - mean_value) ** 2 for x in vector]
        variance = sum(deviations) / len(vector) if axis == 0 else sum(deviations) / len(vector[0])
        std_dev = self.sqrt(variance)
        return std_dev

    def repeat(self, codebook, n=2):
        return codebook * n

    def euclid_distance(self, a, b):
        s = 0.0
        for i in range(len(a)):
            s += (a[i] - b[i]) ** 2
        return self.sqrt(s)

    def argmax(self, lst, axis=0):
        if not lst:
            raise ValueError("List is empty")

        if axis == 0:
            max_indices = [0] * len(lst[0])
            for i in range(len(lst[0])):
                max_val = lst[0][i]
                max_index = 0
                for j in range(1, len(lst)):
                    if lst[j][i] > max_val:
                        max_val = lst[j][i]
                        max_index = j
                max_indices[i] = max_index
            return max_indices
        elif axis == 1:
            return [row.index(max(row)) for row in lst]
        else:
            raise ValueError("Invalid axis value. Use 0 for rows and 1 for columns.")


    def dot(self, matrix1, matrix2):
        if isinstance(matrix1[0], list):
            # Check if matrix2 is a matrix or vector
            if isinstance(matrix2[0], list):
                # Perform matrix-matrix product
                if len(matrix1[0]) != len(matrix2):
                    raise ValueError("Matrices are not compatible for dot product")
                result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]
                for i in range(len(matrix1)):
                    for j in range(len(matrix2[0])):
                        for k in range(len(matrix2)):
                            result[i][j] += matrix1[i][k] * matrix2[k][j]
            else:
                # Perform matrix-vector product
                if len(matrix1[0]) == len(matrix2):
                    result = []
                    for i in range(len(matrix1)):
                        row_sum = 0
                        for j in range(len(matrix2)):
                            row_sum += matrix1[i][j] * matrix2[j]
                        result.append(row_sum)
                else:
                    raise ValueError("Matrix column count must match the vector length.")
                
        else:
            raise ValueError("Matrix1 must be a matrix or vector.")
        
        return result

    def flatten(self, A):
        flattened_A = [item for row in A for item in row]
        return flattened_A
    
    def reshape(self, A, new_shape):
        total_elements = len(A) * len(A[0])

        
        if total_elements != new_shape[0] * new_shape[1]:
            raise ValueError("Total number of elements does not match the new shape")

        
        flattened_A = self.flatten(A)
       
        reshaped_A = []
        index = 0
        for i in range(new_shape[0]):
            row = []
            for j in range(new_shape[1]):
                row.append(flattened_A[index])
                index += 1
            reshaped_A.append(row)

        return reshaped_A

    def scalar_prod(self, sc, l):
        if len(l) == 0:
            raise ValueError("Can't Multiply empty list by a scaler")

        return [sc * i for i in l]
    
    def scalar_add(self, sc, l):
        if len(l) == 0:
            raise ValueError("Can't add empty list by a scaler")

        return [sc + i for i in l]
    
    def scalar_divide(self, sc, l):
        if len(l) == 0:
            raise ValueError("Can't divde empty list by a scaler")

        return [sc / i for i in l]
    
    def scalar_subtract(self, sc, l):
        if len(l) == 0:
            raise ValueError("Can't subtract empty list by a scaler")

        return [sc - i for i in l]
    
    
    def axsum(self, A, axis):
        if axis == 1:
            l = []
            for x in range(len(A)):
                t = sum(A[x])
                l.append(t)  
                t = 0.0      
            return l           
        elif axis == 0:
            P = self.T(A)
            l = []
            for x in range(len(P)):
                t = sum(P[x])
                l.append(t)  
                t = 0.0  
            return l
        else:
            raise IndexError("Unknown Axis")

            
    def minus(self, A, B):
        if len(A) != len(B):
            raise IndexError("Total number of elements in boths lists are different")

        return [x-y for x, y in zip(A, B)]
    
    def add(self, A, B):
        if len(A) != len(B):
            raise IndexError("Total number of elements in boths lists are different")

        return [x+y for x, y in zip(A, B)]
    
        
    def T(self, x):
        return [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))]
    
    def arrange(self, N):
        l = [x for x in range(N)]
        return l
    
    def zeros(self, n, m):
        l = [[0] * m for _ in range(n)]
        return l
    
    def outer(self, A, B):
        # print(len(A), len(A[1]))
        # print(len(B) , len(B[1]))
                
        if len(A[0]) != len(B[0]) or len(A) != len(B):
            raise ValueError("Matrices are not compatible for Outer product")

        result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

        for i in range(len(A)):
            for j in range(len(B[0])):
                    result[i][j] = A[i][j] * B[i][j]

        return result








class RNG:
    def __init__(self, seed=None):
        self.seed = seed if seed != None else int(time.time() * 1000)
    
    def randn(self, min_val, max_val):
        return int(self.rand() * (max_val - min_val) + min_val)
    
    def rand(self):
        a = 1103515245
        c = 12345
        m = 2 ** 31 - 1
        self.seed = (a * self.seed + c) % m
        return self.seed / m
    
    def uniform(self, a, b):
        range_length = abs(b - a)
        random_fraction = self.rand()
        return min(a, b) + random_fraction * range_length

    def shuffle(self, lst):
        shuffled = lst[:]  

        for i in range(len(shuffled) - 2, 0, -1):
            j = int(self.rand() * (i + 1)) 
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        return shuffled
    
    
    
math = Math()
rng = RNG()


def _split_(x, y, data):
    for row in data:
        x.append(row[1:])
        y.append(row[0])
    y = [int(target) for target in y]
    return x, y

def create_train_test_sets(data):
    data = rng.shuffle(data)

    Xn = []
    yn = []
   
    Xn, yn = _split_(Xn, yn, data)
    
    Xn = math.T(Xn)

    for i in range(len(Xn)):
        for j in range(len(Xn[i])):
            Xn[i][j] = Xn[i][j] / 255 
        

    return Xn, yn

def load_csv(path):
        data = []
        features = []
        with open(path, 'r') as file:
            for i, line in enumerate(file):
                if i == 0:
                    features = line
                    continue
                row = line.strip().split(',')
                c = []
                for r in row:
                    r = float(r.replace('\'', ''))
                    c.append(r)
                data.append(c)
        return features, data




class Model:
    def __init__(self, input_size, hidden_size, output_size):
        print("Initialisation")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.initialize_weights_and_biases()
    
    def initialize_weights_and_biases(self):
        self.w1 = [[rng.uniform(-0.5, 0.5) for _ in range(self.input_size)] for _ in range(self.hidden_size)]
        self.b1 = [rng.uniform(-0.5, 0.5)  for _ in range(self.hidden_size)]
        self.w2 = [[rng.uniform(-0.5, 0.5) for _ in range(self.hidden_size)] for _ in range(self.output_size)]
        self.b2 = [rng.uniform(-0.5, 0.5) for _ in range(self.output_size)]

    def relu(self, x):
        return [[max(0, val) for val in y] for y in x]

    def softmax(self, x):
        p = x.copy()
        total = 0.0
        for i in range(len(p)):
            for j in range(len(p[0])):
                total += math.exp(p[i][j])
        for i in range(len(p)):
            for j in range(len(p[0])):
                p[i][j] = math.exp(p[i][j])/total
        return p

    def one_hot_encode(self, y):
        num_classes = max(y)+1
        one_hot_y = []
        one_hot_y = math.zeros(len(y), num_classes)
        for i in range(len(y)):
            one_hot_y[i][int(y[i])] = 1
        one_hot_y = math.T(one_hot_y)
        return one_hot_y
    
    def save(self, filename):
        print("Model Saved")
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print("Model Loaded")
        return model
    
    def derivative_Relu(self, z1):
        return [[1 if val > 0 else 0 for val in row] for row in z1]
         
    def forward(self, X):
        z1 = math.dot(self.w1, X)
        
        for i in range(len(z1)):
            for j in range(len(z1[0])):
                z1[i][j] += self.b1[i]
        a1 = self.relu(z1)
        
        
        z2 = math.dot(self.w2, a1)        
       
       
        for i in range(len(z2)):
            for j in range(len(z2[0])):
                z2[i][j] += self.b2[i]
        a2 = self.softmax(z2)
     
        return z1, a1, z2, a2

    def backward(self, z1, a1, z2, a2, X, y):
        m = len(y) 
        one_hot_y = self.one_hot_encode(y)
        dz2 = [math.minus(a , b) for a, b in zip(a2, one_hot_y)]
        dw2 = [math.scalar_prod(1/m , j) for j in math.dot(dz2, math.T(a1))]
        db2 = math.scalar_prod(1/m, math.axsum(dz2, 1))
        dz1 = math.outer(math.dot(math.T(self.w2), dz2), self.derivative_Relu(z1))
        dw1 = [math.scalar_prod(1/m , z) for z in math.dot(dz1, math.T(X))]
        db1 = math.scalar_prod(1/m , math.axsum(dz1, 1))
        return dw1, db1, dw2, db2

    def update_weights_and_biases(self, dw1, db1, dw2, db2, lr):
        # Update weights and biases
        self.w1 = [[self.w1[i][j] - lr * dw1[i][j] for j in range(len(self.w1[0]))] for i in range(len(self.w1))]
        self.b1 = [self.b1[i] - lr * db1[i] for i in range(len(self.b1))]
        self.w2 = [[self.w2[i][j] - lr * dw2[i][j] for j in range(len(self.w2[0]))] for i in range(len(self.w2))]
        self.b2 = [self.b2[i] - lr * db2[i] for i in range(len(self.b2))]

    
    def get_predictions(self, a2):
        return math.argmax(a2, 0)
            
    def get_accuracy(self, predictions, y):
        correct = sum(1 for pred, target in zip(predictions, y) if pred == target)
        return (correct / len(predictions)) * 100
            
    def predict(self, X):
        z1, a1, z2, a2 = self.forward(X)
        predictions = self.get_predictions(a2)
        return predictions

    def fit(self, X_train, y_train, eval, epochs, lr, show_training_info=True):
        for  epoch in range(epochs):
            z1, a1, z2, a2 = self.forward(X_train)
            dw1, db1, dw2, db2 = self.backward(z1, a1, z2, a2, X_train, y_train)
            self.update_weights_and_biases(dw1, db1, dw2, db2, lr)
            if epoch % eval==0:
                if show_training_info == True:
                    print('iter:', epoch, "---------------------------------------")
                    print('Accuracy: ', self.get_accuracy(self.get_predictions(a2), y_train))
        
        print("......................................................")




features, train = load_csv("train.csv")
_, test = load_csv("test.csv")
X_train, y_train = create_train_test_sets(train)
X_test, y_test = create_train_test_sets(test)


#.................................................................................................
model = Model(input_size=784, hidden_size=100, output_size=10)
model.fit(X_train, y_train, eval=10, epochs=0, lr=0.1, show_training_info=True)

model.save('model.pkl')

print("Test On the test set of the dataset")
predictions = model.predict(X_test)
print('Accuracy: ', model.get_accuracy(predictions, y_test))
print("\n\n")








