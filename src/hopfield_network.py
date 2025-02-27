import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import copy

def read_pbm(path):
    data = np.zeros((16, 16))
    with open(path, "r") as pbm:
        row_number = 0
        while row_number<16:
            row = pbm.readline().split()
            if row[0]=="#" or len(row)!=16:
                continue
            #row = ["-1" for i in row if i=="0"]
            data[row_number] = np.asarray(row)
            row_number+=1
        pbm.close()
    data = np.where(data == 0, -1, data)
    return data

def create_noisy_image(img, p):
    noisy_img = copy.deepcopy(img)
    for i in range(noisy_img.shape[0]):
        for j in range(noisy_img.shape[1]):
            if np.random.random() < p:
                noisy_img[i][j]*=-1
    return noisy_img

def create_bounding_box(img, _type=-1):
    bounded_img = copy.deepcopy(img)
    start_point = (np.random.randint(0, 6), np.random.randint(0, 6))
    end_point = (start_point[0]+10, start_point[1]+10)
    print(start_point, end_point)
    for i in range(bounded_img.shape[0]):
        for j in range(bounded_img.shape[1]):
            if i<=start_point[0] or i >= end_point[0] or j<= start_point[1] or j >= end_point[1]:
                bounded_img[i][j] = 0
    return bounded_img 

class HopfieldNetwork:
    def __init__(self):
        self.neurons = 256
        self.weights = np.zeros(shape=(self.neurons, self.neurons))
        self.state = np.random.randint(-1, 2, (16, 16))
        self.memories = 0
        
    def train_network(self, train_data):
        self.memories = 0
        for train_img in train_data:
            train_img_flattened = train_img.flatten()
            self.weights += np.outer(train_img_flattened, train_img_flattened)
            np.fill_diagonal(self.weights, 0)
            self.memories+=1
        #self.weights = self.weights/self.memories
    
    def update_state_async(self, theta, n):
        state = self.state.flatten()
        for i in range(n):
            x = np.random.randint(0, 256)
            activation = np.dot(self.weights[x], state)
            state[x] = 1 if activation > theta else -1
        self.state = state.reshape((16,16))

    def update_state_sync(self, theta):
        converged = False
        states = []
        steps = 0
        while not converged:
            prev_state = copy.deepcopy(self.state)
            states.append(prev_state)
            state = self.state.flatten()
            activation = np.dot(self.weights, state)
            new_state = np.where(activation >= theta, 1, -1)
            self.state = new_state.reshape(16, 16)
            steps+=1
            if (prev_state == self.state).all():
                converged=True
        states.append(self.state)
        return steps-1, states

    def load_state(self, img):
        self.state = img
    
    def print_state(self, state=None):
        if state is None:
            state = self.state
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                print(state[i][j], end=" ")
            print()
    
    def print_weights(self):
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                print(self.weights[i][j], end=" ")
            print()


def create_train_data(images):
    train_data = []
    for img in images:
        train_data.append(np.reshape(img, (1, 256)))
    return train_data

def predict_async(hn: HopfieldNetwork, _test_image, noise, p):
    if noise:
        test_image = create_noisy_image(_test_image, p)
    else: 
        test_image = create_bounding_box(_test_image)
    hn.load_state(test_image)
    start_state = copy.deepcopy(hn.state)
    states = []
    for i in range(10):
        hn.update_state_async(0, 100)
        states.append(copy.deepcopy(hn.state))
    fin_state = (hn.state)
    f, axarr = plt.subplots(4, 5)
    axarr[0, 2].imshow(hn.weights)
    axarr[1, 1].imshow(test_image)
    axarr[1, 2].imshow(start_state)
    axarr[1, 3].imshow(_test_image)
    for i in range(2, 4):
        for j in range(0, 5):
            axarr[i, j].imshow(states[(i-2)*5+j])
    plt.show()

def predict_sync(hn: HopfieldNetwork, _test_image, noise, p, plot):
    states = []
    if noise:
        test_image = create_noisy_image(_test_image, p)
    else: 
        test_image = create_bounding_box(_test_image)
    hn.load_state(test_image)
    n, states = hn.update_state_sync(0)
    if plot:
        f, axarr = plt.subplots(2, 5)
        axarr[0, 1].imshow(hn.weights)
        axarr[0, 2].imshow(_test_image)
        axarr[0, 3].imshow(test_image)
        for i in range(1, 2):
            for j in range(0, len(states)):
                axarr[i, j].imshow(states[(i-1)*5+j])
        plt.show()
    else:
        return n, states



if __name__ == "__main__":
    train_data = []
    for path in os.listdir("data/"):
        if path.split(".")[1]!="pbm":
            continue
        img = read_pbm("data/"+path)
        train_data.append(img)

    test_image = read_pbm("data/flower.pbm")

    train_data = create_train_data(train_data)
    hn = HopfieldNetwork()
    hn.train_network(train_data)
    
    #predict_async(hn, test_image, True, 0.2)
    predict_sync(hn, test_image, True, 0.2, True)