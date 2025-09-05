import math
import random
import time
import timeit


N_INPUTS = 3
N_HIDDENS = 3
LEARNING_RATE = 0.5
SEED = 65535
MAX_SAMPLES = 100
ERROR_INIT = 100
ERROR_LIMIT = 0.001


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def load_data(datas, filename="input.txt"):
    n_samples = 0
    with open(filename, "r") as f:
        for line in f:
            datas[n_samples] = [float(x) for x in line.split()]
            n_samples += 1
    return n_samples


def forward(w_hidden, w_output, hidden_output, datas):
    for i in range(N_HIDDENS):
        temp_hidden = 0
        for j in range(N_INPUTS):
            temp_hidden += datas[j] * w_hidden[i][j]
        temp_hidden -= w_hidden[i][N_INPUTS]
        hidden_output[i] = sigmoid(temp_hidden)

    temp_final = 0
    for i in range(N_HIDDENS):
        temp_final += hidden_output[i] * w_output[i]
    temp_final -= w_output[N_HIDDENS]
    return sigmoid(temp_final)


def bp_output(w_output, hidden_output, datas, output):
    d = datas[N_INPUTS] - output
    delta_o = output * (1 - output) * d

    for i in range(N_HIDDENS):
        w_output[i] += LEARNING_RATE * delta_o * hidden_output[i]

    w_output[N_HIDDENS] += LEARNING_RATE * (-1) * delta_o


def bp_hidden(w_hidden, w_output, hidden_output, datas, output):
    for j in range(N_HIDDENS):
        d = hidden_output[j] * (1 - hidden_output[j]) * (w_output[j] * (datas[N_INPUTS] - output) * output * (1 - output))

        for i in range(N_INPUTS):
            w_hidden[j][i] += LEARNING_RATE * d * datas[i]

        w_hidden[j][N_INPUTS] += LEARNING_RATE * (-1) * d


def main():
    time_start = time.time()
    random.seed(SEED)

    w_hidden = [[random.uniform(-1, 1) for _ in range(N_INPUTS + 1)] for _ in range(N_HIDDENS)]
    w_output = [random.uniform(-1, 1) for _ in range(N_HIDDENS + 1)]
    datas = [[0 for _ in range(N_INPUTS + 1)] for _ in range(MAX_SAMPLES)]
    hidden_output = [0 for _ in range(N_HIDDENS + 1)]
    error = ERROR_INIT
    epoch = 0

    n_samples = load_data(datas, "input.txt")

    while error > ERROR_LIMIT:
        error = 0
        for j in range(n_samples):
            output = forward(w_hidden, w_output, hidden_output, datas[j])
            bp_output(w_output, hidden_output, datas[j], output)
            bp_hidden(w_hidden, w_output, hidden_output, datas[j], output)
            error += (datas[j][N_INPUTS] - output) ** 2
        epoch += 1
        print(epoch, error)

    time_end = time.time()

    print("-------------------")
    print(f"処理時間: {time_end - time_start:.7f} 秒")

    print("-------------------")
    print("w_hidden", *w_hidden, sep="\n")
    print("-------------------")
    print("w_output")
    print(*w_output)
    print("-------------------")
    
    for i in range(n_samples):
        print(i, ":", datas[i], "->", round(forward(w_hidden, w_output, hidden_output, datas[i]), 4))

print(timeit.timeit("main()", setup="from __main__ import main", number=10))