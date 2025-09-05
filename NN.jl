using Random
using Statistics
using BenchmarkTools


const N_INPUTS = 3
const N_HIDDENS = 3
const LEARNING_RATE = 0.5
const SEED = 65535
const MAX_SAMPLES = 100
const ERROR_INIT = 100.0
const ERROR_LIMIT = 0.001


sigmoid(x) = 1.0 / (1.0 + exp(-x))


function load_data(filename="input.txt")
    datas = []
    open(filename, "r") do f
        for line in eachline(f)
            push!(datas, parse.(Float64, split(line)))
        end
    end
    return datas
end


function forward(w_hidden, w_output, hidden_output, datas)
    for i in 1:N_HIDDENS
        temp_hidden = 0.0
        for j in 1:N_INPUTS
            temp_hidden += datas[j] * w_hidden[i][j]
        end
        temp_hidden -= w_hidden[i][N_INPUTS+1]
        hidden_output[i] = sigmoid(temp_hidden)
    end

    temp_final = 0.0
    for i in 1:N_HIDDENS
        temp_final += hidden_output[i] * w_output[i]
    end
    temp_final -= w_output[N_HIDDENS+1]
    return sigmoid(temp_final)
end


function bp_output(w_output, hidden_output, datas, output)
    d = datas[N_INPUTS+1] - output
    delta_o = output * (1 - output) * d

    for i in 1:N_HIDDENS
        w_output[i] += LEARNING_RATE * delta_o * hidden_output[i]
    end
    w_output[N_HIDDENS+1] += LEARNING_RATE * (-1) * delta_o
end


function bp_hidden(w_hidden, w_output, hidden_output, datas, output)
    for j in 1:N_HIDDENS
        d = hidden_output[j] * (1 - hidden_output[j]) *
            (w_output[j] * (datas[N_INPUTS+1] - output) * output * (1 - output))

        for i in 1:N_INPUTS
            w_hidden[j][i] += LEARNING_RATE * d * datas[i]
        end
        w_hidden[j][N_INPUTS+1] += LEARNING_RATE * (-1) * d
    end
end


function main()
    Random.seed!(SEED)

    w_hidden = [rand(Float64, N_INPUTS+1) .* 2 .- 1 for _ in 1:N_HIDDENS]
    w_output = rand(Float64, N_HIDDENS+1) .* 2 .- 1
    datas = load_data("input.txt")
    hidden_output = zeros(Float64, N_HIDDENS)
    error = ERROR_INIT
    epoch = 0

    while error > ERROR_LIMIT
        error = 0.0
        for sample in datas
            output = forward(w_hidden, w_output, hidden_output, sample)
            bp_output(w_output, hidden_output, sample, output)
            bp_hidden(w_hidden, w_output, hidden_output, sample, output)
            error += (sample[N_INPUTS+1] - output)^2
        end
        epoch += 1
        println(epoch, " ", error)
    end

    println("-------------------")
    println("w_hidden")
    println(join(w_hidden, "\n"))
    println("-------------------")
    println("w_output")
    println(w_output)
    println("-------------------")

    for (i, sample) in enumerate(datas)
        println(i, " : ", sample, " -> ", round(forward(w_hidden, w_output, hidden_output, sample), digits=4))
    end
end

@btime main()