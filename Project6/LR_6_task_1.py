from data import train_data, test_data
import numpy as np
from numpy.random import randn

# Створити словник
vocab = list(set([w for text in train_data.keys() for w in text.split()]))
vocab_size = len(vocab)

print('%d unique words found' % vocab_size)

# Призначити індекс кожному слову
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

print(word_to_idx)
print(idx_to_word)


def create_inputs(text):
    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)

    return inputs


def softmax(xs):
    # Застосування функції Softmax для вхідного масиву
    return np.exp(xs) / sum(np.exp(xs))


def process_data(data, rnn, backprop=True):
    items = list(data.items())
    np.random.shuffle(items)

    loss = 0
    num_correct = 0

    for x, y in items:
        inputs = create_inputs(x)
        target = int(y)

        # Пряме разповсюдження
        out, _ = rnn.forward(inputs)
        probs = softmax(out)

        # Обчислення втрат / точності
        loss -= float(np.log(probs[target]))
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            # Створення dL/dy
            d_L_d_y = probs
            d_L_d_y[target] -= 1

            # Зворотнє разповсюдження
            rnn.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)


class RNN:
    # Класична рекурентна нейронна мережа
    def __init__(self, input_size, output_size, hidden_size=64):
        # Вес
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        # Зміщення
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = {0: h}

        # Виконання кожного кроку в нейронній мережі RNN
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        # Підрахунок значення виводу
        y = self.Why @ h + self.by

        return y, h

    def backprop(self, d_y, learn_rate=2e-2):
        n = len(self.last_inputs)

        # Обчислення dL/dWhy и dL/dby
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        # Ініціалізація dL/dWhh, dL/dWxh, і dL/dbh до нуля
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        # Обчислення dL/dh для останнього h
        d_h = self.Why.T @ d_y

        # Зворотне розповсюдження по часу
        for t in reversed(range(n)):
            # Среднее значение: dL/dh * (1 - h^2)
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

            d_bh += temp
            d_Whh += temp @ self.last_hs[t].T
            d_Wxh += temp @ self.last_inputs[t].T

            d_h = self.Whh @ temp

        # Відсікаємо, щоб попередити розрив градієнтів
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        # Обновляємо ваги і зміщення з використанням градієнтного спуску
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by


# Ініціалізація рекурентної нейронної мережі RN
rnn = RNN(vocab_size, 2)

# Цикл тренування
for epoch in range(1000):
    train_loss, train_acc = process_data(train_data, rnn, backprop=True)

    if epoch % 100 == 99:
        print(f"Epoch {epoch + 1}")
        print(f"Train loss: {train_loss:0.3f}, Train accuracy: {train_acc:0.3f}")

        test_loss, test_acc = process_data(test_data, rnn, backprop=False)
        print(f"Test loss: {test_loss:.3f}, Test accuracy: {test_acc:.3f}\n")
