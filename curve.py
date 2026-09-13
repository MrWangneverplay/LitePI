import numpy as np
import matplotlib.pyplot as plt


def square(x):
    return x ** 2


def p_2_square(a_0, a_1, a_2, x):
    return a_0 + a_1 * x + a_2 * x ** 2


def p_3_square(a_0, a_1, a_2, a_3, x):
    return a_0 + a_1 * x + a_2 * x ** 2 + a_3 * x ** 3


def relu(x):
    return np.maximum(0, x)


def main():
    x = np.linspace(-10, 10, 1000)
    y = square(x)
    y_pred = p_2_square(0.0625, 0.5, 0.5, x)  #a_0 = 0.0625, a_1 = 0.5, a_2 = 0.5
    y_relu = relu(x)
    y_pred_3 = p_3_square(0.0586, 0.5, 0.8203, -0.4102, x)  #a_0 = 0.0586, a_1 = 0.5, a_2 = 0.8203, a_3 = −0.4102

    plt.xlim(-2, 2)
    plt.ylim(0, 5)
    plt.title('$x^2$ True vs $x^2$ Pred vs ReLU Pred')

    plt.plot(x, y, label='$x^2$ True')
    plt.plot(x, y_pred, label='$x^2$ Pred')
    plt.plot(x, y_relu, label='ReLU Pred')
    plt.plot(x, y_pred_3, label='$x^3$ Pred 3')
    plt.legend()
    plt.show()
    plt.savefig('datas/curve.png')


if __name__ == "__main__":
    main()
