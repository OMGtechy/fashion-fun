import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import math

parser = argparse.ArgumentParser("A toy fashion classifier")
parser.add_argument("--epochs_per_iteration", type=int, required=False, default=5)
parser.add_argument("--maximum_iterations", type=int, required=False, default=None)
parser.add_argument("--hidden_layer_neurons", type=int, required=False, default=128)
parser.add_argument("--hidden_layer_activation", type=str, required=False, default="relu", choices=["relu", "relu6", "elu", "softplus", "softsign"])
parser.add_argument("--target_test_loss", type=float, required=False, default=float("NaN"))
args = parser.parse_args()

print("Loading Fashion MNIST data")
fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

hidden_layer_activation = None
if args.hidden_layer_activation == "relu":
    hidden_layer_activation = tf.nn.relu
elif args.hidden_layer_activation == "relu6":
    hidden_layer_activation = tf.nn.relu6
elif args.hidden_layer_activation == "elu":
    hidden_layer_activation = tf.nn.elu
elif args.hidden_layer_activation == "softplus":
    hidden_layer_activation = tf.nn.softplus
elif args.hidden_layer_activation == "softsign":
    hidden_layer_activation = tf.nn.softsign

if hidden_layer_activation == None:
    print("Invalid hidden layer activation: {args.hidden_layer_activation}")
    exit()

print("Designing model")
model = tf.keras.Sequential([
    # images are 28x28, input layer matches this
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(args.hidden_layer_neurons, activation=hidden_layer_activation),
    # 10 categories of item, hence 10 output neurons
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

print("Compiling model")
model.compile(
    optimiser=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy"
)

previousTrainLoss = None
previousTestLoss = None
iterations = 1
while True:
    print(f"\nIteration {iterations}")
    print("-------------------------")
    iterations += 1

    print("Fitting model")

    class LossCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            self.lastLogs = logs

    lossCallback = LossCallback()
    model.fit(training_images, training_labels, epochs=args.epochs_per_iteration, callbacks=[lossCallback])
    trainLoss = lossCallback.lastLogs['loss']

    print("Evaluating")
    testLoss = model.evaluate(test_images, test_labels, verbose=0)

    print(f"Model training scored: {trainLoss}")
    print(f"Model evaluation scored: {testLoss}")

    trainLossWorse = previousTrainLoss != None and previousTrainLoss < trainLoss
    testLossWorse = previousTestLoss != None and previousTestLoss < testLoss

    if trainLossWorse and testLossWorse:
        print(f"Possible overfitting!")

    if math.isnan(args.target_test_loss):
        print(f"No target to meet, stopping training")
        break
    else:
        target_met = False
        if testLoss < args.target_test_loss:
            target_met = True

        print(f"Target was {args.target_test_loss}: {'met' if target_met else 'missed'}")

        if target_met:
            break

    previousTrainLoss = trainLoss
    previousTestLoss = testLoss

    if args.maximum_iterations == None or iterations > args.maximum_iterations:
        print(f"Maximum iterations hit")
        break

print("\nPredicting")
predicted = model.predict(test_images)
plt.imshow(test_images[0])

def print_prediction(prediction):
    def format_value(value):
        return "%.2f" % value

    print(f"T-shirt:  {format_value(prediction[0])}")
    print(f"Trousers: {format_value(prediction[1])}")
    print(f"Hoodie:   {format_value(prediction[2])}")
    print(f"Dress:    {format_value(prediction[3])}")
    print(f"Coat:     {format_value(prediction[4])}")
    print(f"Sandal:   {format_value(prediction[5])}")
    print(f"Shirt:    {format_value(prediction[6])}")
    print(f"Trainers: {format_value(prediction[7])}")
    print(f"Bag:      {format_value(prediction[8])}")
    print(f"Boot:     {format_value(prediction[9])}")

print_prediction(predicted[0])

plt.show()
