import matplotlib.pyplot as plt
import matplotlib

plt.rcParams["font.serif"] = ["NewComputerModern10"]
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14


epochs = []
losses_train = []
losses_test = []
with open("loss_log.txt", "r") as f:
    next(f)
    for line in f:
        epoch, loss_train, loss_test = line.strip().split("\t")
        epochs.append(int(epoch))
        losses_train.append(float(loss_train))
        losses_test.append(float(loss_test))
plt.figure(figsize=(8, 6))
plt.plot(epochs, losses_train, "b-", marker="o", label="Training Loss")
plt.plot(epochs, losses_test, "g-", marker="s", label="Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.savefig("fig/loss_curve.pdf")
plt.show()
