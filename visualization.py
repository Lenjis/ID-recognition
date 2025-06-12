import matplotlib.pyplot as plt
import matplotlib

plt.rcParams["font.serif"] = ["Microsoft Yahei"]
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14


epochs = []
losses = []
with open("loss_log.txt", "r") as f:
    next(f)
    for line in f:
        epoch, loss = line.strip().split("\t")
        epochs.append(int(epoch))
        losses.append(float(loss))
plt.figure(figsize=(8, 6))
plt.semilogy(epochs, losses, "b-", marker="o", label="Training Loss")
plt.legend()
plt.xlabel("迭代次数")
plt.ylabel("损失值")
plt.title("Training Loss Curve")
plt.grid()
plt.savefig("fig/loss_curve.pdf")
plt.show()
