import matplotlib.pyplot as plt
import pickle

TARGET = "english"
LOSS_LOG_FILENAME = "{}_sg_emb_loss.pkl".format(TARGET)

f = open(LOSS_LOG_FILENAME, "rb")
loss_logs = pickle.load(f)
f.close()

x = list(range(len(loss_logs)))
y = loss_logs[:]

plt.plot(x, y)
plt.xlabel("Epochs")
plt.ylabel("Running loss (SkipGram)")
plt.title("Training loss trend for {} embeddings".format(TARGET))
plt.savefig("loss_plot.png")
