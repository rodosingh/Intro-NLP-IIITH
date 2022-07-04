import codecs
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import argparse
import os

# ===========================================================================================================================

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--log_file', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
# Parse the argument
args = parser.parse_args()

# read log file and draw plots...
with codecs.open(f"{args.log_file}", "r", "utf-8") as f:
    epochs, pos_loss, neg_loss, dev_loss, dev_map = [], [], [], [], []
    _ = f.readline()
    while True: 
        line = f.readline()
        if not line:
            break
        eles = line.strip().split()
        epochs.append(int(eles[0]))
        pos_loss.append(float(eles[2]))
        neg_loss.append(float(eles[3]))
        dev_loss.append(float(eles[4]))
        dev_map.append(float(eles[5]))



#=========================================================================================================
# Plot the model...
fig, ax = plt.subplots(2, 2, figsize=(18, 12))
# xticks = [str(epochs[k]) for k in range(0, len(epochs), 5)]
# ax = plt.subplot()
ax[0, 0].plot(epochs, pos_loss, 'r', linewidth=2)
ax[0, 0].set_xlabel("Epochs")
ax[0, 0].set_ylabel("Positive Loss")
# ax[0, 0].xaxis.set_major_locator(ticker.MaxNLocator(len(xticks)))
# ax[0, 0].set_xticks([k for k in epochs if k%5==0])
# ax[0, 0].set_xticklabels(xticks, rotation=90, ha="right")#-->FixedLocator

ax[0, 1].plot(epochs, neg_loss, 'y', linewidth=2)
ax[0, 1].set_xlabel("Epochs")
ax[0, 1].set_ylabel("Negative Loss")
# ax[0, 1].set_xticks([k for k in epochs if k % 5 == 0])
# ax[0, 1].set_xticklabels(xticks, rotation=90, ha="right")#-->FixedLocator

ax[1, 0].plot(epochs, dev_loss, 'b', linewidth=2)
ax[1, 0].set_xlabel("Epochs")
ax[1, 0].set_ylabel("Dev Set Loss")
# ax[1, 0].set_xticks([k for k in epochs if k%5==0])
# ax[1, 0].set_xticklabels(xticks, rotation=90, ha="right")#-->FixedLocator

ax[1, 1].plot(epochs, dev_map, 'g', linewidth=2)
ax[1, 1].set_xlabel("Epochs")
ax[1, 1].set_ylabel("MAP score on Dev Set")
# ax[1, 1].set_xticks([k for k in epochs if k%5==0])
# ax[1, 1].set_xticklabels(xticks, rotation=90, ha="right")#-->FixedLocator
# plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
fig.suptitle("Summary of Model performance", fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
if os.path.exists("./plots/"):
    pass
else:
    os.mkdir("./plots/")
plt.savefig(f"./plots/{args.model_name}.png", dpi=200)  #

# https: // stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator

# Save params .txt file.
# Run by:
# python plotter.py --log_file ckpts/ckpt_medical/log_1.txt --model_name medical1
