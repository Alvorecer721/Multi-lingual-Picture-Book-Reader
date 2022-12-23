matches = ["Original:", "Decoded:", "Epoch "]
new_lines = []

# Read specific row from txt and store in list
with open('read.txt', 'r', encoding='UTF-8') as fr:
    lines = fr.readlines()
    for l in lines:
        if any(x in l for x in matches):
            new_lines.append(l)

# Write to file
fw = open('out.txt', 'w+', encoding='UTF-8')

for idx, nl in enumerate(new_lines):
    fw.write(nl)
    if (idx+1) % len(matches) == 0:
        fw.write("\n")

fw.close()

# Store result string in list
result_string = []
lines = open('out.txt', 'r', encoding='UTF-8').readlines()
for idx, l in enumerate(lines):
    if "Epoch " in l:
        result_string.append(l)

train_cost = []
train_label_err = []
for i in range(len(result_string)):
    results = result_string[i].split(", ")
    train_cost.append(float(results[1].split(": ")[1]) / 250)
    train_label_err.append(1 - float(results[2].split(": ")[1]))

epoch = list(range(50))

import matplotlib.pyplot as plt

plt.style.use('seaborn')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
axes[0].plot(epoch, train_cost, linestyle="--", color="C1")
axes[1].plot(epoch, train_label_err, linestyle="--", Color="C2")

axes[0].set_title("Training Loss per Epoch")
axes[1].set_title("Training Label Accuracy per Epoch")

fig = plt.gcf()
fig.suptitle("Bidirectional RNN Speech Recognition Model Performance", fontsize=14)
plt.savefig("foo.png", dpi=150)
plt.show()

