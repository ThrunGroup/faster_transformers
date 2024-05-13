import matplotlib.pyplot as plt

# Data setup
layer_indices = list(range(32))
cats_50_sparsity = [
    35.765,
    35.155,
    54.565,
    45.684,
    40.666,
    37.898,
    34.639,
    32.378,
    32.406,
    31.375,
    33.385,
    33.794,
    35.858,
    34.143,
    32.993,
    34.598,
    35.585,
    39.706,
    40.587,
    41.302,
    40.498,
    43.413,
    44.748,
    48.473,
    48.376,
    47.112,
    44.733,
    45.636,
    42.981,
    44.842,
    54.396,
    68.841,
]
cats_70_sparsity = [
    53.039,
    53.228,
    77.561,
    65.718,
    60.079,
    56.781,
    52.372,
    48.606,
    48.014,
    47.400,
    50.333,
    51.176,
    51.483,
    51.487,
    47.645,
    51.022,
    52.087,
    59.440,
    58.227,
    60.934,
    60.394,
    63.718,
    64.439,
    68.146,
    66.913,
    66.697,
    67.048,
    68.402,
    67.738,
    71.307,
    87.651,
    90.158,
]
cats_90_sparsity = [
    78.861,
    79.481,
    96.167,
    88.661,
    86.029,
    80.718,
    78.125,
    74.026,
    73.138,
    71.282,
    74.193,
    75.714,
    75.440,
    74.706,
    72.911,
    76.791,
    78.989,
    83.385,
    83.018,
    82.552,
    84.970,
    83.900,
    83.489,
    82.921,
    82.522,
    82.238,
    88.959,
    88.983,
    92.817,
    93.832,
    96.082,
    95.717,
]

labels = ["CATS 50%", "CATS 70%", "CATS 90%"]
total_sparsity = [41.142, 60.567, 82.519]
colors = ["#227CF6", "#2BA02B", "#D62727"]

# Plotting
plt.figure(figsize=(10, 8))

# Layer sparsity plot
# plt.subplot(1, 2, 1)
plt.plot(
    layer_indices,
    cats_50_sparsity,
    label="CATS 50%",
    marker="o",
    color=colors[0],
)
plt.plot(
    layer_indices,
    cats_70_sparsity,
    label="CATS 70%",
    marker="o",
    color=colors[1],
)
plt.plot(
    layer_indices,
    cats_90_sparsity,
    label="CATS 90%",
    marker="o",
    color=colors[2],
)

# for i in range(3):
#     plt.plot(
#         layer_indices,
#         [total_sparsity[i]] * len(layer_indices),
#         label=labels[i],
#         linestyle="--",
#         color=colors[i],
#         linewidth=2,
#     )
plt.hlines(
    total_sparsity[0],
    layer_indices[0],
    layer_indices[-1],
    colors=colors[0],
    linestyles="dashed",
    label="Total CATS 50% Sparsity",
)
plt.hlines(
    total_sparsity[1],
    layer_indices[0],
    layer_indices[-1],
    colors=colors[1],
    linestyles="dashed",
    label="Total CATS 70% Sparsity",
)
plt.hlines(
    total_sparsity[2],
    layer_indices[0],
    layer_indices[-1],
    colors=colors[2],
    linestyles="dashed",
    label="Total CATS 90% Sparsity",
)

for i in range(3):
    plt.annotate(
        f"{total_sparsity[i]:.2f}%",
        (layer_indices[0], total_sparsity[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

# plt.title("Layer-wise Sparsity")
plt.xlabel("Layer Index")
plt.ylabel("Sparsity (%)")
plt.legend()

plt.tight_layout()
plt.show()
