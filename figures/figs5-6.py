import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.style.use("../molecular_encodings/molecular.mplstyle")

source_file = "../data/paper_classification/master_scores.xlsx"



def get_data(runid: int):

    dataframe = pd.read_excel(source_file, sheet_name=f"Run{runid:d}")

    output = {
        "num_layers": sorted(dataframe["num_ansatz_layers"].unique()),
        "train_mean": [],
        "train_std+": [],
        "train_std-": [],
        "test_mean": [],
        "test_std+": [],
        "test_std-": [],
    }

    for n_layers in output["num_layers"]:
        output["train_mean"].append(
            (
                dataframe
                .loc[dataframe["num_ansatz_layers"] == n_layers]["training_score"]
                .median()
            )
        )
        output["train_std+"].append(
            (
                dataframe
                .loc[dataframe["num_ansatz_layers"] == n_layers]["training_score"]
                .quantile(0.84)
            ) -
            (
                dataframe
                .loc[dataframe["num_ansatz_layers"] == n_layers]["training_score"]
                .median()
            )
        )
        output["train_std-"].append(
            (
                dataframe
                .loc[dataframe["num_ansatz_layers"] == n_layers]["training_score"]
                .median()
            ) -
            (
                dataframe
                .loc[dataframe["num_ansatz_layers"] == n_layers]["training_score"]
                .quantile(0.16)
            )
        )
        output["test_mean"].append(
            (
                dataframe
                .loc[dataframe["num_ansatz_layers"] == n_layers]["test_score"]
                .median()
            )
        )
        output["test_std+"].append(
            (
                dataframe
                .loc[dataframe["num_ansatz_layers"] == n_layers]["test_score"]
                .quantile(0.84)
            ) -
            (
                dataframe
                .loc[dataframe["num_ansatz_layers"] == n_layers]["test_score"]
                .median()
            )
        )
        output["test_std-"].append(
            (
                dataframe
                .loc[dataframe["num_ansatz_layers"] == n_layers]["test_score"]
                .median()
            ) -
            (
                dataframe
                .loc[dataframe["num_ansatz_layers"] == n_layers]["test_score"]
                .quantile(0.16)
            )
        )

    for k, v in output.items():
        output[k] = np.asarray(v)

    return output


# fig, axes = plt.subplots(2, 2, figsize=(2.5 * 2, 5), constrained_layout=True, sharex=True, sharey='row')
#
# # Add styling and labels
# data = get_data(1)
#
# for ax in axes.flat:
#     ax.axhline(1, lw=2, c='k', ls='--')
#     ax.set_xticks(data["num_layers"], minor=True)
#     ax.tick_params(axis='x', which='both', top=False)
#
#     for i in range(1, 6):
#         if i % 2 == 1:
#             ax.axvspan(i - 0.5, i + 0.5, fc='grey', alpha=0.15, ec='none')
#
# ax = axes[0, 0]
# ax.tick_params(axis='y', which='both', right=False)
# ax.set_ylabel('Median accuracy score (train)', fontsize="medium")
#
# ax = axes[1, 0]
# ax.tick_params(axis='y', which='both', right=False)
# ax.set_ylabel('Median accuracy score (test)', fontsize="medium")
# ax.set_xlabel('$L_{\\theta}$', fontsize="large")
#
# ax = axes[1, 1]
# ax.set_xlabel('$L_{\\theta}$', fontsize="large")
#
# ax = axes[0, 0]
# ax.set_title('a)', loc='left')
# ax.set_title('Alkanes subdataset classification\n',)
#
# # Add simulation data
# data = get_data(1)
# ax.errorbar(data["num_layers"] - 0.36, data["train_mean"], yerr=[data["train_std-"], data["train_std+"]], marker='o', ms=5, ls='none', c="tab:red", mec="w", mew=0.4)
#
# data = get_data(2)
# ax.errorbar(data["num_layers"] + 0.12, data["train_mean"], yerr=[data["train_std-"], data["train_std+"]], marker='o', ms=5, ls='none', c="tab:blue", mec="w", mew=0.4)
#
# data = get_data(3)
# ax.errorbar(data["num_layers"] - 0.12, data["train_mean"], yerr=[data["train_std-"], data["train_std+"]], marker='s', ms=5, ls='none', c="tab:red", mec="w", mew=0.4)
#
# data = get_data(4)
# ax.errorbar(data["num_layers"] + 0.36, data["train_mean"], yerr=[data["train_std-"], data["train_std+"]], marker='s', ms=5, ls='none', c="tab:blue", mec="w", mew=0.4)
#
# handles = [
#     Patch(color='tab:red', label='Linear entanglement'),
#     Patch(color='tab:blue', label='Pairwise entanglement'),
#     Line2D([0], [0], label='Fingerprint', marker='o', markersize=6, markeredgecolor='none', markerfacecolor='k', linestyle=''),
#     Line2D([0], [0], label='QMSE (this work)', marker='s', markersize=5, markeredgecolor='none', markerfacecolor='k', linestyle=''),
# ]
# ax.set_ylim(0.6, 1.015)
# ax.legend(handles=handles, fontsize="small", ncol=1, loc="lower right", handlelength=1.5)
#
# ax = axes[1, 0]
# ax.set_title('b)', loc='left')
#
# data = get_data(1)
# ax.errorbar(data["num_layers"] - 0.36, data["test_mean"], yerr=[data["test_std-"], data["test_std+"]], marker='o', ms=5, ls='none', c="tab:red", mec="w", mew=0.4)
#
# data = get_data(2)
# ax.errorbar(data["num_layers"] + 0.12, data["test_mean"], yerr=[data["test_std-"], data["test_std+"]], marker='o', ms=5, ls='none', c="tab:blue", mec="w", mew=0.4)
#
#
# data = get_data(3)
# ax.errorbar(data["num_layers"] - 0.12, data["test_mean"], yerr=[data["test_std-"], data["test_std+"]], marker='s', ms=5, ls='none', c="tab:red", mec="w", mew=0.4)
#
# data = get_data(4)
# ax.errorbar(data["num_layers"] + 0.36, data["test_mean"], yerr=[data["test_std-"], data["test_std+"]], marker='s', ms=5, ls='none', c="tab:blue", mec="w", mew=0.4)
#
#
# ax = axes[0, 1]
# ax.set_title('c)', loc='left')
# ax.set_title('Complete dataset classification\n')
#
# data = get_data(5)
# ax.errorbar(data["num_layers"] - 0.40 * 0.8, data["train_mean"], yerr=[data["train_std-"], data["train_std+"]], marker='v', ms=5, ls='none', c="k", mec="w", mew=0.4)
#
# data = get_data(6)
# ax.errorbar(data["num_layers"] - 0.20 * 0.8, data["train_mean"], yerr=[data["train_std-"], data["train_std+"]], marker='o', ms=5, ls='none', c="k", mec="w", mew=0.4)
#
# data = get_data(7)
# ax.errorbar(data["num_layers"] + 0. * 0.8, data["train_mean"], yerr=[data["train_std-"], data["train_std+"]], marker='o', ms=5, ls='none', c="tab:red", mec="w", mew=0.4)
#
# data = get_data(8)
# ax.errorbar(data["num_layers"] + 0.20 * 0.8, data["train_mean"], yerr=[data["train_std-"], data["train_std+"]], marker='o', ms=5, ls='none', c="tab:green", mec="w", mew=0.4)
#
# data = get_data(9)
# ax.errorbar(data["num_layers"] + 0.40 * 0.8, data["train_mean"], yerr=[data["train_std-"], data["train_std+"]], marker='o', ms=5, ls='none', c="tab:blue", mec="w", mew=0.4)
#
#
# ax.set_ylim(0.6, 1.015)
#
# ax = axes[1, 1]
# ax.set_title('d)', loc='left')
#
# data = get_data(5)
# ax.errorbar(data["num_layers"] - 0.40 * 0.8, data["test_mean"], yerr=[data["test_std-"], data["test_std+"]], marker='v', ms=5, ls='none', c="k", mec="w", mew=0.4)
#
# data = get_data(6)
# ax.errorbar(data["num_layers"] - 0.20 * 0.8, data["test_mean"], yerr=[data["test_std-"], data["test_std+"]], marker='o', ms=5, ls='none', c="k", mec="w", mew=0.4)
#
# data = get_data(7)
# ax.errorbar(data["num_layers"] + 0. * 0.8, data["test_mean"], yerr=[data["test_std-"], data["test_std+"]], marker='o', ms=5, ls='none', c="tab:red", mec="w", mew=0.4)
#
# data = get_data(8)
# ax.errorbar(data["num_layers"] + 0.20 * 0.8, data["test_mean"], yerr=[data["test_std-"], data["test_std+"]], marker='o', ms=5, ls='none', c="tab:green", mec="w", mew=0.4)
#
# data = get_data(9)
# ax.errorbar(data["num_layers"] + 0.40 * 0.8, data["test_mean"], yerr=[data["test_std-"], data["test_std+"]], marker='o', ms=5, ls='none', c="tab:blue", mec="w", mew=0.4)
#
#
# handles = [
#     Line2D([0], [0], label=r'$CZ$ (run 5)', marker='v', markersize=6, markeredgecolor='none', markerfacecolor='k', linestyle=''),
#     Line2D([0], [0], label=r'$CRX$ (run 6)', marker='o', markersize=6, markeredgecolor='none', markerfacecolor='k', linestyle=''),
# ]
# legend = ax.legend(handles=handles, fontsize="small", ncol=1, loc="lower left", handlelength=1, title="2-qubit gate\n(global $\\mathcal{H}$)",
#                    bbox_to_anchor=(0.2, 0))
# ax.add_artist(legend)
#
# handles = [
#     Line2D([0], [0], label=r'$IIIIZZIIII$ (run 7)', marker='o', markersize=6, markeredgecolor='none', markerfacecolor='tab:red', linestyle=''),
#     Line2D([0], [0], label=r'$IIIZZZZIII$ (run 8)', marker='o', markersize=6, markeredgecolor='none', markerfacecolor='tab:green', linestyle=''),
#     Line2D([0], [0], label=r'$ZZIIIIIIII$ (run 9)', marker='o', markersize=6, markeredgecolor='none', markerfacecolor='tab:blue', linestyle=''),
# ]
# legend = ax.legend(handles=handles, fontsize="small", ncol=1, loc="lower right", handlelength=1, title=r"Local $\mathcal{H}$")
# ax.add_artist(legend)
#
# plt.savefig("fig5.pdf")
# plt.show()



fig, axes = plt.subplots(1, 2, figsize=(2.5 * 2, 2.5), constrained_layout=True, sharex=True)

# Add styling and labels
data = get_data(1)

for ax in axes.flat:
    ax.axhline(1, lw=2, c='k', ls='--')
    ax.set_xticks(data["num_layers"], minor=True)
    ax.tick_params(axis='x', which='both', top=False)
    ax.tick_params(axis='y', which='both', right=False)
    ax.set_xlabel('$L_{\\theta}$', fontsize="large")
    ax.set_xticks([1, 2, 3, 4, 5, 6])

    for i in range(1, 6):
        if i % 2 == 1:
            ax.axvspan(i - 0.5, i + 0.5, fc='grey', alpha=0.15, ec='none')

ax = axes[0]
ax.set_ylabel(r'Median $R^2$ (train)', fontsize="medium")

ax = axes[1]
ax.set_ylabel(r'Median $R^2$ (test)', fontsize="medium")

ax = axes[0]
ax.set_title('a)', loc='left')
# ax.set_title('Complete dataset regression\n',)

# Add simulation data
data = get_data(10)
ax.errorbar(data["num_layers"] - 0.15, data["train_mean"], yerr=[data["train_std-"], data["train_std+"]], marker='^', ms=6, ls='none', c="tab:red", mec="w", mew=0.4)

data = get_data(11)
ax.errorbar(data["num_layers"] + 0.15, data["train_mean"], yerr=[data["train_std-"], data["train_std+"]], marker='s', ms=5, ls='none', c="tab:blue", mec="w", mew=0.4)

handles = [
    Line2D([0], [0], label='Pairwise entanglement', marker='^', markersize=6, markeredgecolor='none', markerfacecolor='tab:red', linestyle=''),
    Line2D([0], [0], label='Full entanglement', marker='s', markersize=5, markeredgecolor='none', markerfacecolor='tab:blue', linestyle=''),
]
ax.legend(handles=handles, fontsize="small", ncol=1, loc="lower right", handlelength=1.5)

ax = axes[1]
ax.set_title('b)', loc='left')

data = get_data(10)
ax.errorbar(data["num_layers"] - 0.15, data["test_mean"], yerr=[data["test_std-"], data["test_std+"]], marker='^', ms=6, ls='none', c="tab:red", mec="w", mew=0.4)

data = get_data(11)
ax.errorbar(data["num_layers"] + 0.15, data["test_mean"], yerr=[data["test_std-"], data["test_std+"]], marker='s', ms=5, ls='none', c="tab:blue", mec="w", mew=0.4)

plt.savefig("fig6.pdf")
plt.show()