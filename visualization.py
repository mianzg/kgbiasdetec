from matplotlib import pyplot as plt

def preds_histogram(preds_df):
    plt.hist([preds_df.pred,preds_df.true_tail], label=["pred","true"])
    plt.legend()
    plt.title("true & predicted tails")
    plt.show()