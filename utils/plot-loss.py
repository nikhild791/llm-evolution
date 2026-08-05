def plot_train_loss(loss):
    x_a = [i for i in range(len(loss)-1)]
    train_l = train_history["train_loss"]
    val_l = train_history["val_loss"]
    plt.title("Train val loss", loc="left")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(x_a, train_l, label="Training loss")
    plt.plot(x_a, val_l, linestyle="-." ,label="Validation loss")
    plt.legend(loc="upper right")
    plt.show()
plot_train_loss(train_history)