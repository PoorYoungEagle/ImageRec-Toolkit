import matplotlib.pyplot as plt

def plot_lr_finder(lr_results):
    """
    Plot learning rate vs loss.
    
    Args:
        lr_results(dict): Dictionary containing learning rate and loss.
            - lrs (list): Learning rates tested.
            - losses (list): Corresponding loss values.
    """

    plt.figure(figsize=(8,5))
    plt.plot(lr_results["lrs"], lr_results["losses"])
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("LR Finder Curve")
    plt.show()