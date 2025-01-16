import matplotlib.pyplot as plt

def plot_metrics(metrics):
    """
    Plots benchmark metrics as a bar chart.

    Args:
        metrics (dict): Dictionary of benchmark metrics.
    """
    plt.figure(figsize=(10, 6))
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.bar(names, values)
    plt.title("Model Performance Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
