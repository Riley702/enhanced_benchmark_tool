from enhanced_benchmark_tool.visualizations import plot_metrics

def test_plot_metrics():
    metrics = {"accuracy": 0.95, "precision": 0.92, "recall": 0.93, "f1_score": 0.94}
    plot_metrics(metrics)
    print("test_plot_metrics passed.")
