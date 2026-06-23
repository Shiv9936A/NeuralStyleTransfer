import json
import matplotlib.pyplot as plt

def generate_loss_graph(style_name):

    with open(
        f"loss_history_{style_name}.json",
        "r"
    ) as f:

        losses = json.load(f)

    plt.figure(figsize=(8,4))

    plt.plot(losses)

    plt.title(
        f"{style_name} Training Loss"
    )

    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.grid(True)

    output_path = (
        f"saved_models/{style_name}_curve.png"
    )

    plt.savefig(output_path)

    plt.close()

    return output_path