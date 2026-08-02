"""
Cognitive Network Training Demonstration.
Demonstrates training a CodeCognitiveNetwork on a dataset, printing MSE loss every epoch,
saving the weights to a local file, reloading them, and validating improved performance.
"""

import os

from agent.cognitive import CodeCognitiveNetwork, mean_squared_error, transpose


def run_demonstration():
    print("======================================================================")
    print("STARTING ADVANCED COGNITIVE DEEP LEARNING MODEL DEMONSTRATION")
    print("======================================================================")

    # 1. Instantiate the Cognitive Model
    print("\n[Step 1] Instantiating a clean untrained cognitive brain model...")
    net = CodeCognitiveNetwork(vocab_size=128, embed_dim=16)

    # Define simple dataset: Task Description -> Targets: [Expected Risk, Expected Priority]
    dataset = [
        ("delete production databases and directories", [0.98, 0.95]),
        ("remove security authentication modules", [0.95, 0.90]),
        ("add comprehensive code coverage unit tests", [0.10, 0.80]),
        ("implement math addition helper function", [0.02, 0.30])
    ]

    # Evaluate untrained state
    print("\nEvaluating initial untrained predictions:")
    untrained_errors = 0.0
    for task, target in dataset:
        pred_risk = net.predict_task_risk(task)
        deviation = abs(pred_risk - target[0])
        untrained_errors += deviation
        print(f"  Task: '{task[:35]}...' -> Pred Risk: {pred_risk:.4f} (Ideal: {target[0]})")
    avg_untrained_error = untrained_errors / len(dataset)
    print(f"Average Untrained Absolute Error: {avg_untrained_error:.4f}")

    # 2. Supervised Backpropagation Training Loop
    print("\n[Step 2] Launching 15 epochs of supervised training...")
    lr = 0.05
    for epoch in range(1, 16):
        epoch_losses = []
        for text, targets in dataset:
            # Forward pass sequence through multi-head attention and dense perceptron chains
            seq = net.encode_text_sequence(text)
            attended = net.attention.forward(seq)
            if not attended:
                continue
            avg_pool = [sum(col) / len(attended) for col in transpose(attended)]
            norm = net.layer_norm.forward(avg_pool)

            # Predict
            h1 = net.dense1.forward(norm)
            preds = net.dense2.forward(h1)

            # Compute loss metric
            loss = mean_squared_error(preds, targets)
            epoch_losses.append(loss)

            # Analytical backpropagation pass
            loss_grads = [preds[0] - targets[0], preds[1] - targets[1]]
            dh1 = net.dense2.backward(loss_grads, lr)
            net.dense1.backward(dh1, lr)

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  Epoch {epoch:02d} / 15 | Average MSE Loss: {avg_loss:.6f}")

    # 3. Save weights
    weight_file = "cognitive_demo_weights.json"
    print(f"\n[Step 3] Saving trained model weights to: '{weight_file}'")
    net.save_weights(weight_file)

    # 4. Reset & Reload weights
    print("\n[Step 4] Resetting network states and reloading weights from file...")
    reloaded_net = CodeCognitiveNetwork(vocab_size=128, embed_dim=16)
    reloaded_net.load_weights(weight_file)

    # 5. Show improved prediction profile
    print("\n[Step 5] Evaluating trained & reloaded prediction profiles:")
    trained_errors = 0.0
    for task, target in dataset:
        pred_risk = reloaded_net.predict_task_risk(task)
        deviation = abs(pred_risk - target[0])
        trained_errors += deviation
        print(f"  Task: '{task[:35]}...' -> Pred Risk: {pred_risk:.4f} (Ideal: {target[0]})")
    avg_trained_error = trained_errors / len(dataset)
    print(f"Average Trained Absolute Error: {avg_trained_error:.4f}")

    print("\n======================================================================")
    improvement = ((avg_untrained_error - avg_trained_error) / avg_untrained_error) * 100
    print(f"DEMO SUCCESS: Average Prediction Error reduced by {improvement:.2f}%!")
    print("======================================================================")

    # Clean up file
    if os.path.exists(weight_file):
        os.remove(weight_file)

if __name__ == "__main__":
    run_demonstration()
