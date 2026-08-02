"""
Unit and Integration tests for the Neural Cognitive Module.
Verifies from-scratch matrix, Layer, attention, GCN, LSTM, and CodeCognitiveNetwork states.
"""
import unittest
import math
import os
from agent.cognitive import (
    dot_product,
    vector_add,
    matrix_multiply,
    matrix_vector_multiply,
    transpose,
    sigmoid,
    softmax,
    DenseLayer,
    MultiHeadAttention,
    GraphAttentionLayer,
    LSTMCell,
    CodeCognitiveNetwork,
    DeepCognitiveBlock,
    CodeHeuristicRanker,
    SGDMomentum,
    RMSprop,
    Adam,
    huber_loss,
    cross_entropy_loss,
    train_unsupervised_mlm,
    train_reinforcement_learning,
    train_evolutionary_strategy,
    mean_squared_error
)

class TestCognitiveModule(unittest.TestCase):
    def test_basic_linear_algebra(self):
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]
        self.assertAlmostEqual(dot_product(v1, v2), 32.0)
        self.assertEqual(vector_add(v1, v2), [5.0, 7.0, 9.0])

        m1 = [[1.0, 2.0], [3.0, 4.0]]
        m2 = [[2.0, 0.0], [1.0, 2.0]]
        expected = [[4.0, 4.0], [10.0, 8.0]]
        self.assertEqual(matrix_multiply(m1, m2), expected)

        v = [2.0, 3.0]
        self.assertEqual(matrix_vector_multiply(m1, v), [8.0, 18.0])
        self.assertEqual(transpose(m1), [[1.0, 3.0], [2.0, 4.0]])

    def test_activations(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertGreater(sigmoid(5.0), 0.9)
        self.assertLess(sigmoid(-5.0), 0.1)

        v = [1.0, 2.0, 3.0]
        s = softmax(v)
        self.assertAlmostEqual(sum(s), 1.0)
        self.assertGreater(s[2], s[0])

    def test_dense_layer_backprop(self):
        layer = DenseLayer(in_features=3, out_features=2, activation="relu")
        inputs = [1.0, -1.0, 0.5]
        out = layer.forward(inputs)
        self.assertEqual(len(out), 2)

        # Backward propagation update
        d_out = [0.1, -0.2]
        d_in = layer.backward(d_out, lr=0.01)
        self.assertEqual(len(d_in), 3)

    def test_multi_head_attention(self):
        attn = MultiHeadAttention(embed_dim=4, num_heads=2)
        seq = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        out = attn.forward(seq)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 4)

    def test_graph_attention(self):
        gat = GraphAttentionLayer(in_features=3, out_features=2)
        node_features = [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]
        adj_matrix = [[1.0, 1.0], [1.0, 1.0]]
        out = gat.forward(node_features, adj_matrix)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 2)

    def test_lstm_cell(self):
        lstm = LSTMCell(in_dim=3, hidden_dim=2)
        x = [1.0, 0.0, -1.0]
        h = [0.0, 0.0]
        c = [0.0, 0.0]
        h_next, c_next = lstm.step(x, h, c)
        self.assertEqual(len(h_next), 2)
        self.assertEqual(len(c_next), 2)

    def test_cognitive_network_predictions(self):
        net = CodeCognitiveNetwork(vocab_size=128, embed_dim=8)

        # Test text risk scoring
        risk = net.predict_task_risk("implement safety filters for authentication module")
        self.assertTrue(0.0 <= risk <= 1.0)

        # Test project file scanning evaluation
        file_contents = {
            "main.py": "def foo(): pass",
            "utils.py": "def bar(): return 42"
        }
        dependencies = {
            "main.py": ["utils.py"]
        }
        metrics = net.process_project_dependency_graph(file_contents, dependencies)
        self.assertIn("main.py", metrics)
        self.assertIn("utils.py", metrics)
        self.assertEqual(len(metrics["main.py"]), 2)

    def test_deep_cognitive_block_norm(self):
        block = DeepCognitiveBlock(embed_dim=4)
        seq = [[1.0, 2.0, -1.0, 0.5]]
        out = block.forward(seq)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 4)

    def test_heuristic_ranker(self):
        ranker = CodeHeuristicRanker(feature_dim=4)
        bug = [1.0, 0.0, 0.0, 0.0]
        patches = [
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0]
        ]
        ranks = ranker.rank_candidates(bug, patches)
        self.assertEqual(len(ranks), 2)

    def test_optimizers_and_losses(self):
        # test Huber Loss
        self.assertAlmostEqual(huber_loss([1.0], [1.5]), 0.125)
        # test Cross entropy
        self.assertGreater(cross_entropy_loss([0.1, 0.9], [0.0, 1.0]), 0.0)

        # verify optimizer classes initialization
        opt_momentum = SGDMomentum()
        opt_rmsprop = RMSprop()
        opt_adam = Adam()
        self.assertEqual(opt_momentum.lr, 0.01)
        self.assertEqual(opt_rmsprop.lr, 0.001)
        self.assertEqual(opt_adam.lr, 0.001)

    def test_multi_paradigm_trainings(self):
        net = CodeCognitiveNetwork(vocab_size=128, embed_dim=8)

        # Test Unsupervised Masked Language Modeling
        corpus = ["def hello(): return 'world'", "class Matrix: pass", "import os"]
        train_unsupervised_mlm(net, corpus, epochs=2)

        # Test Reinforcement Learning
        train_reinforcement_learning(net, episodes=2)

        # Test Evolutionary Strategies
        train_evolutionary_strategy(net, population_size=4, generations=2)

        # Verify predictions still output valid probability scales
        risk = net.predict_task_risk("delete repository")
        self.assertTrue(0.0 <= risk <= 1.0)

    def test_demonstrated_weights_updates_and_loss_reduction(self):
        """Autotests verifying training updates weight states and dynamically reduces prediction loss."""
        net = CodeCognitiveNetwork(vocab_size=128, embed_dim=8)

        dataset = [
            ("drop prod tables", [0.99, 0.90]),
            ("add standard logging framework", [0.05, 0.40])
        ]

        # Get baseline untrained absolute loss
        baseline_err = 0.0
        for task, target in dataset:
            baseline_err += abs(net.predict_task_risk(task) - target[0])

        # Get baseline weights signature
        dense2_orig_weights = [[x for x in row] for row in net.dense2.weights]

        # Train for 5 quick epochs
        lr = 0.1
        for epoch in range(5):
            for task, target in dataset:
                seq = net.encode_text_sequence(task)
                attended = net.attention.forward(seq)
                if not attended:
                    continue
                avg_pool = [sum(col) / len(attended) for col in transpose(attended)]
                norm = net.layer_norm.forward(avg_pool)
                h1 = net.dense1.forward(norm)
                preds = net.dense2.forward(h1)

                loss_grads = [preds[0] - target[0], preds[1] - target[1]]
                dh1 = net.dense2.backward(loss_grads, lr)
                net.dense1.backward(dh1, lr)

        # 1. Assert weights changed
        updated_weights = net.dense2.weights
        self.assertNotEqual(dense2_orig_weights, updated_weights)

        # 2. Assert error on target set is reduced after backpropagation
        trained_err = 0.0
        for task, target in dataset:
            trained_err += abs(net.predict_task_risk(task) - target[0])

        self.assertLess(trained_err, baseline_err)

        # 3. Verify weights persistence (save/load)
        weight_filepath = "test_persistence.json"
        net.save_weights(weight_filepath)
        self.assertTrue(os.path.exists(weight_filepath))

        # Reload
        reloaded = CodeCognitiveNetwork(vocab_size=128, embed_dim=8)
        reloaded.load_weights(weight_filepath)

        for task, _ in dataset:
            self.assertAlmostEqual(net.predict_task_risk(task), reloaded.predict_task_risk(task))

        # clean file
        if os.path.exists(weight_filepath):
            os.remove(weight_filepath)

if __name__ == "__main__":
    unittest.main()
