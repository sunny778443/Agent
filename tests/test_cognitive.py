"""
Unit and Integration tests for the Neural Cognitive Module.
Verifies from-scratch matrix, Layer, attention, GCN, LSTM, and CodeCognitiveNetwork states.
"""
import unittest
import math
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
    train_evolutionary_strategy
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

if __name__ == "__main__":
    unittest.main()
