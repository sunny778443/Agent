"""
Unit and Integration tests for the Neural Cognitive Module.
Verifies from-scratch matrix, Layer, attention, GCN, LSTM, and CodeCognitiveNetwork states.
"""
import unittest

from agent.cognitive import (
    CodeCognitiveNetwork,
    CodeHeuristicRanker,
    DeepCognitiveBlock,
    DenseLayer,
    GraphConvolution,
    LSTMCell,
    SelfAttention,
    dot_product,
    matrix_multiply,
    matrix_vector_multiply,
    sigmoid,
    softmax,
    transpose,
    vector_add,
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

    def test_self_attention(self):
        attn = SelfAttention(embed_dim=4)
        seq = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        out = attn.forward(seq)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 4)

    def test_graph_convolution(self):
        gcn = GraphConvolution(in_features=3, out_features=2)
        node_features = [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]
        adj_matrix = [[1.0, 1.0], [1.0, 1.0]]
        out = gcn.forward(node_features, adj_matrix)
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
        self.assertEqual(len(metrics["main.py"]), 2) # [bug_risk, importance]

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
            [0.0, 1.0, 0.0, 0.0], # unrelated
            [0.9, 0.1, 0.0, 0.0]  # highly similar
        ]
        ranks = ranker.rank_candidates(bug, patches)
        self.assertEqual(len(ranks), 2)

if __name__ == "__main__":
    unittest.main()
