"""``_predict_hierarchical`` must combine tensors on the device of its input rows."""

import pytest

torch = pytest.importorskip("torch")

from tababax._model.layers import ClassNode
from tababax._model.learning import ICLearning


def _tree():
    """Root over two groups; each group is a leaf holding two of the four classes."""
    root = ClassNode(depth=0)
    root.classes_ = [0, 1, 2, 3]
    root.R = torch.zeros(4, 2)
    root.group_indices = torch.tensor([0, 0, 1, 1])
    for group, classes in enumerate([[0, 1], [2, 3]]):
        leaf = ClassNode(depth=1)
        leaf.is_leaf = True
        leaf.classes_ = classes
        leaf.R = torch.zeros(2, 2)
        leaf.y = torch.tensor(classes)
        root.child_nodes.append(leaf)
    return root


def _model():
    model = ICLearning.__new__(ICLearning)
    torch.nn.Module.__init__(model)
    model.root = _tree()

    def predict_standard(R, y_train, **kwargs):
        # Like InferenceManager with offload="auto", the output may live on another device than R.
        return torch.full((1, R.shape[1] - y_train.shape[1], 2), 0.5)

    model._predict_standard = predict_standard
    model._label_encoding = lambda y: y  # torch.unique has no kernel for the meta device
    return model


def test_predict_hierarchical_runs_on_the_input_device():
    model = _model()
    # The "meta" device stands in for a second device: a CPU tensor cannot be mixed with it, as it
    # cannot with CUDA, and this runs without a GPU.
    probs = model._predict_hierarchical(torch.zeros(3, 2, device="meta"))
    assert probs.device.type == "meta"
    assert probs.shape == (3, 4)


def test_predict_hierarchical_values_on_cpu():
    probs = _model()._predict_hierarchical(torch.zeros(3, 2))
    # Two groups at 0.5, two classes in each leaf at 0.5.
    assert torch.allclose(probs, torch.full((3, 4), 0.25))
