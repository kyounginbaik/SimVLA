import numpy as np

from hdf5_compat import write_success_if_missing


class Handler:
    def __init__(self, count):
        self.count = count
        self.writes = []
        self.flushes = 0

    def get_num_episodes(self):
        return self.count

    def write_episode(self, episode):
        self.writes.append(episode)
        self.count += 1

    def flush(self):
        self.flushes += 1


class Manager:
    def __init__(self, count):
        self._dataset_file_handler = Handler(count)


def test_manager_already_wrote_episode_does_not_duplicate():
    manager = Manager(count=1)
    assert not write_success_if_missing(
        manager, "episode", succeeded=True, expected_success_count=1
    )
    assert manager._dataset_file_handler.writes == []
    assert manager._dataset_file_handler.flushes == 0


def test_manager_that_only_returned_episode_gets_one_write(monkeypatch):
    import sys
    from types import SimpleNamespace

    fake_torch = SimpleNamespace(
        Tensor=type("FakeTensor", (), {}),
        from_numpy=lambda value: value,
        stack=lambda values: np.stack(values),
        as_tensor=lambda value: np.asarray(value),
        empty=lambda size: np.empty(size),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    manager = Manager(count=0)
    class Episode:
        success = True
        data = {
            "actions": {"ee_6D_pos": [np.array([4.0])]},
            "raw_actions": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        }

    assert write_success_if_missing(
        manager, Episode(), succeeded=True, expected_success_count=1
    )
    written = manager._dataset_file_handler.writes[0]
    assert np.array_equal(written.data["actions"], [[1.0, 2.0], [3.0, 4.0]])
    assert manager._dataset_file_handler.flushes == 1


def test_unsuccessful_episode_is_never_written():
    manager = Manager(count=0)
    assert not write_success_if_missing(
        manager, "episode", succeeded=False, expected_success_count=1
    )
    assert manager._dataset_file_handler.writes == []
    assert manager._dataset_file_handler.flushes == 0
