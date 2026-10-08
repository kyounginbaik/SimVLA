"""Keep manual retry resets consistent with the public recorder lifecycle."""


def ensure_initial_snapshot(recorder, env_ids):
    """Record missing post-reset state once, before an episode's first action.

    The collector's private `_reset_idx` retry path clears recorder buffers but
    bypasses the public reset's post-reset callback. Do not duplicate snapshots
    on the ordinary reset path: exporters require exactly one initial state.
    """
    missing = [int(index) for index in env_ids
               if "initial_state" not in recorder.get_episode(int(index)).data]
    if missing:
        recorder.record_post_reset(missing)
