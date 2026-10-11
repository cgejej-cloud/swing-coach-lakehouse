from swing_coach.store import Store


def add(store: Store, digest: str, session: str = "s1", shot_at: str = "2026-10-07 10:00:00"):
    return store.add_video(content_hash=digest, path=f"/v/{digest}.mp4", source="upload",
                           session_key=session, session_label="t", shot_at=shot_at)


def test_same_content_is_not_queued_twice(tmp_path):
    store = Store(tmp_path / "app.db")
    first, created = add(store, "abc")
    again, created_again = add(store, "abc")
    assert created and not created_again and first == again


def test_seq_no_follows_upload_order_within_session(tmp_path):
    store = Store(tmp_path / "app.db")
    a, _ = add(store, "a")
    b, _ = add(store, "b")
    other, _ = add(store, "c", session="s2")
    with store.conn() as c:
        seq = {r["id"]: r["seq_no"] for r in c.execute("SELECT id, seq_no FROM videos")}
    assert (seq[a], seq[b], seq[other]) == (1, 2, 1)


def test_claim_next_is_fifo_and_marks_processing(tmp_path):
    store = Store(tmp_path / "app.db")
    a, _ = add(store, "a")
    add(store, "b")
    job = store.claim_next()
    assert job["id"] == a
    with store.conn() as c:
        assert c.execute("SELECT status, attempts FROM videos WHERE id = ?", (a,)).fetchone()[:] == ("processing", 1)
    assert store.claim_next()["id"] != a
    assert store.claim_next() is None
