from truthlens.hashing import sha256_of


def test_sha256_of_known_content(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes(b"abc")

    assert sha256_of(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
