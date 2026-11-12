import sys

import pytest

from olholimpo import cli


def test_unavailable_camera_releases_device_and_exits_with_message(monkeypatch, capsys):
    class Camera:
        released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr(sys, "argv", ["olholimpo", "webcam"])
    monkeypatch.setattr(cli, "Detector", lambda *args: object())
    monkeypatch.setattr(cli.cv2, "VideoCapture", lambda index: camera)
    monkeypatch.setattr(cli.cv2, "destroyAllWindows", lambda: None)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert camera.released
    assert "Câmera indisponível" in capsys.readouterr().err
