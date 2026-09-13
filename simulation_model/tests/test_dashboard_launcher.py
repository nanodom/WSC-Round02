import main


def test_open_dashboard_browser_uses_windows_url_handler(monkeypatch):
    opened = []
    monkeypatch.setattr(main.sys, "platform", "win32")
    monkeypatch.setattr(main.os, "startfile", opened.append, raising=False)

    main.open_dashboard_browser(main.DASHBOARD_URL)

    assert opened == [main.DASHBOARD_URL]
