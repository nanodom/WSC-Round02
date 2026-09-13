import io

import main


def test_tee_text_writer_writes_to_console_and_log():
    console = io.StringIO()
    log = io.StringIO()
    tee = main.TeeTextWriter(console, log)

    print("simulation progress", file=tee)
    tee.flush()

    assert console.getvalue() == "simulation progress\n"
    assert log.getvalue() == "simulation progress\n"


def test_main_creates_timestamped_simulation_log(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "LOGS_DIRECTORY", tmp_path)
    monkeypatch.setattr(main, "run_simulation", lambda: print("completed test run"))
    monkeypatch.setattr(main, "launch_dashboard", lambda: None)

    main.main()

    logs = list(tmp_path.glob("*_SimulationProgressResults.log"))
    assert len(logs) == 1
    assert "completed test run" in logs[0].read_text(encoding="utf-8")


def test_format_running_time():
    assert main._format_running_time(0) == "00:00:00"
    assert main._format_running_time(3661.9) == "01:01:01"
    assert main._format_running_time(25 * 3600 + 2) == "25:00:02"
